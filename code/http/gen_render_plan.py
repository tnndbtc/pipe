#!/usr/bin/env python3
# =============================================================================
# gen_render_plan.py — Build AssetManifest_final + RenderPlan (deterministic)
# =============================================================================
#
# Replaces the LLM-based Stage 9. Reads resolved media URIs and timing data
# from the merged manifest and assembles two downstream artifacts:
#
#   1. AssetManifest_final.{locale}.json  — envelope + resolved items
#   2. RenderPlan.{locale}.json           — per-shot render instructions
#
# Key design decisions vs old p_9.txt:
#   • VO timeline_in_ms / timeline_out_ms come from start_sec / end_sec
#     in AssetManifest_merged (measured by post_tts_analysis), NOT evenly
#     distributed approximations.
#   • duration_ms respects background_overrides for overflow shots.
#   • timing_lock_hash comes from AssetManifest_merged (locale-adjusted).
#   • duck_intervals / duck_db / fade_sec are passed into each shot's music
#     entry as extra fields (RenderedShot.additionalProperties allows this).
#
# Usage:
#   python gen_render_plan.py \
#       --manifest  projects/slug/ep/AssetManifest_merged.zh-Hans.json \
#       --media     projects/slug/ep/AssetManifest.media.zh-Hans.json
#
#   python gen_render_plan.py \
#       --manifest  ... \
#       --media     ... \
#       --shared    projects/slug/ep/AssetManifest_draft.shared.json \
#       --shotlist  projects/slug/ep/ShotList.json \
#       --profile   draft_720p \
#       --out-final projects/slug/ep/AssetManifest_final.zh-Hans.json \
#       --out-plan  projects/slug/ep/RenderPlan.zh-Hans.json
#
# Requirements: stdlib only (json, pathlib, argparse)
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

PRODUCER = "gen_render_plan.py"
RENDER_RESOLUTION = "1280x720"
RENDER_ASPECT    = "16:9"
RENDER_FPS       = 24
BASE_MUSIC_DB    = -6.0   # music un-ducked level (mirrors render_video.py)
INTER_LINE_PAUSE_MS = 300   # 0.3s gap between consecutive VO lines (matches post_tts_analysis)
VO_TAIL_MS          = 2000  # minimum silence after last spoken line before shot ends

# Formats where VO IS the shot — duration is capped to VO_end + VO_TAIL_MS.
# For these formats any silence gap beyond 2 s is jarring (no characters on
# screen, no lip-sync or acting to fill the gap — only narration + background).
NARRATIVE_FORMATS = {"continuous_narration", "documentary", "illustrated_narration", "ssml_narration"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Lookup helpers ────────────────────────────────────────────────────────────

def build_media_map(media: dict) -> dict[str, dict]:
    """
    Build media lookup from AssetManifest.media items.

    Per-shot entries (with shot_id) → key = "bg_id:shot_id"
    Background-level entries        → key = "bg_id"
    Multi-segment entries (with segment_index) → grouped in "_segments" dict
        keyed as "bg_id:shot_id" → [items sorted by segment_index]

    Used to check is_placeholder and fetch resolved URIs.
    """
    out: dict[str, dict] = {}
    segments: dict[str, list] = {}  # "bg_id:shot_id" → [items sorted by segment_index]
    for item in media.get("items", []):
        aid = item["asset_id"]
        if "segment_index" in item:
            key = f"{aid}:{item['shot_id']}"
            segments.setdefault(key, []).append(item)
        elif "shot_id" in item:
            out[f"{aid}:{item['shot_id']}"] = item
        else:
            out[aid] = item
    # Sort segments by index and store under special key
    for key, seg_list in segments.items():
        seg_list.sort(key=lambda x: x.get("segment_index", 0))
    if segments:
        out["_segments"] = segments  # type: ignore[assignment]
    return out


def load_tts_chunk_info(episode_dir: Path) -> dict[str, dict]:
    """Load TTS chunk info keyed by item_id for chunk-WAV deferred slicing.

    Reads gen_tts_cloud_results.json (per-item chunk offsets) and
    gen_tts_cloud_chunks.json (chunk → WAV path mapping) from the episode's
    assets/meta/ directory.

    Returns {item_id: {"chunk_wav": str, "start_sec": float, "end_sec": float}}
    for items that have a chunk WAV available.  Returns {} if files are absent.
    """
    meta_dir = episode_dir / "assets" / "meta"
    results_path = meta_dir / "gen_tts_cloud_results.json"
    chunks_path  = meta_dir / "gen_tts_cloud_chunks.json"

    if not results_path.exists() or not chunks_path.exists():
        return {}

    try:
        results = json.load(results_path.open(encoding="utf-8"))
        chunks  = json.load(chunks_path.open(encoding="utf-8"))
    except Exception:
        return {}

    # Build chunk_id → chunk_wav lookup
    chunk_wav_by_id: dict[int, str] = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        wav = ch.get("chunk_wav", "")
        if cid is not None and wav:
            from pathlib import Path as _Path
            if _Path(wav).exists():
                chunk_wav_by_id[cid] = wav

    # Build item_id → chunk info
    info: dict[str, dict] = {}
    for r in results:
        iid        = r.get("item_id", "")
        chunk_id   = r.get("source_chunk")
        start_sec  = r.get("chunk_offset_start_sec")
        end_sec    = r.get("chunk_offset_end_sec")
        if not iid or chunk_id is None or start_sec is None or end_sec is None:
            continue
        chunk_wav = chunk_wav_by_id.get(chunk_id, "")
        if not chunk_wav:
            continue
        info[iid] = {
            "chunk_wav": chunk_wav,
            "start_sec": float(start_sec),
            "end_sec":   float(end_sec),
        }

    return info


def build_vo_map(merged: dict) -> dict[str, dict]:
    """Build {item_id → vo_item} from merged manifest for timing lookups."""
    return {v["item_id"]: v for v in merged.get("vo_items", [])}


def build_music_map(merged: dict) -> dict[str, dict]:
    """Build {item_id → music_item} for duck_intervals lookup."""
    return {m["item_id"]: m for m in merged.get("music_items", [])}


def build_override_map(merged: dict) -> dict[str, float]:
    """Build {shot_id → overridden_duration_sec} from background_overrides."""
    return {
        o["shot_id"]: float(o["duration_sec"])
        for o in merged.get("background_overrides", [])
        if o.get("shot_id") and o.get("duration_sec") is not None
    }


def compute_duck_intervals_from_vo(
    vo_lines: list[dict],
    fade_ms:  int,
) -> list[list[float]]:
    """
    Recompute music duck intervals from shot-relative VO lines.

    For each VO line, the music is ducked from
        max(0, timeline_in_ms  - fade_ms)  to
              timeline_out_ms + fade_ms
    Overlapping intervals are merged.  Units: seconds (for compatibility with
    the AssetManifest_merged duck_intervals convention).
    """
    raw: list[tuple[float, float]] = []
    for vl in vo_lines:
        t0 = max(0.0, (vl["timeline_in_ms"]  - fade_ms) / 1000.0)
        t1 =          (vl["timeline_out_ms"] + fade_ms) / 1000.0
        raw.append((t0, t1))
    if not raw:
        return []
    sorted_ivs = sorted(raw, key=lambda x: x[0])
    merged: list[list[float]] = [list(sorted_ivs[0])]
    for t0, t1 in sorted_ivs[1:]:
        if t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [[round(a, 3), round(b, 3)] for a, b in merged]


# ── AssetManifest_final builder ───────────────────────────────────────────────

def build_final(
    merged:    dict,
    media:     dict,
    shared:    dict,
) -> dict:
    """
    Build AssetManifest_final.{locale}.json.

    items[] = all media items, except placeholder VO (those have no audio file).
    All other types (char, bg, sfx, music) are included even if placeholder so
    the renderer can log missing assets correctly.
    """
    items = [
        item for item in media.get("items", [])
        if not (item["asset_type"] == "vo" and item["is_placeholder"])
    ]

    return {
        "schema_id":      "AssetManifest_final",
        "schema_version": "1.0.0",
        "manifest_id":    merged.get("manifest_id", ""),
        "project_id":     shared.get("project_id", merged.get("project_id", "")),
        "shotlist_ref":   shared.get("shotlist_ref", merged.get("shotlist_ref", "")),
        "items":          items,
    }


# ── RenderPlan shot builder ───────────────────────────────────────────────────

def build_shot(
    shot:              dict,
    media_map:         dict[str, dict],
    vo_map:            dict[str, dict],
    music_map:         dict[str, dict],
    override_map:      dict[str, float],
    story_format:      str = "episodic",
    ref_dur_map:       dict | None = None,
    tts_chunk_info:    dict | None = None,
    scene_tails:       dict | None = None,
    sfx_plan_by_shot:  dict | None = None,
) -> dict:
    """
    Build one RenderedShot entry for RenderPlan.shots[].

    VO timeline is from real TTS audio: start_sec / end_sec × 1000.
    duration_ms is overridden for shots where locale VO ran longer.
    duck_intervals / duck_db / fade_sec are injected into the music entry
    (RenderedShot.additionalProperties is true so extra fields are allowed).
    """
    shot_id  = shot["shot_id"]
    scene_id = shot.get("scene_id", "")

    # duration_ms — base from ShotList, overridden for overflow shots
    base_dur = shot.get("duration_sec", 0)
    dur_sec  = override_map.get(shot_id, base_dur)
    duration_ms = round(dur_sec * 1000)

    # background_asset_id + background_media_type + background_segments
    # background_media_type tells the renderer whether to loop a still image
    # or play (and optionally loop) a video clip.
    # background_segments (v3): ordered list of media segments that fill the shot
    # duration via concat (no looping).
    bg_id = shot.get("background_id")
    bg_media = None
    bg_segments: list[dict] | None = None
    if bg_id:
        seg_key = f"{bg_id}:{shot_id}"
        seg_list = media_map.get("_segments", {}).get(seg_key)
        if seg_list:
            # Multi-segment mode (v3)
            bg_segments = []
            for seg_item in seg_list:
                uri = seg_item.get("uri", "")
                path_part = uri.split("://", 1)[-1] if "://" in uri else uri
                ext = Path(path_part).suffix.lower()
                media_type = "video" if ext in {".mp4", ".mov", ".webm", ".mkv"} else "image"
                seg_entry = {
                    "asset_id":       seg_item["asset_id"],
                    "uri":            uri,
                    "media_type":     media_type,
                    "duration_sec":   seg_item.get("duration_sec"),
                    "hold_sec":       seg_item.get("hold_sec"),
                    "animation_type": seg_item.get("animation_type"),  # e.g. zoom_in / pan_lr / ken_burns
                }
                # Clip trim range — start_sec / end_sec for video sub-clips.
                # When present, renderer seeks to start_sec and plays until end_sec.
                # duration_sec is recomputed as (end_sec - start_sec) for consistency.
                if seg_item.get("start_sec") is not None:
                    seg_entry["start_sec"] = seg_item["start_sec"]
                if seg_item.get("end_sec") is not None:
                    seg_entry["end_sec"] = seg_item["end_sec"]
                    # Recompute duration_sec from trim range (source of truth: start_sec + end_sec)
                    s = seg_entry.get("start_sec", 0)
                    seg_entry["duration_sec"] = seg_item["end_sec"] - s
                bg_segments.append(seg_entry)
            # Set primary bg from first segment (for backward compat fields)
            bg_media = seg_list[0]
        else:
            # Try shot-specific media first (per-shot selection), fall back to background-level
            bg_media = media_map.get(f"{bg_id}:{shot_id}") or media_map.get(bg_id)
    background_asset_id = bg_media["asset_id"] if bg_media else None

    # Warn when a shot expects a background but no media was resolved
    if bg_id and not bg_media:
        print(f"  ⚠ Shot {shot_id} has background_id={bg_id!r} but NO media resolved — will render BLACK")
    elif bg_id and bg_media and bg_media.get("is_placeholder", False):
        print(f"  ⚠ Shot {shot_id} has background_id={bg_id!r} but media is placeholder — will render BLACK")

    background_media_type: str | None = None
    if bg_media:
        uri = bg_media.get("uri", "")
        # Strip scheme (file://, placeholder://, http://) to get the path
        path_part = uri.split("://", 1)[-1] if "://" in uri else uri
        ext = Path(path_part).suffix.lower()
        if ext in {".mp4", ".mov", ".webm", ".mkv"}:
            background_media_type = "video"
        elif ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            background_media_type = "image"

    # character_asset_ids — ShotList character_id matches character_packs asset_id
    character_asset_ids: list[str] = []
    for char in shot.get("characters", []):
        cid = char.get("character_id")
        if cid and cid in media_map:
            character_asset_ids.append(media_map[cid]["asset_id"])

    # vo_lines — only non-placeholder VO with real timing
    #
    # post_tts_analysis.py may have processed all VO items as a single
    # episode-wide sequence (because locale manifest vo_items lack shot_id).
    # In that case start_sec/end_sec are episode-cumulative, not shot-relative.
    #
    # We compute shot-relative timing by using WAV duration (end_sec - start_sec)
    # and re-stacking items from cursor=0 within the shot, using the same
    # DEFAULT_PAUSE_SEC=0.3 inter-line pause that post_tts_analysis uses.
    # This gives correct timeline_in_ms / timeline_out_ms relative to shot start.
    #
    # If total VO exceeds duration_ms (overflow), we extend duration_ms so the
    # renderer knows the shot is longer than originally planned.

    audio_intent = shot.get("audio_intent", {})
    vo_lines: list[dict] = []
    cursor_ms = 0
    _seen_vo: set[tuple] = set()   # deduplicate by (speaker_id, text) within shot

    for vid in audio_intent.get("vo_item_ids", []):
        media_item = media_map.get(vid)
        if not media_item or media_item.get("is_placeholder", True):
            continue
        vo_item = vo_map.get(vid, {})
        start_sec = vo_item.get("start_sec")
        end_sec   = vo_item.get("end_sec")
        if start_sec is None or end_sec is None:
            continue

        # WAV duration of this clip
        wav_dur_ms = round((end_sec - start_sec) * 1000)
        if wav_dur_ms <= 0:
            continue

        # Skip duplicate VO lines (same speaker + same text) within the same shot.
        # The LLM sometimes assigns two IDs for the same line (e.g. -001 and -002
        # with identical text), causing the sentence to play twice in the render.
        speaker = vo_item.get("speaker_id", "")
        text    = vo_item.get("text", "").strip()
        dedup_key = (speaker, text)
        if dedup_key in _seen_vo:
            print(f"  [DEDUP] Skipping duplicate VO '{vid}' "
                  f"(speaker={speaker!r}, text={text[:50]!r}…)")
            continue
        _seen_vo.add(dedup_key)

        timeline_in_ms  = cursor_ms
        timeline_out_ms = cursor_ms + wav_dur_ms
        item_pause_ms = vo_item.get("pause_after_ms", INTER_LINE_PAUSE_MS)
        cursor_ms = timeline_out_ms + item_pause_ms

        vo_line: dict = {
            "line_id":         vid,
            "speaker_id":      speaker,
            "text":            text,
            "timeline_in_ms":  timeline_in_ms,
            "timeline_out_ms": timeline_out_ms,
        }
        # Phase 3, Step 10: chunk-WAV deferred slicing fields (optional)
        chunk_info = (tts_chunk_info or {}).get(vid)
        if chunk_info:
            from pathlib import Path as _Path
            chunk_wav = chunk_info["chunk_wav"]
            vo_line["audio_chunk_uri"]  = _Path(chunk_wav).as_uri()
            vo_line["audio_start_sec"]  = chunk_info["start_sec"]
            vo_line["audio_end_sec"]    = chunk_info["end_sec"]
        vo_lines.append(vo_line)

    # Ensure at least tail_ms of silence after the last spoken line.
    # tail_ms: per-scene override from manifest scene_tails dict, else VO_TAIL_MS default.
    # When VO overflows the ShotList estimate this also prevents the shot from
    # ending at the exact frame "Good night." finishes (zero-tail bug observed
    # in production: sh02/sh04/sh05 had duration_ms == last_vo_out_ms).
    tail_ms = VO_TAIL_MS
    if scene_tails:
        tail_ms = int(scene_tails.get(scene_id, scene_tails.get(shot_id, VO_TAIL_MS)))
    if vo_lines:
        vo_end_ms = vo_lines[-1]["timeline_out_ms"]
        duration_ms = max(duration_ms, vo_end_ms + tail_ms)

    # sfx_asset_ids — skip placeholder SFX
    sfx_asset_ids: list[str] = []
    for sid in audio_intent.get("sfx_item_ids", []):
        media_item = media_map.get(sid)
        if media_item and not media_item.get("is_placeholder", True):
            sfx_asset_ids.append(media_item["asset_id"])

    # music_asset_id + ducking fields
    #
    # duck_intervals are recomputed from the shot-relative vo_lines above.
    # The merged manifest's duck_intervals use episode-cumulative VO offsets
    # (because post_tts_analysis doesn't have per-shot timing context);
    # recomputing here gives correct shot-relative values.
    music_asset_id: str | None = None
    music_extra: dict = {}
    mid = audio_intent.get("music_item_id")
    if mid:
        music_media = media_map.get(mid)
        if music_media and not music_media.get("is_placeholder", True):
            music_asset_id = music_media["asset_id"]
            music_item = music_map.get(mid, {})
            # duck_db and fade_sec from the merged manifest (correct)
            duck_db  = music_item.get("duck_db",  -12.0)
            fade_sec = music_item.get("fade_sec",  0.15)
            base_db  = music_item.get("base_db",  BASE_MUSIC_DB)  # per-track vol offset
            fade_ms  = round(fade_sec * 1000)
            # Recompute shot-relative duck_intervals from the vo_lines we just built
            duck_ivs = compute_duck_intervals_from_vo(vo_lines, fade_ms)
            music_extra["duck_intervals"] = duck_ivs
            music_extra["duck_db"]        = duck_db
            music_extra["music_fade_sec"] = fade_sec
            music_extra["base_db"]        = base_db
            # start_sec: delay before music begins within the shot (from MusicPlan override)
            _music_start = music_item.get("start_sec", 0.0)
            if _music_start:
                music_extra["music_delay_sec"] = float(_music_start)

    # ── Dynamic ceiling for narrative formats (EXEC STEP 3 / Proposal 3) ────────
    #
    # For continuous_narration / documentary / illustrated_narration the VO IS
    # the shot.  There are no characters on screen, no lip-sync or acting to
    # fill a silence gap — only background + narration.  Any gap beyond the
    # 2-second tail buffer is immediately jarring to the listener.
    #
    # If Stage 4 estimated a duration much longer than the actual VO (common
    # when the LLM cannot know TTS rate/style at planning time), we cap it.
    #
    # The floor (VO_TAIL_MS) was already applied above, so for narrative formats
    # both floor and ceiling converge to  last_vo_out_ms + VO_TAIL_MS,
    # effectively pinning duration to exactly that value.
    #
    # For episodic / monologue we keep creative timing — only the floor applies.
    if story_format in NARRATIVE_FORMATS and vo_lines:
        last_vo_out_ms = vo_lines[-1]["timeline_out_ms"]
        ceiling_ms = last_vo_out_ms + tail_ms
        if duration_ms > ceiling_ms:
            print(f"  [CEILING] {shot_id}: {duration_ms} ms → {ceiling_ms} ms "
                  f"(VO ends {last_vo_out_ms} ms, format={story_format})")
            duration_ms = ceiling_ms
            # Cap duck_intervals so no endpoint exceeds the shortened duration.
            # duck_intervals are stored in seconds; compare against duration_sec.
            duration_sec = duration_ms / 1000.0
            for di in music_extra.get("duck_intervals", []):
                di[1] = min(di[1], duration_sec)   # cap end_sec
                di[0] = min(di[0], duration_sec)   # cap start_sec (in case of edge)
    # ── end ceiling logic ─────────────────────────────────────────────────────

    # ── EN reference floor (Phase 2 — Timeline Lock) ──────────────────────────
    #
    # When a reference plan (RenderPlan.en.json) is provided, floor the locale
    # shot duration to the EN shot duration.  This prevents ZH shots from being
    # shorter than their EN counterparts even after the convergence loop succeeds:
    # the video was edited to EN timing and cutting it short creates sync
    # problems with background / SFX / music tracks.
    #
    # Applied AFTER the narrative ceiling so that even pinned-short narration
    # shots are extended to match EN if the EN shot ran longer.
    if ref_dur_map:
        en_dur_ms = ref_dur_map.get(shot_id, 0)
        if en_dur_ms and duration_ms < en_dur_ms:
            print(f"  [EN-FLOOR] {shot_id}: {duration_ms} ms → {en_dur_ms} ms "
                  f"(floored to EN reference)")
            duration_ms = en_dur_ms
    # ── end EN reference floor ────────────────────────────────────────────────

    # sfx_plan_entries: user-selected SFX from SfxPlan.json (with timing)
    sfx_plan_entries = (sfx_plan_by_shot or {}).get(shot_id, [])

    rendered: dict = {
        "shot_id":                shot_id,
        "scene_id":               scene_id,
        "duration_ms":            duration_ms,
        "background_asset_id":    background_asset_id,
        "background_media_type":  background_media_type,   # "image" | "video" | None
        "background_segments":    bg_segments,              # list or None (v3 multi-segment)
        "character_asset_ids":    character_asset_ids,
        "vo_lines":               vo_lines,
        "sfx_asset_ids":          sfx_asset_ids,
        "sfx_plan_entries":       sfx_plan_entries,        # from SfxPlan.json
        "music_asset_id":         music_asset_id,
    }
    rendered.update(music_extra)
    return rendered


# ── RenderPlan builder ────────────────────────────────────────────────────────

def build_plan(
    merged:        dict,
    media:         dict,
    final:         dict,
    shotlist:      dict,
    profile:       str,
    story_format:  str = "episodic",
    ref_dur_map:   dict | None = None,
    episode_dir:   Path | None = None,
) -> dict:
    """Build the full RenderPlan document."""
    project_id = merged.get("project_id", "")
    episode_id = merged.get("episode_id", "")
    locale     = merged.get("locale", "")

    media_map    = build_media_map(media)
    vo_map       = build_vo_map(merged)
    music_map    = build_music_map(merged)
    override_map = build_override_map(merged)

    # Phase 3, Step 10: load TTS chunk info for deferred WAV slicing
    # Determines which vo_lines can reference chunk WAVs instead of sentence WAVs.
    tts_chunk_info: dict = {}
    if episode_dir is not None:
        tts_chunk_info = load_tts_chunk_info(episode_dir)
        if tts_chunk_info:
            print(f"  [CHUNK-DEFERRED] {len(tts_chunk_info)} items have chunk WAV references")

    # Load SfxPlan.json — user-selected SFX with timing from the SFX tab.
    # Keyed by shot_id so build_shot() can inject sfx_plan_entries per shot.
    sfx_plan_by_shot: dict[str, list] = {}
    if episode_dir is not None:
        sfx_plan_path = episode_dir / "assets" / "sfx" / "SfxPlan.json"
        if sfx_plan_path.exists():
            try:
                sfx_plan = json.load(sfx_plan_path.open(encoding="utf-8"))
                for entry in sfx_plan.get("sfx_entries", []):
                    sid = entry.get("shot_id", "")
                    if sid:
                        sfx_plan_by_shot.setdefault(sid, []).append(entry)
                n_sfx_entries = sum(len(v) for v in sfx_plan_by_shot.values())
                print(f"  [SFX-PLAN] {n_sfx_entries} entries across "
                      f"{len(sfx_plan_by_shot)} shot(s)")
            except Exception as e:
                print(f"  [WARN] Could not load SfxPlan.json: {e}", file=sys.stderr)

    # resolved_assets: flat view of AssetManifest_final.items[]
    resolved_assets = [
        {
            "asset_id":     item["asset_id"],
            "asset_type":   item["asset_type"],
            "uri":          item["uri"],
            "license_type": item["metadata"]["license_type"],
            "is_placeholder": item["is_placeholder"],
        }
        for item in final.get("items", [])
    ]

    # ── VO completeness guard ─────────────────────────────────────────────
    # Stage 4 (ShotList LLM) sometimes drops the last line of a scene from
    # vo_item_ids.  The merged manifest is authoritative (Stage 5 reads Script
    # directly), so any vo_item that has a shot_id but is not referenced by any
    # ShotList vo_item_ids list is silently injected here.
    shotlist_vo_ids: set[str] = {
        vid
        for shot in shotlist.get("shots", [])
        for vid in shot.get("audio_intent", {}).get("vo_item_ids", [])
    }
    shot_map: dict[str, dict] = {s["shot_id"]: s for s in shotlist.get("shots", [])}
    injected = 0
    for vo_item in merged.get("vo_items", []):
        vid = vo_item.get("item_id", "")
        sid = vo_item.get("shot_id", "")
        if vid and sid and vid not in shotlist_vo_ids and sid in shot_map:
            shot_map[sid].setdefault("audio_intent", {}).setdefault("vo_item_ids", []).append(vid)
            shotlist_vo_ids.add(vid)
            injected += 1
            print(f"  [VO-INJECT] {vid} → {sid} (missing from ShotList, added from manifest)")
    if injected:
        print(f"  [VO-INJECT] {injected} item(s) injected — ShotList vo_item_ids were incomplete")
    # ── end VO completeness guard ──────────────────────────────────────────

    # shots: one RenderedShot per ShotList shot
    scene_tails = merged.get("scene_tails", {})
    shots = [
        build_shot(shot, media_map, vo_map, music_map, override_map,
                   story_format, ref_dur_map or {}, tts_chunk_info,
                   scene_tails, sfx_plan_by_shot)
        for shot in shotlist.get("shots", [])
    ]

    # ── Background coverage summary ──────────────────────────────────────
    black_shots = [s for s in shots if s.get("background_asset_id") is None]
    if black_shots:
        print(f"\n  ⚠ WARNING: {len(black_shots)} shot(s) have NO background media — will render BLACK:")
        for bs in black_shots:
            print(f"    • {bs['shot_id']} (duration {bs['duration_ms']}ms)")
        print()
    # ── end background coverage summary ──────────────────────────────────

    # plan_id includes locale so per-locale plans don't collide
    plan_id = f"plan-{project_id}-{episode_id}"
    if locale:
        plan_id += f"-{locale}"

    return {
        "schema_id":         "RenderPlan",
        "schema_version":    "1.0.0",
        "plan_id":           plan_id,
        "project_id":        project_id,
        "manifest_ref":      final.get("manifest_id", ""),
        "timing_lock_hash":  merged.get("timing_lock_hash", ""),
        "profile":           profile,
        "resolution":        RENDER_RESOLUTION,
        "aspect_ratio":      RENDER_ASPECT,
        "fps":               RENDER_FPS,
        "resolved_assets":   resolved_assets,
        "shots":             shots,
    }


# ── Auto-detect helpers ───────────────────────────────────────────────────────

def find_shared(episode_dir: Path) -> Path | None:
    candidate = episode_dir / "AssetManifest_draft.shared.json"
    return candidate if candidate.exists() else None


def find_shotlist(episode_dir: Path) -> Path | None:
    candidate = episode_dir / "ShotList.json"
    return candidate if candidate.exists() else None


# ── Output paths ──────────────────────────────────────────────────────────────

def derive_final_path(episode_dir: Path, locale: str) -> Path:
    return episode_dir / f"AssetManifest_final.{locale}.json"


def derive_plan_path(episode_dir: Path, locale: str) -> Path:
    return episode_dir / f"RenderPlan.{locale}.json"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build AssetManifest_final.{locale}.json + RenderPlan.{locale}.json "
                    "from merged manifest and resolved media.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="AssetManifest_merged.{locale}.json (locale_scope='merged').")
    p.add_argument("--media",    required=True, metavar="PATH",
                   help="AssetManifest.media.{locale}.json from resolve_assets.py.")
    p.add_argument("--shared",   default=None, metavar="PATH",
                   help="AssetManifest_draft.shared.json. "
                        "Default: auto-detect from episode directory.")
    p.add_argument("--shotlist", default=None, metavar="PATH",
                   help="ShotList.json. Default: auto-detect from episode directory.")
    p.add_argument("--profile",  default="preview_local",
                   choices=["preview_local", "draft_720p", "final_1080p"],
                   help="Render profile (default: preview_local).")
    p.add_argument("--out-final", default=None, metavar="PATH",
                   help="Output for AssetManifest_final. "
                        "Default: AssetManifest_final.{locale}.json in episode dir.")
    p.add_argument("--out-plan",  default=None, metavar="PATH",
                   help="Output for RenderPlan. "
                        "Default: RenderPlan.{locale}.json in episode dir.")
    p.add_argument("--story-format", default="episodic",
                   choices=["episodic", "continuous_narration", "illustrated_narration",
                            "documentary", "monologue", "ssml_narration"],
                   help="Story format from pipeline_vars.sh (default: episodic). "
                        "Narrative formats apply a shot duration ceiling = "
                        "last_vo_out_ms + 2000 ms to prevent silence gaps.")
    p.add_argument("--reference-plan", default=None, metavar="PATH",
                   help="RenderPlan.en.json from the EN locale pass. "
                        "When supplied, each shot's duration_ms is floored to "
                        "the EN shot duration so locale VO never under-runs its "
                        "EN counterpart (Phase 2 — Timeline Lock).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    media_path    = Path(args.media).resolve()

    for label, path in [("--manifest", manifest_path), ("--media", media_path)]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    merged = load_json(manifest_path)
    media  = load_json(media_path)

    # Guard: must be merged manifest
    locale_scope = merged.get("locale_scope")
    if locale_scope != "merged":
        raise SystemExit(
            f"[ERROR] --manifest has locale_scope='{locale_scope}'. "
            "Expected 'merged'."
        )

    locale     = merged.get("locale", "")
    episode_dir = manifest_path.parent

    # Resolve --shared
    if args.shared:
        shared_path = Path(args.shared).resolve()
    else:
        shared_path = find_shared(episode_dir)
        if not shared_path:
            raise SystemExit(
                "[ERROR] Could not find AssetManifest_draft.shared.json in episode dir. "
                "Pass --shared explicitly."
            )
    shared = load_json(shared_path)

    # Resolve --shotlist
    if args.shotlist:
        shotlist_path = Path(args.shotlist).resolve()
    else:
        shotlist_path = find_shotlist(episode_dir)
        if not shotlist_path:
            raise SystemExit(
                "[ERROR] Could not find ShotList.json in episode dir. "
                "Pass --shotlist explicitly."
            )
    shotlist = load_json(shotlist_path)

    # Derive output paths
    out_final = Path(args.out_final).resolve() if args.out_final \
                else derive_final_path(episode_dir, locale)
    out_plan  = Path(args.out_plan).resolve()  if args.out_plan \
                else derive_plan_path(episode_dir, locale)

    # Load optional EN reference plan for Timeline Lock (Phase 2)
    ref_dur_map: dict[str, int] = {}
    if args.reference_plan:
        ref_plan_path = Path(args.reference_plan).resolve()
        if ref_plan_path.exists():
            ref_plan = load_json(ref_plan_path)
            ref_dur_map = {s["shot_id"]: s["duration_ms"]
                           for s in ref_plan.get("shots", [])
                           if "shot_id" in s and "duration_ms" in s}
        else:
            print(f"  [WARN] --reference-plan not found: {ref_plan_path}", file=sys.stderr)

    print("=" * 60)
    print("  gen_render_plan")
    print(f"  Manifest    : {manifest_path.name}")
    print(f"  Media       : {media_path.name}")
    print(f"  Shared      : {shared_path.name}")
    print(f"  ShotList    : {shotlist_path.name}")
    print(f"  Locale      : {locale}")
    print(f"  Profile     : {args.profile}")
    print(f"  Format      : {args.story_format}")
    if ref_dur_map:
        print(f"  Ref-plan    : {Path(args.reference_plan).name}  ({len(ref_dur_map)} EN shots)")
    print(f"  Out-final   : {out_final.name}")
    print(f"  Out-plan    : {out_plan.name}")
    print("=" * 60)

    # Build AssetManifest_final
    final = build_final(merged, media, shared)
    save_json(final, out_final)
    n_final = len(final["items"])
    n_placeholder_final = sum(1 for i in final["items"] if i["is_placeholder"])
    print(f"  AssetManifest_final  : {n_final} items  ({n_placeholder_final} placeholders)")

    # Build RenderPlan
    plan = build_plan(merged, media, final, shotlist, args.profile, args.story_format,
                      ref_dur_map or None, episode_dir=episode_dir)
    save_json(plan, out_plan)

    n_shots   = len(plan["shots"])
    n_vo_lines = sum(len(s["vo_lines"]) for s in plan["shots"])
    n_overflow = sum(
        1 for s in plan["shots"]
        if s["shot_id"] in {o["shot_id"] for o in merged.get("background_overrides", [])}
    )
    n_with_music = sum(1 for s in plan["shots"] if s.get("music_asset_id"))
    n_with_duck  = sum(1 for s in plan["shots"] if s.get("duck_intervals"))
    # Count shots where the ceiling was applied (duration < ShotList estimate)
    shotlist_dur_map = {s["shot_id"]: round(s.get("duration_sec", 0) * 1000)
                        for s in shotlist.get("shots", [])}
    n_ceiling = sum(
        1 for s in plan["shots"]
        if s["duration_ms"] < shotlist_dur_map.get(s["shot_id"], s["duration_ms"])
    )
    n_en_floor = sum(
        1 for s in plan["shots"]
        if ref_dur_map.get(s["shot_id"], 0) and
           s["duration_ms"] == ref_dur_map.get(s["shot_id"], 0) and
           s["duration_ms"] > shotlist_dur_map.get(s["shot_id"], 0)
    ) if ref_dur_map else 0

    n_bg_video = sum(1 for s in plan["shots"] if s.get("background_media_type") == "video")
    n_bg_image = sum(1 for s in plan["shots"] if s.get("background_media_type") == "image")
    n_bg_multi = sum(1 for s in plan["shots"]
                     if s.get("background_segments") and len(s["background_segments"]) > 1)

    print(f"  RenderPlan shots     : {n_shots}")
    print(f"  BG type — video      : {n_bg_video}  image: {n_bg_image}  multi-segment: {n_bg_multi}")
    print(f"  VO lines             : {n_vo_lines}")
    print(f"  Overflow shots       : {n_overflow}")
    print(f"  Ceiling applied      : {n_ceiling} shots  "
          f"({'narrative ceiling active' if args.story_format in NARRATIVE_FORMATS else 'n/a — episodic/monologue'})")
    if ref_dur_map:
        print(f"  EN-floor applied     : {n_en_floor} shots  (timeline lock active)")
    print(f"  Shots with music     : {n_with_music}")
    print(f"  Shots with ducking   : {n_with_duck}")
    print(f"  Timing lock hash     : {plan['timing_lock_hash'][:16]}…")

    print(f"\n  [OK] {out_final.name}")
    print(f"  [OK] {out_plan.name}")


if __name__ == "__main__":
    main()
