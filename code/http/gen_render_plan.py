#!/usr/bin/env python3
# =============================================================================
# gen_render_plan.py — Build RenderPlan (deterministic)
# =============================================================================
#
# Replaces the LLM-based Stage 9. Reads resolved media URIs and timing data
# from the unified manifest and assembles the downstream artifact:
#
#   1. RenderPlan.{locale}.json           — per-shot render instructions
#
# Key design decisions vs old p_9.txt:
#   • VO timing (start_sec / end_sec) is read directly from VOPlan.{locale}.json
#     vo_items[] (written authoritatively at VO Approve). No vo_lines are written
#     to RenderPlan — consumers read AssetManifest directly.
#   • duration_ms respects background_overrides for overflow shots.
#   • timing_lock_hash comes from AssetManifest (locale-adjusted).
#   • duck_intervals / duck_db / fade_sec are passed into each shot's music
#     entry as extra fields (RenderedShot.additionalProperties allows this).
#
# Usage:
#   python gen_render_plan.py \
#       --manifest  projects/slug/ep/VOPlan.zh-Hans.json
#
#   python gen_render_plan.py \
#       --manifest  projects/slug/ep/VOPlan.zh-Hans.json \
#       --shared    projects/slug/ep/AssetManifest.shared.json \
#       --shotlist  projects/slug/ep/ShotList.json \
#       --profile   draft_720p \
#       --out-plan  projects/slug/ep/RenderPlan.zh-Hans.json
#
# Requirements: stdlib only (json, os, pathlib, argparse)
# =============================================================================

import argparse
import json
import os
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
    Build media lookup from resolved_assets items.

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



def load_vo_approved_timing(manifest_path: Path) -> dict[str, dict]:
    """Load approved VO timing from VOPlan.{locale}.json vo_items[].

    Returns {item_id → item_dict} where each item_dict has:
        duration_sec, start_sec, end_sec

    Returns {} if the file does not exist or has no vo_approval block.
    """
    if not manifest_path.exists():
        return {}
    try:
        doc = load_json(manifest_path)
        if not doc.get("vo_approval", {}).get("approved_at"):
            return {}
        return {item["item_id"]: item for item in doc.get("vo_items", [])}
    except Exception as exc:
        print(f"  [WARN] Could not load {manifest_path.name}: {exc}", file=sys.stderr)
        return {}


def build_override_map(merged: dict) -> dict[str, float]:
    """Build {shot_id → overridden_duration_sec} from background_overrides."""
    return {
        o["shot_id"]: float(o["duration_sec"])
        for o in merged.get("background_overrides", [])
        if o.get("shot_id") and o.get("duration_sec") is not None
    }


def compute_duck_intervals_from_vo(
    vo_items: list[dict],
    fade_ms: int,
    shot_start_sec: float = 0.0,
) -> list[list[float]]:
    """Compute music duck intervals from AssetManifest vo_items timing.

    vo_items: VO items with start_sec / end_sec (episode-cumulative).
    fade_ms:  fade padding in milliseconds.
    shot_start_sec: episode-cumulative start of the shot (from ShotList).
    """
    raw: list[tuple[float, float]] = []
    for vi in vo_items:
        start_sec = vi.get("start_sec")
        end_sec   = vi.get("end_sec")
        if start_sec is None or end_sec is None:
            continue
        rel_in_ms  = round((start_sec - shot_start_sec) * 1000)
        rel_out_ms = round((end_sec   - shot_start_sec) * 1000)
        t0 = max(0.0, (rel_in_ms  - fade_ms) / 1000.0)
        t1 =          (rel_out_ms + fade_ms) / 1000.0
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


# ── RenderPlan shot builder ───────────────────────────────────────────────────

def build_shot(
    shot:              dict,
    media_map:         dict[str, dict],
    vo_map:            dict[str, dict],
    override_map:      dict[str, float],
    story_format:      str = "episodic",
    ref_dur_map:       dict | None = None,
    tts_chunk_info:    dict | None = None,
    scene_tails:       dict | None = None,
    sfx_plan_segments: list | None = None,
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

    # VO timing: read start_sec / end_sec directly from VOPlan.{locale}.json
    # vo_items[] (written authoritatively at VO Approve). Compute shot-relative
    # last_vo_out_ms needed for duration ceiling/floor and duck interval computation.
    # vo_lines are NOT written to RenderPlan — consumers read AssetManifest directly.

    audio_intent = shot.get("audio_intent", {})
    _seen_vo: set[tuple] = set()   # deduplicate by (speaker_id, text) within shot
    _vo_items_for_shot: list[dict] = []  # timing-valid, deduplicated items for this shot
    last_vo_out_ms = 0
    shot_start_sec = shot.get("start_sec", 0.0)

    for vid in audio_intent.get("vo_item_ids", []):
        # VO WAV files are NOT in resolved_assets — do not gate on media_map.
        # The only valid skip condition is missing start_sec/end_sec (checked below).
        vo_item = vo_map.get(vid, {})
        start_sec = vo_item.get("start_sec")
        end_sec   = vo_item.get("end_sec")
        if start_sec is None or end_sec is None:
            continue

        wav_dur_ms = round((end_sec - start_sec) * 1000)
        if wav_dur_ms <= 0:
            continue

        # Skip duplicate VO lines (same speaker + same text) within the same shot.
        speaker = vo_item.get("speaker_id", "")
        text    = vo_item.get("text", "").strip()
        dedup_key = (speaker, text)
        if dedup_key in _seen_vo:
            print(f"  [DEDUP] Skipping duplicate VO '{vid}' "
                  f"(speaker={speaker!r}, text={text[:50]!r}…)")
            continue
        _seen_vo.add(dedup_key)

        # Shot-relative end time (episode-cumulative end_sec minus shot start)
        rel_out_ms = round((end_sec - shot_start_sec) * 1000)
        last_vo_out_ms = max(last_vo_out_ms, rel_out_ms)
        _vo_items_for_shot.append(vo_item)

    # Derive per-shot tail from ShotList duration_sec (set by patch_shotlist_durations.py
    # from AssetManifest vo_items).  This correctly reflects the actual inter-scene gap
    # (e.g. 3300 ms for sc01→sc02, 2300 ms for subsequent boundaries) rather than
    # the old fixed VO_TAIL_MS=2000 ms assumption which under-counted by up to 1300 ms
    # per boundary and caused the narrative-format CEILING to clip shots incorrectly.
    #
    # Formula: tail_ms = shot.duration_sec × 1000 − last_vo_out_ms
    # (shot.duration_sec already = vo_span + approved_tail from the patcher)
    # Falls back to VO_TAIL_MS (2 s) when shot.duration_sec is absent or stale.
    tail_ms = VO_TAIL_MS  # safety fallback
    if _vo_items_for_shot and shot.get("duration_sec"):
        shot_dur_ms  = round(shot["duration_sec"] * 1000)
        derived_tail = shot_dur_ms - last_vo_out_ms
        if derived_tail > 0:
            tail_ms = derived_tail
    # Explicit scene_tails override from merged manifest (highest priority)
    if scene_tails:
        tail_ms = int(scene_tails.get(scene_id, scene_tails.get(shot_id, tail_ms)))
    if _vo_items_for_shot:
        duration_ms = max(duration_ms, last_vo_out_ms + tail_ms)

    # sfx_asset_ids — populated from SfxPlan only (sfx_item_ids no longer used)
    sfx_asset_ids: list[str] = []

    # music_asset_id — populated from MusicPlan only (music_item_id no longer used)
    music_asset_id: str | None = None
    music_extra: dict = {}

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
    if story_format in NARRATIVE_FORMATS and _vo_items_for_shot:
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
    # v2 schema: sfx_entries[] are episode-absolute; pass them all through unchanged.
    # render_video.py filters them to the correct window using start_sec/end_sec.
    sfx_plan_entries = list(sfx_plan_segments or [])

    rendered: dict = {
        "shot_id":                shot_id,
        "scene_id":               scene_id,
        "duration_ms":            duration_ms,
        "background_asset_id":    background_asset_id,
        "background_media_type":  background_media_type,   # "image" | "video" | None
        "background_segments":    bg_segments,              # list or None (v3 multi-segment)
        "character_asset_ids":    character_asset_ids,
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
    override_map = build_override_map(merged)

    # Phase 3, Step 10: load TTS chunk info (retained for future use / logging).
    tts_chunk_info: dict = {}
    if episode_dir is not None:
        tts_chunk_info = load_tts_chunk_info(episode_dir)
        if tts_chunk_info:
            print(f"  [CHUNK-DEFERRED] {len(tts_chunk_info)} items have chunk WAV references")

    # Load SfxPlan.json — user-selected SFX with episode-absolute timing.
    # v2 schema: sfx_entries[] each carries start_sec/end_sec/source_file directly.
    # No shot_id / item_id — segments are placed by absolute time, not by shot.
    sfx_plan_segments: list[dict] = []
    if episode_dir is not None:
        sfx_plan_path = episode_dir / "SfxPlan.json"
        if sfx_plan_path.exists():
            try:
                sfx_plan = json.load(sfx_plan_path.open(encoding="utf-8"))
                sfx_plan_segments = sfx_plan.get("shot_overrides", [])
                print(f"  [SFX-PLAN] {len(sfx_plan_segments)} segment(s) loaded")
            except Exception as e:
                print(f"  [WARN] Could not load SfxPlan.json: {e}", file=sys.stderr)

    # resolved_assets: flat view of VOPlan.{locale}.json resolved_assets[]
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
        build_shot(shot, media_map, vo_map, override_map,
                   story_format, ref_dur_map or {}, tts_chunk_info,
                   scene_tails, sfx_plan_segments)
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
        "manifest_ref":      merged.get("manifest_id", ""),
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
    candidate = episode_dir / "AssetManifest.shared.json"
    return candidate if candidate.exists() else None


def find_shotlist(episode_dir: Path) -> Path | None:
    candidate = episode_dir / "ShotList.json"
    return candidate if candidate.exists() else None


# ── Output paths ──────────────────────────────────────────────────────────────

def derive_plan_path(episode_dir: Path, locale: str) -> Path:
    return episode_dir / f"RenderPlan.{locale}.json"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add render_plan section to VOPlan.{locale}.json and write "
                    "RenderPlan.{locale}.json from the unified manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="VOPlan.{locale}.json (unified, locale_scope='merged').")
    p.add_argument("--shared",   default=None, metavar="PATH",
                   help="AssetManifest.shared.json. "
                        "Default: auto-detect from episode directory.")
    p.add_argument("--shotlist", default=None, metavar="PATH",
                   help="ShotList.json. Default: auto-detect from episode directory.")
    p.add_argument("--profile",  default="preview_local",
                   choices=["preview_local", "draft_720p", "final_1080p"],
                   help="Render profile (default: preview_local).")
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

    if not manifest_path.exists():
        print(f"[ERROR] --manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    unified = load_json(manifest_path)
    merged  = unified  # same object
    _media_shim = {"items": unified.get("resolved_assets", [])}

    # ── Apply AssetManifest vo_approval as authoritative timing source ──────────
    # If the user has reviewed and approved VO timing (Stage 3.5 or Stage 8.5),
    # the manifest has a vo_approval block. The timing already lives in vo_items[],
    # so we just confirm the approval is present and log it.
    locale_for_approval = merged.get("locale", "")
    if locale_for_approval:
        approved_map = load_vo_approved_timing(manifest_path)
        if approved_map:
            patched = 0
            for vo_item in merged.get("vo_items", []):
                vid = vo_item.get("item_id", "")
                if vid in approved_map:
                    ap = approved_map[vid]
                    if "duration_sec" in ap:
                        dur = float(ap["duration_sec"])
                        vo_item["start_sec"] = float(ap.get("start_sec", vo_item.get("start_sec", 0.0)))
                        vo_item["end_sec"]   = float(ap.get("end_sec",   vo_item.get("end_sec",   dur)))
                        patched += 1
            if patched:
                print(f"  [VO-APPROVAL] Applied approved timing to {patched} vo_items "
                      f"from VOPlan.{locale_for_approval}.json")
    # ── end vo_approval override ──────────────────────────────────────────────

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
                "[ERROR] Could not find AssetManifest.shared.json in episode dir. "
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
    print(f"  Shared      : {shared_path.name}")
    print(f"  ShotList    : {shotlist_path.name}")
    print(f"  Locale      : {locale}")
    print(f"  Profile     : {args.profile}")
    print(f"  Format      : {args.story_format}")
    if ref_dur_map:
        print(f"  Ref-plan    : {Path(args.reference_plan).name}  ({len(ref_dur_map)} EN shots)")
    _has_approval = bool(merged.get("vo_approval", {}).get("approved_at"))
    if _has_approval:
        print(f"  VO-Approval : VOPlan.{locale}.json → vo_approval  ✓ (timing override active)")
    else:
        print(f"  VO-Approval : not found (using post_tts_analysis timing)")
    print(f"  Out-plan    : {out_plan.name}")
    print("=" * 60)

    # Build RenderPlan
    plan = build_plan(merged, _media_shim, _media_shim, shotlist, args.profile, args.story_format,
                      ref_dur_map or None, episode_dir=episode_dir)

    # Write render_plan back into the unified manifest (atomic replace)
    unified["render_plan"] = plan
    _tmp = str(manifest_path) + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(_tmp, str(manifest_path))

    save_json(plan, out_plan)

    n_shots   = len(plan["shots"])
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
    print(f"  Overflow shots       : {n_overflow}")
    print(f"  Ceiling applied      : {n_ceiling} shots  "
          f"({'narrative ceiling active' if args.story_format in NARRATIVE_FORMATS else 'n/a — episodic/monologue'})")
    if ref_dur_map:
        print(f"  EN-floor applied     : {n_en_floor} shots  (timeline lock active)")
    print(f"  Shots with music     : {n_with_music}")
    print(f"  Shots with ducking   : {n_with_duck}")
    print(f"  Timing lock hash     : {plan['timing_lock_hash'][:16]}…")

    print(f"\n  [OK] render_plan written to {manifest_path.name}")
    print(f"  [OK] {out_plan.name}")


if __name__ == "__main__":
    main()
