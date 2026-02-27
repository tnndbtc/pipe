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
INTER_LINE_PAUSE_MS = 300   # 0.3s gap between consecutive VO lines (matches post_tts_analysis)


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
    Build {asset_id → ResolvedAsset} from AssetManifest.media items.
    Used to check is_placeholder and fetch resolved URIs.
    """
    return {item["asset_id"]: item for item in media.get("items", [])}


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
    shot:         dict,
    media_map:    dict[str, dict],
    vo_map:       dict[str, dict],
    music_map:    dict[str, dict],
    override_map: dict[str, float],
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

    # background_asset_id — direct match (ShotList.background_id == media asset_id)
    bg_id = shot.get("background_id")
    bg_media = media_map.get(bg_id) if bg_id else None
    background_asset_id = bg_media["asset_id"] if bg_media else None

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
        cursor_ms = timeline_out_ms + INTER_LINE_PAUSE_MS

        vo_lines.append({
            "line_id":         vid,
            "speaker_id":      speaker,
            "text":            text,
            "timeline_in_ms":  timeline_in_ms,
            "timeline_out_ms": timeline_out_ms,
        })

    # If VO overruns the shot, extend duration_ms to fit
    if vo_lines:
        vo_end_ms = vo_lines[-1]["timeline_out_ms"]
        if vo_end_ms > duration_ms:
            duration_ms = vo_end_ms

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
            fade_ms  = round(fade_sec * 1000)
            # Recompute shot-relative duck_intervals from the vo_lines we just built
            duck_ivs = compute_duck_intervals_from_vo(vo_lines, fade_ms)
            music_extra["duck_intervals"] = duck_ivs
            music_extra["duck_db"]        = duck_db
            music_extra["music_fade_sec"] = fade_sec

    rendered: dict = {
        "shot_id":              shot_id,
        "scene_id":             scene_id,
        "duration_ms":          duration_ms,
        "background_asset_id":  background_asset_id,
        "character_asset_ids":  character_asset_ids,
        "vo_lines":             vo_lines,
        "sfx_asset_ids":        sfx_asset_ids,
        "music_asset_id":       music_asset_id,
    }
    rendered.update(music_extra)
    return rendered


# ── RenderPlan builder ────────────────────────────────────────────────────────

def build_plan(
    merged:    dict,
    media:     dict,
    final:     dict,
    shotlist:  dict,
    profile:   str,
) -> dict:
    """Build the full RenderPlan document."""
    project_id = merged.get("project_id", "")
    episode_id = merged.get("episode_id", "")
    locale     = merged.get("locale", "")

    media_map    = build_media_map(media)
    vo_map       = build_vo_map(merged)
    music_map    = build_music_map(merged)
    override_map = build_override_map(merged)

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

    # shots: one RenderedShot per ShotList shot
    shots = [
        build_shot(shot, media_map, vo_map, music_map, override_map)
        for shot in shotlist.get("shots", [])
    ]

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

    print("=" * 60)
    print("  gen_render_plan")
    print(f"  Manifest    : {manifest_path.name}")
    print(f"  Media       : {media_path.name}")
    print(f"  Shared      : {shared_path.name}")
    print(f"  ShotList    : {shotlist_path.name}")
    print(f"  Locale      : {locale}")
    print(f"  Profile     : {args.profile}")
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
    plan = build_plan(merged, media, final, shotlist, args.profile)
    save_json(plan, out_plan)

    n_shots   = len(plan["shots"])
    n_vo_lines = sum(len(s["vo_lines"]) for s in plan["shots"])
    n_overflow = sum(
        1 for s in plan["shots"]
        if s["shot_id"] in {o["shot_id"] for o in merged.get("background_overrides", [])}
    )
    n_with_music = sum(1 for s in plan["shots"] if s.get("music_asset_id"))
    n_with_duck  = sum(1 for s in plan["shots"] if s.get("duck_intervals"))

    print(f"  RenderPlan shots     : {n_shots}")
    print(f"  VO lines             : {n_vo_lines}")
    print(f"  Overflow shots       : {n_overflow}")
    print(f"  Shots with music     : {n_with_music}")
    print(f"  Shots with ducking   : {n_with_duck}")
    print(f"  Timing lock hash     : {plan['timing_lock_hash'][:16]}…")

    print(f"\n  [OK] {out_final.name}")
    print(f"  [OK] {out_plan.name}")


if __name__ == "__main__":
    main()
