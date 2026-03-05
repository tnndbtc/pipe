#!/usr/bin/env python3
# =============================================================================
# apply_music_plan.py — Apply MusicPlan overrides to the music pipeline
# =============================================================================
#
# Reads a MusicPlan.json (validated against MusicPlan.v1.json schema) and:
#   1. For each loop_selection: extracts a segment from the source track,
#      concatenates N copies with crossfade, writes {stem}.loop.wav
#   2. For each shot_override: updates matching music_items[] fields in the
#      merged locale manifest
#
# Usage:
#   python apply_music_plan.py \
#       --plan projects/slug/episodes/ep/MusicPlan.json \
#       --manifest projects/slug/episodes/ep/AssetManifest_merged.en.json
#
#   python apply_music_plan.py --plan ... --manifest ... \
#       --resources projects/resources/music/
#
# Requirements:
#   pip install librosa soundfile numpy jsonschema
# =============================================================================

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PIPE_DIR = Path(__file__).resolve().parent.parent.parent
SR = 48_000
SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg"}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_48k(path: Path) -> np.ndarray:
    """Load any audio file as mono float32 at 48 kHz."""
    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── License sidecar ──────────────────────────────────────────────────────────

def write_license_sidecar(path: Path) -> None:
    """Write a CC0 license sidecar JSON alongside a generated audio file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"spdx_id": "CC0", "attribution_required": False, "text": ""},
            f,
            indent=2,
        )
        f.write("\n")


# ── Schema validation ────────────────────────────────────────────────────────

def validate_plan(plan: dict) -> None:
    """Validate MusicPlan against its JSON Schema. Exits on failure."""
    import jsonschema

    schema_path = PIPE_DIR / "contracts" / "schemas" / "MusicPlan.v1.json"
    if not schema_path.exists():
        print(f"[ERROR] Schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=plan, schema=schema)
    except jsonschema.ValidationError as exc:
        print(f"[ERROR] MusicPlan schema validation failed:\n  {exc.message}",
              file=sys.stderr)
        if exc.absolute_path:
            print(f"  Path: {'.'.join(str(p) for p in exc.absolute_path)}",
                  file=sys.stderr)
        sys.exit(1)

    print("[OK] MusicPlan passes schema validation.")


# ── Loop WAV generation ──────────────────────────────────────────────────────

def find_source_track(stem: str, resources_dir: Path) -> Path | None:
    """Find a source track matching the given stem in resources_dir."""
    for ext in sorted(SUPPORTED_EXTS):
        candidate = resources_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def generate_loop_wav(
    stem: str,
    selection: dict,
    resources_dir: Path,
    out_dir: Path,
    max_shot_duration: float,
) -> Path | None:
    """
    Generate a looped WAV from a source track segment.

    1. Load source track at 48kHz mono
    2. Extract segment [start_sec, start_sec + duration_sec]
    3. Determine N = ceil(max_shot_duration / duration_sec) + 1
    4. Concatenate N copies with crossfade_ms overlap at each join
    5. Write to out_dir/{stem}.loop.wav (48kHz 16bit mono)
    """
    import soundfile as sf

    source_path = find_source_track(stem, resources_dir)
    if source_path is None:
        print(f"  [ERROR] Source track not found for stem '{stem}' "
              f"in {resources_dir}", file=sys.stderr)
        return None

    start_sec    = selection["start_sec"]
    duration_sec = selection["duration_sec"]
    crossfade_ms = selection.get("crossfade_ms", 100)
    mode         = selection.get("mode", "loop")

    print(f"  Source   : {source_path.name}")
    print(f"  Segment  : {start_sec}s – {start_sec + duration_sec}s")
    print(f"  Mode     : {mode}")
    print(f"  Crossfade: {crossfade_ms}ms")

    # Load and extract segment
    audio = load_audio_48k(source_path)
    start_sample = int(start_sec * SR)
    end_sample   = int((start_sec + duration_sec) * SR)
    segment      = audio[start_sample:end_sample]

    if len(segment) == 0:
        print(f"  [ERROR] Empty segment — start_sec={start_sec} exceeds "
              f"track length {len(audio)/SR:.1f}s", file=sys.stderr)
        return None

    if mode == "one_shot":
        # Single copy, no looping
        out_audio = segment
    else:
        # Loop mode: concatenate N copies with crossfade
        n_copies = math.ceil(max_shot_duration / duration_sec) + 1
        crossfade_samples = int(crossfade_ms / 1000 * SR)
        crossfade_samples = min(crossfade_samples, len(segment) // 2)

        print(f"  Copies   : {n_copies}  "
              f"(covers {n_copies * duration_sec:.1f}s)")

        if n_copies <= 1 or crossfade_samples == 0:
            out_audio = segment
        else:
            # Build the concatenation with crossfade overlaps
            seg_len   = len(segment)
            step      = seg_len - crossfade_samples
            total_len = step * n_copies + crossfade_samples
            out_audio = np.zeros(total_len, dtype=np.float32)

            for i in range(n_copies):
                offset = i * step
                if i == 0:
                    out_audio[offset:offset + seg_len] = segment
                else:
                    # Crossfade region: linear ramp
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples,
                                           dtype=np.float32)
                    fade_in  = np.linspace(0.0, 1.0, crossfade_samples,
                                           dtype=np.float32)

                    xf_start = offset
                    xf_end   = offset + crossfade_samples

                    # Fade out the tail of the previous copy
                    out_audio[xf_start:xf_end] *= fade_out
                    # Fade in the head of the new copy
                    out_audio[xf_start:xf_end] += segment[:crossfade_samples] * fade_in
                    # Copy the rest of the segment after the crossfade
                    out_audio[xf_end:offset + seg_len] = segment[crossfade_samples:]

    # Write output
    out_path = out_dir / f"{stem}.loop.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out_audio, SR, subtype="PCM_16")

    # License sidecar
    license_path = out_dir / "licenses" / f"{stem}.loop.license.json"
    write_license_sidecar(license_path)

    size = out_path.stat().st_size
    print(f"  [OK]      {out_path}  ({size:,} bytes)")
    print(f"  [LICENSE] {license_path}")
    return out_path


# ── Shot override application ────────────────────────────────────────────────

OVERRIDE_FIELDS = {
    "duck_db", "fade_sec", "start_sec", "duration_sec",
    "music_asset_id", "crossfade_sec",
    "clip_start_sec", "clip_duration_sec",
}


def extract_clip_from_source(
    source_stem: str,
    resources_dir: Path,
    out_path: Path,
    start_sec: float = 0.0,
    duration_sec: float = 30.0,
) -> bool:
    """
    Extract a clip from a source track and write it to out_path.

    Used when a shot override's music_asset_id is a source track stem
    rather than an existing clip item_id.
    """
    import soundfile as sf

    source_path = find_source_track(source_stem, resources_dir)
    if source_path is None:
        print(f"  [ERROR] Source track not found for stem '{source_stem}' "
              f"in {resources_dir}", file=sys.stderr)
        return False

    audio = load_audio_48k(source_path)
    total_dur = len(audio) / SR

    # Clamp start_sec to valid range
    if start_sec >= total_dur:
        start_sec = 0.0
    end_sec = min(start_sec + duration_sec, total_dur)

    start_sample = int(start_sec * SR)
    end_sample = int(end_sec * SR)
    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        print(f"  [ERROR] Empty segment from '{source_stem}'", file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), segment, SR, subtype="PCM_16")

    print(f"  [EXTRACT] {source_stem} [{start_sec:.1f}s–{end_sec:.1f}s] "
          f"→ {out_path.name}  ({out_path.stat().st_size:,} bytes)")
    return True


def apply_shot_overrides(
    manifest: dict,
    overrides: list[dict],
    resources_dir: Path | None = None,
    assets_music_dir: Path | None = None,
) -> int:
    """
    Apply shot overrides to music_items in the manifest.

    When music_asset_id is a source track stem (not an existing clip item_id),
    extracts a new clip from the source track into assets/music/.

    Returns the number of items updated.
    """
    if not overrides:
        return 0

    # Index music_items by item_id
    music_index: dict[str, dict] = {
        m["item_id"]: m for m in manifest.get("music_items", [])
    }

    # Collect existing clip filenames for stem detection
    existing_clips: set[str] = set()
    if assets_music_dir and assets_music_dir.is_dir():
        for f in assets_music_dir.iterdir():
            if f.suffix in SUPPORTED_EXTS:
                existing_clips.add(f.stem)
                # Also match without .loop suffix
                if f.stem.endswith(".loop"):
                    existing_clips.add(f.stem[:-5])

    updated = 0
    for override in overrides:
        item_id = override["item_id"]
        item = music_index.get(item_id)
        if item is None:
            print(f"  [WARN] item_id '{item_id}' not found in manifest "
                  "— skipping override", file=sys.stderr)
            continue

        # Check if music_asset_id is a source stem (not an existing clip)
        new_asset_id = override.get("music_asset_id")
        if new_asset_id and new_asset_id not in music_index and assets_music_dir:
            # First check if it's a pre-cut WAV already in assets/music/
            pre_cut_wav = assets_music_dir / f"{new_asset_id}.wav"
            if pre_cut_wav.exists():
                # Copy/symlink pre-cut clip over the target item WAV
                out_wav = assets_music_dir / f"{item_id}.wav"
                if pre_cut_wav != out_wav:
                    import shutil as _shutil
                    _shutil.copy2(str(pre_cut_wav), str(out_wav))
                    print(f"  [PRE-CUT→CLIP] {new_asset_id}.wav → {item_id}.wav")
            elif resources_dir and resources_dir.is_dir():
                # It's a source stem — extract a new clip from resources/music/
                source_path = find_source_track(new_asset_id, resources_dir)
                if source_path:
                    clip_start = override.get("clip_start_sec",
                                              override.get("start_sec",
                                              item.get("start_sec", 0.0)))
                    clip_dur = override.get("clip_duration_sec",
                                            override.get("duration_sec",
                                            item.get("duration_sec", 30.0)))
                    out_wav = assets_music_dir / f"{item_id}.wav"
                    ok = extract_clip_from_source(
                        new_asset_id, resources_dir, out_wav,
                        start_sec=float(clip_start),
                        duration_sec=float(clip_dur),
                    )
                    if ok:
                        print(f"  [SOURCE→CLIP] {new_asset_id} → {item_id}")
                else:
                    print(f"  [WARN] Source stem '{new_asset_id}' not found in "
                          f"{resources_dir}", file=sys.stderr)
            else:
                print(f"  [WARN] Source stem '{new_asset_id}' not found in "
                      f"assets/music/ or resources/music/", file=sys.stderr)

        changes = []
        for field in OVERRIDE_FIELDS:
            if field in override:
                old_val = item.get(field, "<unset>")
                item[field] = override[field]
                changes.append(f"{field}={override[field]}")

        if changes:
            print(f"  [OVERRIDE] {item_id}: {', '.join(changes)}")
            updated += 1

    return updated


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Apply MusicPlan overrides: generate loop WAVs and "
                    "update shot-level music parameters in the manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--plan", required=True, metavar="PATH",
                   help="Path to MusicPlan.json")
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to merged locale manifest "
                        "(AssetManifest_merged.{locale}.json)")
    p.add_argument("--resources", default=None, metavar="DIR",
                   help="Directory containing source music files. "
                        "Auto-detected from manifest project_id if omitted.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    plan_path     = Path(args.plan).resolve()
    manifest_path = Path(args.manifest).resolve()

    # ── Load & validate plan ─────────────────────────────────────────────────
    if not plan_path.exists():
        print(f"[ERROR] Plan not found: {plan_path}", file=sys.stderr)
        sys.exit(1)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    validate_plan(plan)

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest = load_manifest(manifest_path)

    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")

    # ── Resolve resources dir ────────────────────────────────────────────────
    if args.resources:
        resources_dir = Path(args.resources).resolve()
    else:
        candidates_dirs = [
            PIPE_DIR / "projects" / project_id / "resources" / "music",
            PIPE_DIR / "projects" / "resources" / "music",
        ]
        resources_dir = next(
            (d for d in candidates_dirs if d.is_dir()),
            None,
        )

    # ── Determine output dir for loop WAVs ───────────────────────────────────
    if project_id and episode_id:
        assets_music_dir = (PIPE_DIR / "projects" / project_id / "episodes"
                            / episode_id / "assets" / "music")
    else:
        assets_music_dir = manifest_path.parent / "assets" / "music"

    loop_selections = plan.get("loop_selections", {})
    shot_overrides  = plan.get("shot_overrides", [])

    print("=" * 60)
    print("  apply_music_plan")
    print(f"  Plan      : {plan_path.name}")
    print(f"  Manifest  : {manifest_path.name}")
    print(f"  Resources : {resources_dir}")
    print(f"  Output    : {assets_music_dir}")
    print(f"  Loops     : {len(loop_selections)}")
    print(f"  Overrides : {len(shot_overrides)}")
    print("=" * 60)

    # ── Compute max shot duration for loop length ────────────────────────────
    music_items = manifest.get("music_items", [])
    max_shot_duration = max(
        (float(m.get("duration_sec", 30)) for m in music_items),
        default=30.0,
    )

    # ── Generate loop WAVs ───────────────────────────────────────────────────
    loop_ok = 0
    loop_fail = 0

    if loop_selections:
        print(f"\n── Loop generation ({len(loop_selections)} tracks) ──")

        if resources_dir is None or not resources_dir.is_dir():
            print("[ERROR] No music resources directory found — cannot "
                  "generate loops.", file=sys.stderr)
            print("  Provide --resources or ensure "
                  "projects/{project_id}/resources/music/ exists.",
                  file=sys.stderr)
            loop_fail = len(loop_selections)
        else:
            for stem, selection in loop_selections.items():
                print(f"\n  [{stem}]")
                result = generate_loop_wav(
                    stem, selection, resources_dir,
                    assets_music_dir, max_shot_duration,
                )
                if result:
                    loop_ok += 1
                else:
                    loop_fail += 1

    # ── Apply shot overrides ─────────────────────────────────────────────────
    override_count = 0

    if shot_overrides:
        print(f"\n── Shot overrides ({len(shot_overrides)} items) ──")
        override_count = apply_shot_overrides(
            manifest, shot_overrides,
            resources_dir=resources_dir,
            assets_music_dir=assets_music_dir,
        )
        save_manifest(manifest, manifest_path)
        print(f"\n  Manifest updated: {manifest_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — apply_music_plan")
    print(f"{'='*60}")
    print(f"  Loops generated : {loop_ok}/{len(loop_selections)}")
    if loop_fail:
        print(f"  Loops failed    : {loop_fail}")
    print(f"  Shot overrides  : {override_count}/{len(shot_overrides)}")
    print(f"  Manifest        : {manifest_path}")


if __name__ == "__main__":
    main()
