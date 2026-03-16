#!/usr/bin/env python3
"""
fetch_ai_assets.py — Delegate AI asset generation to the remote GPU server
(code/ai/http/server.py), check for existing files, and save results.

Mirrors gen_tts_cloud.py conventions:
  • --manifest is the only required path arg; output dir derived from manifest
  • --asset-id for single-item processing
  • Derives output path from manifest["project_id"] / manifest["episode_id"]

Usage:
  python3 fetch_ai_assets.py --manifest /path/to/AssetManifest.shared.json \\
                              --asset_type characters
  python3 fetch_ai_assets.py --manifest ... --asset_type sfx
  python3 fetch_ai_assets.py --manifest ... --asset_type characters \\
                              --asset-id char-amunhotep-v1

Output paths (derived from manifest fields, no --output_dir needed):
  characters  → projects/{project_id}/characters/          (project-level)
  backgrounds → projects/{project_id}/episodes/{ep}/assets/backgrounds/
  bg_video    → projects/{project_id}/episodes/{ep}/assets/bg_video/
  sfx         → projects/{project_id}/episodes/{ep}/assets/sfx/

Environment variables:
  AI_SERVER_URL   default: http://192.168.86.27:8000
  AI_SERVER_KEY   default: change-me
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root

AI_SERVER_URL = os.environ.get("AI_SERVER_URL", "http://192.168.86.27:8000")
AI_SERVER_KEY = os.environ.get("AI_SERVER_KEY", "change-me")

POLL_INTERVAL = 4  # seconds between status polls

# Maps asset_type → manifest section key (what the GPU server reads).
# NOTE: "bg_video" intentionally maps to the same "backgrounds" section as
# "backgrounds".  In the actual manifest, still-image backgrounds
# (search_filters.media_type="photo", no motion field) and video backgrounds
# (media_type="video", motion.type="camera") coexist in the SAME array.
# gen_background_images.py and gen_background_video.py each self-filter
# within that array by checking the motion field — no separate section exists.
SECTION_KEY: dict[str, str] = {
    "characters":  "character_packs",
    "backgrounds": "backgrounds",
    "bg_video":    "backgrounds",  # same section; gen scripts self-filter by motion
    "sfx":         "sfx_items",
}

# sfx_items entries use "shot_id" as their identifier; all others use "asset_id".
ID_FIELD: dict[str, str] = {
    "characters":  "asset_id",
    "backgrounds": "asset_id",
    "bg_video":    "asset_id",
    "sfx":         "shot_id",
}

# Output directory templates relative to PIPE_DIR.
# {p} = project_id, {e} = episode_id — filled from manifest fields.
OUTPUT_DIR_TPL: dict[str, str] = {
    "characters":  "projects/{p}/characters",
    "backgrounds": "projects/{p}/episodes/{e}/assets/backgrounds",
    "bg_video":    "projects/{p}/episodes/{e}/assets/bg_video",
    "sfx":         "projects/{p}/episodes/{e}/assets/sfx",
}

# File extension produced by each asset type (used for already-done check).
OUTPUT_EXT: dict[str, str] = {
    "characters":  ".png",
    "backgrounds": ".png",
    "bg_video":    ".mp4",
    "sfx":         ".wav",
}

# Manifest metadata keys forwarded to GPU node alongside the asset section.
MANIFEST_META_KEYS = (
    "schema_id", "schema_version", "manifest_id",
    "project_id", "episode_id", "shotlist_ref",
)

# ---------------------------------------------------------------------------
# HTTP helpers  (stdlib only — no external dependencies)
# ---------------------------------------------------------------------------

def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        AI_SERVER_URL + path,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Api-Key":    AI_SERVER_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(path: str) -> dict:
    req = urllib.request.Request(
        AI_SERVER_URL + path,
        headers={"X-Api-Key": AI_SERVER_KEY},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_bytes(path: str) -> bytes:
    req = urllib.request.Request(
        AI_SERVER_URL + path,
        headers={"X-Api-Key": AI_SERVER_KEY},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> dict:
    """Load manifest JSON, with fallback for older episode naming.

    If AssetManifest.shared.json is requested but absent, falls back to
    AssetManifest.json (produced by older Stage 5 before the shared/locale
    split was introduced).
    """
    p = Path(manifest_path)
    if not p.exists() and p.name == "AssetManifest.shared.json":
        fallback = p.parent / "AssetManifest.json"
        if fallback.exists():
            print(f"[WARN] {p.name} not found; using fallback {fallback.name}")
            p = fallback
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def strip_manifest(manifest: dict, asset_type: str) -> dict:
    """Return a minimal manifest containing only the section the GPU needs.

    Strips all other asset sections (vo_items, music_items, etc.) so the GPU
    node receives exactly what it must process and nothing else.
    """
    section_key = SECTION_KEY[asset_type]
    stripped = {k: manifest[k] for k in MANIFEST_META_KEYS if k in manifest}
    stripped[section_key] = manifest.get(section_key, [])
    return stripped


def get_all_ids(manifest: dict, asset_type: str) -> list[str]:
    """Return all id values for the requested asset type.

    Uses "asset_id" for characters/backgrounds/bg_video, "shot_id" for sfx.
    """
    section_key = SECTION_KEY[asset_type]
    id_field    = ID_FIELD[asset_type]
    return [entry[id_field] for entry in manifest.get(section_key, [])]


def output_dir(manifest: dict, asset_type: str) -> Path:
    tpl = OUTPUT_DIR_TPL[asset_type]
    rel = tpl.format(p=manifest["project_id"], e=manifest.get("episode_id", ""))
    return PIPE_DIR / rel


# ---------------------------------------------------------------------------
# Already-done check
# ---------------------------------------------------------------------------

def find_missing(
    manifest: dict,
    asset_type: str,
    asset_id_filter: str | None,
) -> list[str]:
    """Return ids that have NOT yet been downloaded locally.

    If asset_id_filter is set, only that single id is considered.

    SFX special case: filenames are sfx_{shot_id}_{tag_slug}.wav so the stem
    does NOT equal the shot_id.  We check for any file matching
    sfx_{shot_id}_*.wav using prefix-strip + rsplit to safely handle shot_ids
    that contain underscores (e.g. "s01_sh01"):
        "sfx_s01-sh01_stone-scraping.wav"
          [4:]          → "s01-sh01_stone-scraping.wav"
          rsplit("_",1) → ["s01-sh01", "stone-scraping.wav"]
          [0]           → "s01-sh01"  ✓
    gen_sfx.py has its own per-tag file skip, so partially-done shots are
    handled correctly if re-submitted.
    """
    all_ids = get_all_ids(manifest, asset_type)
    if asset_id_filter:
        all_ids = [aid for aid in all_ids if aid == asset_id_filter]
        if not all_ids:
            print(
                f"[WARN] id '{asset_id_filter}' not found in manifest section "
                f"'{SECTION_KEY[asset_type]}'",
                file=sys.stderr,
            )
            return []

    out = output_dir(manifest, asset_type)
    ext = OUTPUT_EXT[asset_type]

    if asset_type == "sfx":
        if out.exists():
            done_shots = {
                f.name[4:].rsplit("_", maxsplit=1)[0]
                for f in out.glob(f"sfx_*{ext}")
                if "_" in f.name[4:]  # must have at least one _ after "sfx_" prefix
            }
        else:
            done_shots = set()
        missing = [sid for sid in all_ids if sid not in done_shots]
        already = [sid for sid in all_ids if sid in done_shots]
    else:
        already_stems = (
            {p.stem for p in out.glob(f"*{ext}")} if out.exists() else set()
        )
        missing = [aid for aid in all_ids if aid not in already_stems]
        already = [aid for aid in all_ids if aid in already_stems]

    if already:
        print(
            f"[CACHE] {len(already)}/{len(all_ids)} already present, skipping: "
            + ", ".join(already)
        )
    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch AI-generated assets from remote GPU server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to AssetManifest.shared.json (or base draft).",
    )
    parser.add_argument(
        "--asset_type", required=True,
        choices=list(SECTION_KEY),
        help="Asset type to generate: characters, backgrounds, bg_video, or sfx.",
    )
    parser.add_argument(
        "--asset-id", dest="asset_id", default=None,
        help=(
            "Process only this single id (asset_id for images/video, "
            "shot_id for sfx). Optional."
        ),
    )
    parser.add_argument(
        "--asset-ids", dest="asset_ids", default=None,
        help="Comma-separated list of asset_ids to process. Overrides --asset-id.",
    )
    args = parser.parse_args()

    # 1. Load + validate manifest
    manifest = load_manifest(args.manifest)
    section  = SECTION_KEY[args.asset_type]
    if section not in manifest or not manifest[section]:
        print(f"[SKIP] No '{section}' entries in manifest.")
        return

    # 2. Determine which ids are missing locally
    # Resolve the ID filter — --asset-ids takes priority over --asset-id
    if args.asset_ids:
        asset_ids_set = set(i.strip() for i in args.asset_ids.split(',') if i.strip())
        # Pass None as asset_id_filter; we apply the set filter ourselves below
        _raw_missing = find_missing(manifest, args.asset_type, None)
        missing = [aid for aid in _raw_missing if aid in asset_ids_set]
        # Warn about any requested IDs not present in the manifest
        all_ids_in_manifest = set(get_all_ids(manifest, args.asset_type))
        for rid in asset_ids_set:
            if rid not in all_ids_in_manifest:
                print(
                    f"[WARN] id '{rid}' not found in manifest section "
                    f"'{SECTION_KEY[args.asset_type]}'",
                    file=sys.stderr,
                )
    else:
        missing = find_missing(manifest, args.asset_type, args.asset_id)
    if not missing:
        print(f"[SKIP] All {args.asset_type} already downloaded. Nothing to do.")
        return
    print(f"[INFO] Will generate {len(missing)} {args.asset_type}: {missing}")

    # 3. Build stripped manifest — only the relevant section, only missing entries.
    #    Sending a minimal payload means the GPU node never receives vo_items,
    #    music_items, or other sections it has no use for.
    stripped    = strip_manifest(manifest, args.asset_type)
    id_field    = ID_FIELD[args.asset_type]  # "asset_id" or "shot_id" (sfx)
    missing_set = set(missing)
    stripped[section] = [
        entry for entry in stripped[section]
        if entry[id_field] in missing_set
    ]

    # 4. Submit job to AI server.
    #    Do NOT pass asset_ids here — the manifest section is already filtered to
    #    only the missing entries.  Passing asset_ids would cause server.py to
    #    spawn one subprocess per id, reloading the ~6 GB model each time.
    #    Pre-filtering the section lets the gen script process all items in a
    #    single subprocess with one model load.
    try:
        job = _post("/jobs", {
            "manifest":    stripped,
            "asset_types": [args.asset_type],
        })
    except urllib.error.URLError as exc:
        print(
            f"[ERROR] Cannot reach AI server at {AI_SERVER_URL}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Server sets status="unavailable" (HTTP 200, job_id=null) when offline mode
    # is active (config.json "offline": true).  Treat as a clean skip — not an error.
    if job.get("status") == "unavailable":
        print(f"[SKIP] AI server is offline — skipping {args.asset_type}.")
        return

    job_id = job["job_id"]
    total  = job["total"]
    print(f"[JOB]  {job_id}  queued  ({total} items)")

    # 5. Poll until done
    while True:
        time.sleep(POLL_INTERVAL)
        state  = _get_json(f"/jobs/{job_id}")
        status = state["status"]
        print(f"[POLL] status={status}  done={state['done']}/{state['total']}")

        if status == "failed":
            print("[ERROR] Job failed on GPU server.", file=sys.stderr)
            for err in state.get("errors", []):
                print(f"  • {err}", file=sys.stderr)
            for line in state.get("log_tail", []):
                print(f"  | {line}", file=sys.stderr)
            sys.exit(1)

        if status == "done":
            break

    # 6. Download and save files.
    #    If state["files"] is empty (server produced nothing), the loop is a
    #    no-op and we exit cleanly — no crash.
    out = output_dir(manifest, args.asset_type)
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    for filename in state["files"]:
        data = _get_bytes(f"/jobs/{job_id}/files/{filename}")
        dest = out / filename
        dest.write_bytes(data)
        saved.append(str(dest.relative_to(PIPE_DIR)))
        print(f"[SAVE] {dest.relative_to(PIPE_DIR)}  ({len(data):,} bytes)")

    print(f"[DONE] {len(saved)} file(s) saved to {out.relative_to(PIPE_DIR)}/")

    # 7. Write/merge index.json.
    #    AI server jobs expire after 24 h; index.json gives the pipeline a
    #    persistent local record of what has been downloaded without querying
    #    the server again.  Re-runs accumulate entries rather than overwriting,
    #    so partial runs across multiple invocations build up a complete index.
    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "asset_type":   args.asset_type,
        "assets": [
            {id_field: os.path.splitext(filename)[0], "file": filename}
            for filename in state["files"]
        ],
    }
    index_path = out / "index.json"
    if index_path.exists():
        try:
            existing     = json.loads(index_path.read_text(encoding="utf-8"))
            existing_ids = {a[id_field] for a in existing.get("assets", [])}
            new_entries  = [a for a in index["assets"] if a[id_field] not in existing_ids]
            existing["assets"].extend(new_entries)
            existing["generated_at"] = index["generated_at"]  # refresh timestamp
            index = existing
        except Exception:
            pass  # corrupt index — overwrite cleanly
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[IDX]  {index_path.relative_to(PIPE_DIR)}"
        f"  ({len(index['assets'])} total assets)"
    )


if __name__ == "__main__":
    main()
