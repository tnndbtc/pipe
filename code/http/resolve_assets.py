#!/usr/bin/env python3
# =============================================================================
# resolve_assets.py — Generate AssetManifest.media.{locale}.json
# =============================================================================
#
# Reads AssetManifest_merged.{locale}.json and probes the local filesystem
# for each asset. Emits file:// URIs for found files, placeholder:// for
# missing ones. No external calls — single-pass, known-path resolver.
#
# Bypasses media-agent entirely: we know exactly where every file lives.
#
# Usage:
#   python resolve_assets.py \
#       --manifest projects/slug/ep/AssetManifest_merged.zh-Hans.json
#
#   python resolve_assets.py \
#       --manifest projects/slug/ep/AssetManifest_merged.zh-Hans.json \
#       --assets-root /custom/assets \
#       --out /custom/AssetManifest.media.zh-Hans.json \
#       --strict
#
# File path conventions (relative to assets-root):
#   VO        : {locale}/audio/vo/{item_id}.{wav|mp3|ogg}
#   Music     : music/{item_id}.{wav|mp3|ogg}
#   SFX       : sfx/{item_id}.{wav|mp3|ogg}
#   Background: {asset_id}.{png|jpg|webp|gif}  (then backgrounds/ subdir)
#   Character : projects/{project_id}/characters/ (project-level, FIRST)
#               then assets/characters/{asset_id}.ext  (episode subdir)
#               then assets/{asset_id}.ext             (episode root, last resort)
#               Also tries short form: strip "char-" prefix + "-vN" suffix
#               so amunhotep.png matches char-amunhotep-v1
#
# Output: AssetManifest.media.{locale}.json in episode directory (or --out)
#
# Requirements: stdlib only
# =============================================================================

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Repo root — two levels up from code/http/
PIPE_DIR = Path(__file__).resolve().parent.parent.parent

PRODUCER = "resolve_assets.py"
DETERMINISTIC_TS = "1970-01-01T00:00:00Z"
IMAGE_EXTS = ["png", "jpg", "webp", "gif"]
VIDEO_EXTS = ["mp4", "mov", "webm"]
AUDIO_EXTS = ["wav", "mp3", "ogg"]

# Map manifest license_type → SPDX identifier
SPDX_MAP: dict[str, str] = {
    "generated_local":     "LicenseRef-generated",
    "proprietary_cleared": "LicenseRef-proprietary-cleared",
    "commercial_licensed": "LicenseRef-commercial-licensed",
    "CC0":                 "CC0-1.0",
    "CC0-1.0":             "CC0-1.0",
    "Public Domain":       "CC0-1.0",
    "CC BY":               "CC-BY-4.0",
    "CC BY 4.0":           "CC-BY-4.0",
    "CC BY 3.0":           "CC-BY-3.0",
    "CC BY 2.0":           "CC-BY-2.0",
    "CC BY 1.0":           "CC-BY-1.0",
    "MIT":                 "MIT",
}


# ── High-res download helper ──────────────────────────────────────────────────

def _ensure_hires(local_preview_path: Path, no_hires: bool = False) -> Path:
    """Download the high-resolution version of a selected asset for render.

    Downloads to hires/<uid>.ext alongside the preview file, using an atomic
    .tmp → rename pattern. Falls back to preview path on any failure.

    Args:
        local_preview_path: Path to the downloaded preview file.
        no_hires: If True, skip download and return preview path directly.

    Returns:
        Path to hires file if successfully downloaded, else local_preview_path.
    """
    import urllib.request as _ur
    import json as _json

    if no_hires:
        return local_preview_path

    # Read sidecar
    sidecar = Path(str(local_preview_path) + ".info.json")
    if not sidecar.exists():
        log.warning("[hires] no sidecar for %s — using preview", local_preview_path.name)
        return local_preview_path

    try:
        info = _json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[hires] cannot read sidecar %s: %s — using preview", sidecar, exc)
        return local_preview_path

    file_url = info.get("file_url", "")
    if not file_url:
        log.warning("[hires] no file_url in sidecar for %s — using preview", local_preview_path.name)
        return local_preview_path

    # For videos, pick highest-width entry from video_files[] if available
    is_video = local_preview_path.suffix.lower() in {".mp4", ".mov", ".webm"}
    hires_url = file_url
    if is_video:
        vf = info.get("video_files") or []
        if vf:
            best = max(vf, key=lambda f: f.get("width") or 0)
            hires_url = best.get("url") or file_url

    # Determine paths
    hires_dir  = local_preview_path.parent / "hires"
    hires_dest = hires_dir / local_preview_path.name
    hires_tmp  = hires_dir / f"{local_preview_path.stem}.tmp{os.getpid()}{local_preview_path.suffix}"

    # Validate existing hires file (M3)
    if hires_dest.exists():
        if hires_dest.stat().st_size == 0:
            log.warning("[hires] existing hires file is empty, re-downloading: %s", hires_dest)
            hires_dest.unlink()
        else:
            log.info("[hires] already present: %s", hires_dest)
            return hires_dest

    # Clean up stale .tmp from prior crash (B1)
    if hires_tmp.exists():
        hires_tmp.unlink()

    # Choose download URL and log intent (M4)
    asset_id = local_preview_path.stem
    log.info("[hires] downloading %s → %s from %s", asset_id, hires_dest, hires_url)

    try:
        hires_dir.mkdir(parents=True, exist_ok=True)
        # Download to tmp (B1 — atomic)
        _ur.urlretrieve(hires_url, hires_tmp)
        size = hires_tmp.stat().st_size
        if size == 0:
            raise ValueError("downloaded file is empty")
        # Atomic rename
        hires_tmp.rename(hires_dest)
        log.info("[hires] complete: %s (%d bytes)", hires_dest, size)
        return hires_dest
    except Exception as exc:
        # Clean up partial tmp (C2, C5)
        if hires_tmp.exists():
            try:
                hires_tmp.unlink()
            except Exception:
                pass
        log.warning("[hires] download failed for %s: %s — using preview", asset_id, exc)
        return local_preview_path


# ── File search ───────────────────────────────────────────────────────────────

def normalise_id(raw: str) -> str:
    """
    Normalise an asset ID for filesystem lookup, mirroring media-agent behaviour:
    strip → lowercase → spaces and underscores → hyphens.
    """
    return raw.strip().lower().replace(" ", "-").replace("_", "-")


def find_file(directory: Path, stem: str, extensions: list[str]) -> Path | None:
    """
    Return first match for directory/stem.ext; None if not found.
    Tries the raw stem first, then the normalised (hyphenated) stem.
    """
    for candidate_stem in _stems(stem):
        for ext in extensions:
            candidate = directory / f"{candidate_stem}.{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _stems(raw: str) -> list[str]:
    """Return [raw, normalised] deduplicated — raw first so exact match wins."""
    normed = normalise_id(raw)
    return [raw, normed] if raw != normed else [raw]


def search_dirs(
    dirs: list[Path],
    stem: str,
    extensions: list[str],
) -> Path | None:
    """Try each directory in order; return first match or None."""
    for d in dirs:
        found = find_file(d, stem, extensions)
        if found:
            return found
    return None


# ── Item builders ─────────────────────────────────────────────────────────────

def _resolved(
    asset_id: str,
    asset_type: str,
    file_path: Path,
    license_type: str,
) -> dict:
    spdx_id = SPDX_MAP.get(license_type, "NOASSERTION")
    return {
        "asset_id":       asset_id,
        "asset_type":     asset_type,
        "uri":            file_path.as_uri(),
        "is_placeholder": False,
        "metadata": {
            "license_type":       license_type,
            "attribution":        "",
            "purchase_record":    "",
            "provider_or_model":  "local_library",
            "retrieval_date":     DETERMINISTIC_TS,
        },
        "rights_warning": "",
        "source":  {"type": "local"},
        "license": {
            "spdx_id":             spdx_id,
            "attribution_required": False,
            "text":                "",
        },
        "schema_id":      "urn:media:resolved-asset",
        "schema_version": "1.0.0",
        "producer":       PRODUCER,
    }


def _resolved_stock(
    asset_id:   str,
    file_path:  Path,
    media_type: str,
    source_url: str,
) -> dict:
    """
    Item builder for stock media selected via the VC editor (Pexels / Pixabay).

    Infers the provider from the filename prefix (pexels_* / pixabay_*) and
    sets the appropriate license type:
      pexels_*  → commercial_licensed  (Pexels license, free for commercial use)
      pixabay_* → CC0                  (Pixabay Content License)
    """
    name = file_path.name.lower()
    if name.startswith("pexels"):
        provider     = "pexels"
        license_type = "commercial_licensed"
    elif name.startswith("pixabay"):
        provider     = "pixabay"
        license_type = "CC0"
    else:
        provider     = "stock"
        license_type = "commercial_licensed"
    spdx_id = SPDX_MAP.get(license_type, "NOASSERTION")
    return {
        "asset_id":       asset_id,
        "asset_type":     "background",
        "uri":            file_path.as_uri(),
        "is_placeholder": False,
        "metadata": {
            "license_type":       license_type,
            "attribution":        source_url,
            "purchase_record":    "",
            "provider_or_model":  provider,
            "retrieval_date":     DETERMINISTIC_TS,
        },
        "rights_warning": "",
        "source":  {"type": "stock", "provider": provider, "url": source_url},
        "license": {
            "spdx_id":             spdx_id,
            # Pixabay requires attribution in some commercial contexts
            "attribution_required": provider == "pixabay",
            "text":                "",
        },
        "schema_id":      "urn:media:resolved-asset",
        "schema_version": "1.0.0",
        "producer":       PRODUCER,
    }


def _resolved_sfx(asset_id: str, file_path: Path, sfx_item: dict) -> dict:
    """Build a resolved SFX item, reading the .info.json sidecar for license fields.

    Falls back to manifest sfx_item fields when the sidecar is absent, and to
    'proprietary_cleared' when neither has a recognized license_type.
    """
    # Locate sidecar: {item_id}.mp3 → {item_id}.info.json (same stem, no double-ext)
    sidecar = file_path.parent / (file_path.stem + ".info.json")
    info: dict = {}
    if sidecar.exists():
        try:
            info = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("[sfx] failed to read sidecar %s: %s", sidecar, exc)

    # License fields — sidecar takes priority over manifest
    lic_summary  = info.get("license_summary")  or sfx_item.get("license_summary", "")
    lic_url      = info.get("license_url")       or sfx_item.get("license_url", "")
    attr_text    = info.get("attribution_text")  or sfx_item.get("attribution_text", "")
    author       = info.get("author")            or sfx_item.get("author", "")
    source_site  = info.get("source_site")       or sfx_item.get("source", "")
    attr_req     = info.get("attribution_required",
                            sfx_item.get("attribution_required",
                                         lic_summary not in {"CC0", "Public Domain"}))

    # Map to SPDX — prefer manifest's license_type key (already mapped at save time)
    lic_type = sfx_item.get("license_type") or lic_summary or "proprietary_cleared"
    spdx_id  = SPDX_MAP.get(lic_type, SPDX_MAP.get(lic_summary, "NOASSERTION"))

    rights_warning = ""
    if attr_req and attr_text:
        rights_warning = f"Attribution required: {attr_text}"

    return {
        "asset_id":       asset_id,
        "asset_type":     "sfx",
        "uri":            file_path.as_uri(),
        "is_placeholder": False,
        "metadata": {
            "license_type":       lic_type,
            "attribution":        attr_text,
            "provider_or_model":  source_site,
            "purchase_record":    "",
            "retrieval_date":     DETERMINISTIC_TS,
        },
        "rights_warning": rights_warning,
        "source":  {"type": "library", "provider": source_site, "url": lic_url},
        "license": {
            "spdx_id":              spdx_id,
            "attribution_required": bool(attr_req),
            "text":                 "",
        },
        "schema_id":      "urn:media:resolved-asset",
        "schema_version": "1.0.0",
        "producer":       PRODUCER,
    }


def _placeholder(asset_id: str, asset_type: str) -> dict:
    return {
        "asset_id":       asset_id,
        "asset_type":     asset_type,
        "uri":            f"placeholder://{asset_type}/{asset_id}",
        "is_placeholder": True,
        "metadata": {
            "license_type":       "placeholder",
            "attribution":        "",
            "purchase_record":    "",
            "provider_or_model":  "placeholder_stub_v0",
            "retrieval_date":     DETERMINISTIC_TS,
        },
        "rights_warning": "",
        "source":  {"type": "generated_placeholder"},
        "license": {
            "spdx_id":             "NOASSERTION",
            "attribution_required": False,
            "text":                "",
        },
        "schema_id":      "urn:media:resolved-asset",
        "schema_version": "1.0.0",
        "producer":       PRODUCER,
    }


# ── Selections loader ─────────────────────────────────────────────────────────

def _resolve_one_entry(entry: dict, batch_dir: Path | None) -> dict | None:
    """
    Resolve a single selection entry to { media_type, abs_path, url, score }.

    Returns None if the entry cannot be resolved (missing path info).
    """
    url      = entry.get("url", "")
    rel_path = entry.get("path", "")

    if url.startswith("file://"):
        # file transport: URL IS the authoritative path on this machine.
        # file:///mnt/shared/…/img.jpg  →  Path("/mnt/shared/…/img.jpg")
        abs_path = Path(url[len("file://"):]).resolve()
    elif rel_path and batch_dir:
        # http transport: reconstruct path from batch_dir + rel_path
        abs_path = (batch_dir / rel_path).resolve()
    else:
        return None

    return {
        "media_type": entry.get("media_type") or entry.get("type", "image"),
        "abs_path":   abs_path,
        "url":        url,
        "score":      entry.get("score"),
    }


def _load_selections(selections_path: Path) -> dict[str, dict]:
    """
    Parse selections.json written by the VC editor.

    Returns:
      Old format (version 1) → { asset_id → { media_type, abs_path, url, score } }
      New format (version 2) → { asset_id → { per_shot: { shot_id → { media_type, abs_path, url, score } } } }

    The path in selections.json is relative to the batch directory
    (assets/media/<batch_id>/), so we reconstruct the absolute path
    as  selections_path.parent / batch_id / rel_path.
    """
    try:
        with open(selections_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [WARN] Could not load {selections_path}: {exc}", file=sys.stderr)
        return {}

    batch_id  = data.get("batch_id", "")
    batch_dir = selections_path.parent / batch_id if batch_id else None

    out: dict[str, dict] = {}
    for asset_id, entry in data.get("selections", {}).items():
        if "per_shot" in entry:
            # Per-shot format (version 2 or 3)
            per_shot_resolved: dict[str, dict] = {}
            for shot_id, shot_entry in entry["per_shot"].items():
                if "segments" in shot_entry:
                    # v3: multi-segment per shot
                    resolved_segs: list[dict] = []
                    for seg in shot_entry["segments"]:
                        resolved = _resolve_one_entry(seg, batch_dir)
                        if resolved:
                            resolved["duration_sec"] = seg.get("duration_sec")
                            resolved["hold_sec"]     = seg.get("hold_sec")
                            resolved["duration_override_sec"] = seg.get("duration_override_sec")
                            resolved["natural_duration_sec"]  = seg.get("natural_duration_sec")
                            resolved_segs.append(resolved)
                    if resolved_segs:
                        per_shot_resolved[shot_id] = {"segments": resolved_segs}
                else:
                    # v2: single entry per shot
                    resolved = _resolve_one_entry(shot_entry, batch_dir)
                    if resolved:
                        per_shot_resolved[shot_id] = resolved
            if per_shot_resolved:
                out[asset_id] = {"per_shot": per_shot_resolved}
        else:
            # Old format (single entry per background)
            resolved = _resolve_one_entry(entry, batch_dir)
            if resolved:
                out[asset_id] = resolved
    return out


# ── Main resolver ─────────────────────────────────────────────────────────────

def resolve_all(
    merged:      dict,
    assets_root: Path,
    selections:  dict | None = None,
    no_hires:    bool = False,
) -> list[dict]:
    """
    Walk all asset types in the merged manifest and probe the filesystem.

    selections — optional dict loaded from selections.json by _load_selections().
                 When present, stock media chosen via the VC editor takes priority
                 over the regular filesystem scan for background assets.
    no_hires   — when True, skip hires download and use preview paths directly.

    Returns a list of ResolvedAsset dicts (file:// or placeholder://).
    """
    locale = merged.get("locale", "")
    items: list[dict] = []
    n_found = 0
    n_missing = 0

    # ── 1. Characters ────────────────────────────────────────────────────────
    # Search order (project-level wins so all locales share the same portraits):
    #   1. projects/{project_id}/characters/           (project-level, shared — FIRST)
    #   2. assets/characters/{asset_id}.ext            (episode-level subdir)
    #   3. assets/{asset_id}.ext                       (episode-level root, last resort)
    # Also tries short form in each dir — strip "char-" prefix and "-vN" suffix
    # so amunhotep.png matches asset_id char-amunhotep-v1
    project_id    = merged.get("project_id", "")
    proj_char_dir = PIPE_DIR / "projects" / project_id / "characters"
    char_search   = [proj_char_dir, assets_root / "characters", assets_root]
    for pack in merged.get("character_packs", []):
        aid = pack["asset_id"]
        lt  = pack.get("license_type", "proprietary_cleared")
        f   = search_dirs(char_search, aid, IMAGE_EXTS + AUDIO_EXTS)
        if not f:
            short = re.sub(r"^char-", "", re.sub(r"-v\d+$", "", aid))
            if short != aid:
                f = search_dirs(char_search, short, IMAGE_EXTS + AUDIO_EXTS)
        if f:
            items.append(_resolved(aid, "character", f, lt))
            n_found += 1
        else:
            items.append(_placeholder(aid, "character"))
            n_missing += 1

    # ── 2. Backgrounds ───────────────────────────────────────────────────────
    # Resolution priority:
    #   0. User-selected stock media (selections.json)  — highest, skips fs scan
    #   1. projects/{project_id}/backgrounds/           (project-level, shared — FIRST)
    #   2. assets/{asset_id}.ext                        (episode-level root)
    #   3. assets/backgrounds/{asset_id}.ext            (episode-level backgrounds subdir)
    #
    # Filesystem scan now includes VIDEO_EXTS so downloaded stock clips that
    # were placed manually in the assets tree are also discovered.
    proj_bg_dir = PIPE_DIR / "projects" / project_id / "backgrounds"
    bg_search   = [proj_bg_dir, assets_root, assets_root / "backgrounds"]
    for bg in merged.get("backgrounds", []):
        aid = bg.get("asset_id") or bg["item_id"]
        lt  = bg.get("license_type", "proprietary_cleared")

        # 0. User selection from VC editor (highest priority)
        if selections and aid in selections:
            sel = selections[aid]
            if "per_shot" in sel:
                # Per-shot format (version 2/3): emit one item per (asset_id, shot_id[, segment_index])
                for shot_id, shot_sel in sel["per_shot"].items():
                    if "segments" in shot_sel:
                        # v3: emit one item per segment
                        for seg_idx, seg in enumerate(shot_sel["segments"]):
                            abs_path = seg["abs_path"]
                            if abs_path.is_file():
                                abs_path = _ensure_hires(abs_path, no_hires=no_hires)
                                item = _resolved_stock(
                                    aid, abs_path,
                                    seg.get("media_type", "image"),
                                    seg.get("url", ""),
                                )
                                item["shot_id"]       = shot_id
                                item["segment_index"] = seg_idx
                                item["duration_sec"]  = seg.get("duration_sec")
                                item["hold_sec"]      = seg.get("hold_sec")
                                if seg.get("animation_type"):
                                    item["animation_type"] = seg["animation_type"]
                                items.append(item)
                                n_found += 1
                            else:
                                print(f"  [WARN] Segment {aid!r}/{shot_id}[{seg_idx}] not found: {abs_path}")
                    else:
                        # v2: single item per shot
                        abs_path = shot_sel["abs_path"]
                        if abs_path.is_file():
                            abs_path = _ensure_hires(abs_path, no_hires=no_hires)
                            item = _resolved_stock(
                                aid, abs_path,
                                shot_sel.get("media_type", "image"),
                                shot_sel.get("url", ""),
                            )
                            item["shot_id"] = shot_id
                            items.append(item)
                            n_found += 1
                        else:
                            print(f"  [WARN] Per-shot selection {aid!r}/{shot_id} not found: {abs_path}")
                continue
            else:
                # Old format: single selection per background
                abs_path = sel["abs_path"]
                if abs_path.is_file():
                    abs_path = _ensure_hires(abs_path, no_hires=no_hires)
                    items.append(_resolved_stock(
                        aid, abs_path,
                        sel.get("media_type", "image"),
                        sel.get("url", ""),
                    ))
                    n_found += 1
                    continue
                print(f"  [WARN] Selection for {aid!r} not found on disk: {abs_path}")
                # fall through to filesystem scan

        # 1-3. Filesystem scan (images, videos, audio)
        f = search_dirs(bg_search, aid, IMAGE_EXTS + VIDEO_EXTS + AUDIO_EXTS)
        if f:
            items.append(_resolved(aid, "background", f, lt))
            n_found += 1
        else:
            items.append(_placeholder(aid, "background"))
            n_missing += 1

    # ── 3. VO (locale-specific path) ─────────────────────────────────────────
    # Path: assets/{locale}/audio/vo/{item_id}.ext
    if locale:
        vo_dir = assets_root / locale / "audio" / "vo"
    else:
        vo_dir = assets_root / "audio" / "vo"
    for vo in merged.get("vo_items", []):
        aid = vo["item_id"]
        lt  = vo.get("license_type", "generated_local")
        f   = search_dirs([vo_dir], aid, AUDIO_EXTS)
        if f:
            items.append(_resolved(aid, "vo", f, lt))
            n_found += 1
        else:
            items.append(_placeholder(aid, "vo"))
            n_missing += 1

    # ── 4. SFX ───────────────────────────────────────────────────────────────
    # Path: assets/sfx/{item_id}.ext
    sfx_dir = assets_root / "sfx"
    for sfx in merged.get("sfx_items", []):
        aid = sfx["item_id"]
        f   = search_dirs([sfx_dir], aid, AUDIO_EXTS)
        if f:
            items.append(_resolved_sfx(aid, f, sfx))
            n_found += 1
        else:
            items.append(_placeholder(aid, "sfx"))
            n_missing += 1

    # ── 5. Music ─────────────────────────────────────────────────────────────
    # Path: assets/music/{item_id}.ext
    music_dir = assets_root / "music"
    for music in merged.get("music_items", []):
        aid = music["item_id"]
        lt  = music.get("license_type", "generated_local")
        f   = search_dirs([music_dir], aid, AUDIO_EXTS)
        if f:
            items.append(_resolved(aid, "music", f, lt))
            n_found += 1
        else:
            items.append(_placeholder(aid, "music"))
            n_missing += 1

    print(f"  Total items   : {len(items)}")
    print(f"  Resolved (✓)  : {n_found}")
    print(f"  Placeholder (✗): {n_missing}")
    return items


# ── Output path ───────────────────────────────────────────────────────────────

def derive_output_path(manifest_path: Path, locale: str) -> Path:
    """Default: AssetManifest.media.{locale}.json next to the merged manifest."""
    return manifest_path.parent / f"AssetManifest.media.{locale}.json"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resolve AssetManifest_merged.{locale}.json into "
                    "AssetManifest.media.{locale}.json using local file paths only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File path conventions (relative to --assets-root):
  VO        : {locale}/audio/vo/{item_id}.wav
  Music     : music/{item_id}.wav
  SFX       : sfx/{item_id}.wav
  Background: projects/{project_id}/backgrounds/ (project-level, shared — FIRST)
              then assets/{asset_id}.png, assets/backgrounds/{asset_id}.png
  Character : projects/{project_id}/characters/ (project-level, shared — FIRST)
              then assets/characters/{asset_id}.png, assets/{asset_id}.png
""",
    )
    p.add_argument(
        "--manifest", required=True, metavar="PATH",
        help="Path to AssetManifest_merged.{locale}.json (locale_scope='merged').",
    )
    p.add_argument(
        "--assets-root", default=None, metavar="PATH",
        help="Root directory containing resolved asset files. "
             "Default: {episode_dir}/assets/",
    )
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="Output path. Default: AssetManifest.media.{locale}.json "
             "next to the input manifest.",
    )
    p.add_argument(
        "--selections", default=None, metavar="PATH",
        help="Path to selections.json written by the VC editor Media tab. "
             "When present, user-chosen stock media (Pexels / Pixabay) takes "
             "priority over the regular filesystem scan for background assets.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any asset is a placeholder (useful in CI).",
    )
    p.add_argument(
        "--no-hires-download",
        action="store_true",
        default=False,
        help="Skip high-res download; use preview assets directly (for dry runs / offline use).",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        merged = json.load(f)

    # Guard: must be a merged manifest
    locale_scope = merged.get("locale_scope")
    if locale_scope != "merged":
        raise SystemExit(
            f"[ERROR] --manifest has locale_scope='{locale_scope}'. "
            "Expected 'merged'. Pass AssetManifest_merged.{{locale}}.json."
        )

    locale = merged.get("locale", "")
    if not locale:
        raise SystemExit(
            "[ERROR] Merged manifest is missing 'locale' field. "
            "Cannot derive output filename."
        )

    # Paths
    episode_dir = manifest_path.parent
    assets_root = Path(args.assets_root).resolve() if args.assets_root \
                  else (episode_dir / "assets")
    out_path    = Path(args.out).resolve() if args.out \
                  else derive_output_path(manifest_path, locale)

    if not assets_root.is_dir():
        print(f"[WARN] assets-root does not exist: {assets_root} — all assets will be placeholders")

    # Load optional user-selections (stock media chosen via VC editor)
    selections: dict | None = None
    if args.selections:
        sel_path   = Path(args.selections).resolve()
        selections = _load_selections(sel_path)
    else:
        # Auto-detect: look for selections.json in episode_dir/assets/media/
        auto_sel = episode_dir / "assets" / "media" / "selections.json"
        if auto_sel.exists():
            selections = _load_selections(auto_sel)

    n_selections = len(selections) if selections else 0

    print("=" * 60)
    print("  resolve_assets")
    print(f"  Manifest    : {manifest_path.name}")
    print(f"  Locale      : {locale}")
    print(f"  Assets root : {assets_root}")
    print(f"  Selections  : {n_selections} item(s) from VC editor"
          if n_selections else "  Selections  : none")
    print(f"  Output      : {out_path}")
    print("=" * 60)

    # Resolve
    items = resolve_all(merged, assets_root, selections, no_hires=args.no_hires_download)

    # Build output document
    output = {
        "schema_id":      "AssetManifest.media",
        "schema_version": "1.0.0",
        "manifest_id":    merged.get("manifest_id", ""),
        "project_id":     merged.get("project_id", ""),
        "producer":       PRODUCER,
        "generated_at":   DETERMINISTIC_TS,
        "items":          items,
    }

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n  [OK] Written: {out_path}")

    # Strict mode: fail on any placeholder
    if args.strict:
        placeholders = [i for i in items if i["is_placeholder"]]
        if placeholders:
            print(
                f"[ERROR] --strict: {len(placeholders)} placeholder(s) found:",
                file=sys.stderr,
            )
            for item in placeholders[:10]:
                print(f"  {item['uri']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
