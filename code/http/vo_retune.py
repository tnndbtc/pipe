"""vo_retune.py — Core shared module for selective VO re-tuning.

Architecture: this module is the single source of truth for retune logic.
  - retune_vo.py        → thin CLI wrapper
  - /api/retune_vo      → thin API wrapper in test_server.py

Design invariants enforced here:
  - Manifest writes are ATOMIC (write-to-.tmp then os.replace)
  - Per-item patch-synthesize-commit loop (never patch-all-then-synthesize-all)
  - Manifest/WAV consistency: on synthesis failure, manifest is restored from snapshot
  - zero-patch re-synthesis is valid (no patch fields = just re-run TTS)
  - start_sec / end_sec are read-only in v1; azure_voice is editable

See todo.txt for full design documentation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent          # code/http/
PIPE_DIR  = _THIS_DIR.parent.parent                  # repo root
CODE_DIR  = _THIS_DIR                                # code/http/

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDITABLE_FIELDS = frozenset({
    "text",
    "azure_voice",
    "azure_style",
    "azure_style_degree",
    "azure_rate",
    "azure_pitch",
    "azure_break_ms",
})

BLOCKED_FIELDS = frozenset({
    "speaker_id",
    "item_id",
    "start_sec",
    "end_sec",
    "duration_sec",
})

# Regex patterns for format validation
_RATE_RE  = re.compile(r'^[+-]\d+(\.\d+)?%$|^0%$')
_PITCH_RE = re.compile(r'^[+-]\d+(\.\d+)?%$')

# Duration drift threshold for ⚠️  warning
DURATION_WARN_THRESHOLD = 0.3   # seconds


# ===========================================================================
# Public API
# ===========================================================================

def load_retune_context(manifest_path: str, locale: str) -> dict:
    """Load AssetManifest and build item index.

    Returns a context dict:
      manifest_path  str            path to AssetManifest.{locale}.json
      draft_path     str | None     path to AssetManifest.{locale}.json (draft copy, if present)
      locale         str            e.g. "en", "zh-Hans"
      manifest       dict           parsed merged manifest (mutable)
      items_idx      dict           { item_id -> vo_item } for fast lookup
    """
    path = Path(manifest_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest  = json.loads(path.read_text(encoding="utf-8"))
    items_idx = {it["item_id"]: it for it in manifest.get("vo_items", [])}

    # Locate draft manifest (for durable patching that survives Stage 9 re-runs)
    # Naming convention: AssetManifest.{locale}.json (merged)
    #                 →  AssetManifest.{locale}.json  (draft copy, written by Stage 5)
    draft_path_str = str(path).replace("_merged.", "_draft.")
    draft_path = Path(draft_path_str)

    return {
        "manifest_path": str(path),
        "draft_path":    str(draft_path) if draft_path.exists() else None,
        "locale":        locale,
        "manifest":      manifest,
        "items_idx":     items_idx,
    }


def resolve_target_items(
    context: dict,
    item_ids: Optional[list] = None,
    scene:    Optional[str]  = None,
) -> list:
    """Resolve target vo_items from item_ids list or scene selector.

    Returns list of vo_item dicts in manifest index order (stable, deterministic).
    Raises ValueError on invalid or empty selection.
    """
    if item_ids is not None and scene is not None:
        raise ValueError("Supply exactly one of item_ids or scene, not both.")
    if item_ids is None and scene is None:
        raise ValueError("Supply exactly one of item_ids or scene.")

    all_items = context["manifest"].get("vo_items", [])
    items_idx = context["items_idx"]

    if item_ids is not None:
        # Explicit list — validate all exist first, then return in manifest order
        missing = [iid for iid in item_ids if iid not in items_idx]
        if missing:
            raise ValueError(f"item_ids not found in manifest: {missing}")
        id_set = set(item_ids)
        return [it for it in all_items if it["item_id"] in id_set]

    # Scene selector: match in manifest index order (not sorted by item_id)
    result = []
    for it in all_items:
        # 1. Try scene_id field (written by gen_vo_manifest.py)
        item_scene = it.get("scene_id")
        # 2. Fallback: extract from item_id pattern, e.g. "vo-s01e02-sc02-003" → "sc02"
        if not item_scene:
            m = re.search(r'(sc\d+)', it["item_id"])
            if m:
                item_scene = m.group(1)
        if item_scene == scene:
            result.append(it)
        elif not item_scene:
            print(f"  WARNING: cannot determine scene for {it['item_id']} — skipping")

    if not result:
        raise ValueError(f"Scene '{scene}' resolved no items in manifest.")
    return result


def validate_patches(patches: dict) -> None:
    """Validate patch field names and values.

    Raises ValueError with a clear message on any validation failure.
    Empty patches dict is valid (zero-patch re-synthesis).
    """
    if not patches:
        return  # zero-patch re-synthesis is explicitly allowed

    for field, value in patches.items():
        if field in BLOCKED_FIELDS:
            raise ValueError(
                f"Field '{field}' is not editable in v1 retune "
                f"(blocked: {sorted(BLOCKED_FIELDS)})."
            )
        if field not in EDITABLE_FIELDS:
            raise ValueError(
                f"Unknown field '{field}'. "
                f"Allowed editable fields: {sorted(EDITABLE_FIELDS)}"
            )
        if field == "text":
            if not str(value).strip():
                raise ValueError("'text' must be non-empty after trimming whitespace.")

        elif field == "azure_style_degree":
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"'azure_style_degree' must be a positive number, got {value!r}"
                )
            if v <= 0:
                raise ValueError(
                    f"'azure_style_degree' must be > 0, got {v}"
                )

        elif field == "azure_rate":
            if not _RATE_RE.match(str(value)):
                raise ValueError(
                    f"'azure_rate' must match +N%/-N%/0% (e.g. '+10%', '-5%', '0%'), "
                    f"got {value!r}"
                )

        elif field == "azure_pitch":
            if not _PITCH_RE.match(str(value)):
                raise ValueError(
                    f"'azure_pitch' must match +N%/-N% (e.g. '+5%', '-3%'), "
                    f"got {value!r}"
                )

        elif field == "azure_break_ms":
            try:
                v = int(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"'azure_break_ms' must be a non-negative integer, got {value!r}"
                )
            if v < 0:
                raise ValueError(
                    f"'azure_break_ms' must be >= 0, got {v}"
                )


def write_retune_log(context: dict, results: list, changes: dict) -> None:
    """Append one JSON record to assets/meta/retune_vo_log.jsonl.

    Log is append-only, one JSON object per line, never truncated.
    """
    ep_dir   = Path(context["manifest_path"]).parent
    log_path = ep_dir / "assets" / "meta" / "retune_vo_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    succeeded = [r for r in results if r["status"] == "ok"]
    failed    = [r for r in results if r["status"] == "error"]

    if   failed and succeeded: overall = "partial"
    elif failed:               overall = "error"
    else:                      overall = "ok"

    paths = [context["manifest_path"]]
    if context.get("draft_path"):
        paths.append(context["draft_path"])

    record = {
        "ts":             datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "locale":         context["locale"],
        "item_ids":       [r["item_id"] for r in results],
        "changes":        changes,
        "manifest_paths": paths,
        "wav_paths":      [r["wav_path"] for r in succeeded if r.get("wav_path")],
        "status":         overall,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===========================================================================
# Main orchestrator
# ===========================================================================

def retune_vo_items(
    manifest_path:    str,
    locale:           str,
    item_ids:         Optional[list] = None,
    scene:            Optional[str]  = None,
    patches:          Optional[dict] = None,
    per_item_patches: Optional[dict] = None,
    dry_run:          bool = False,
    backup:           bool = True,
) -> list:
    """Orchestrate selective VO retune for one or more items.

    Target selection (exactly one required):
      item_ids  — explicit list of item IDs
      scene     — all items in this scene (resolved in manifest index order)

    Patch modes (both optional; may be combined):
      patches          — global patch applied to ALL target items (CLI pattern)
      per_item_patches — per-item patch dict keyed by item_id (API pattern)
      Merge rule: per_item_patches[item_id] fields override global patches for
                  that item; fields absent from per_item_patches fall back to
                  global patches.
      Both None → zero-patch re-synthesis (re-run TTS with current parameters).

    Returns list of result dicts (one per target item), in manifest index order.
    """
    # 1. Load context + resolve targets
    context = load_retune_context(manifest_path, locale)
    targets = resolve_target_items(context, item_ids=item_ids, scene=scene)

    # 2. Build effective per-item patch map
    #    Merge global patches + per_item_patches (per-item overrides per field)
    global_patch   = patches          or {}
    per_item       = per_item_patches or {}
    effective: dict[str, dict] = {}
    for it in targets:
        iid = it["item_id"]
        merged_patch = {**global_patch, **(per_item.get(iid, {}))}
        effective[iid] = merged_patch

    # 3. Validate ALL patches before touching any file
    for iid, patch in effective.items():
        try:
            validate_patches(patch)
        except ValueError as e:
            raise ValueError(f"Validation failed for {iid}: {e}") from e

    # 4. Dry-run mode: show what would change, no writes
    if dry_run:
        results = []
        for it in targets:
            iid   = it["item_id"]
            patch = effective[iid]
            results.append({
                "item_id":             iid,
                "status":              "dry_run",
                "patch":               patch,
                "before_duration_sec": _wav_duration(context, iid),
                "after_duration_sec":  None,
                "duration_delta_sec":  None,
                "duration_warn":       False,
                "wav_path":            None,
                "error":               None,
            })
        return results

    # 5. Backup manifests (always, timestamped, before any writes)
    if backup:
        _backup_manifests(context)

    # 6. Per-item patch-synthesize-commit loop
    #    NEVER patch-all then synthesize-all — that violates the
    #    manifest/WAV consistency invariant on partial failure.
    results = []
    for it in targets:
        iid    = it["item_id"]
        patch  = effective[iid]
        result = _process_one_item(context, it, patch, backup=backup)
        results.append(result)

    # 7. Write audit log
    all_changes: dict = {}
    for patch in effective.values():
        all_changes.update(patch)
    write_retune_log(context, results, all_changes)

    return results


# ===========================================================================
# Private helpers
# ===========================================================================

def _process_one_item(context: dict, item: dict, patch: dict, backup: bool) -> dict:
    """Per-item patch → synthesize → commit-or-revert.

    Maintains manifest/WAV consistency invariant:
      synthesis succeeds → manifest patched,   WAV updated   ✓
      synthesis fails    → manifest restored,  WAV unchanged  ✓
    """
    iid        = item["item_id"]
    before_dur = _wav_duration(context, iid)

    # Optional WAV backup before overwriting
    if backup:
        _backup_wav(context, iid)

    # Snapshot current manifest fields for this item (in memory)
    snapshot_merged = _snapshot_item(context["manifest"], iid)
    snapshot_draft  = None
    draft_manifest  = None

    if context.get("draft_path"):
        draft_manifest = json.loads(
            Path(context["draft_path"]).read_text(encoding="utf-8")
        )
        snapshot_draft = _snapshot_item(draft_manifest, iid)

    # Apply patch to in-memory manifests (only if patch is non-empty)
    if patch:
        _patch_item_in_manifest(context["manifest"], iid, patch)
        if draft_manifest:
            _patch_item_in_manifest(draft_manifest, iid, patch)

        # Atomically write both manifests to disk
        # _merged patch → enables immediate re-synthesis without a Stage 9 re-run
        # _draft  patch → survives any future Stage 9 re-run (the durable one)
        _write_manifest_atomic(context["manifest_path"], context["manifest"])
        if draft_manifest and context.get("draft_path"):
            _write_manifest_atomic(context["draft_path"], draft_manifest)

    # Call gen_tts_cloud.py --asset-id --force
    success, error_msg = _resynth_item(context, iid)

    if success:
        after_dur  = _wav_duration(context, iid)
        delta      = round((after_dur or 0.0) - (before_dur or 0.0), 3)
        dur_warn   = abs(delta) > DURATION_WARN_THRESHOLD
        return {
            "item_id":             iid,
            "status":              "ok",
            "before_duration_sec": before_dur,
            "after_duration_sec":  after_dur,
            "duration_delta_sec":  delta,
            "duration_warn":       dur_warn,
            "wav_path":            _wav_path(context, iid),
            "error":               None,
        }
    else:
        # Revert manifests to pre-patch state (restore from snapshot)
        if patch:
            _restore_item_in_manifest(context["manifest"], iid, snapshot_merged)
            _write_manifest_atomic(context["manifest_path"], context["manifest"])
            if draft_manifest and snapshot_draft and context.get("draft_path"):
                _restore_item_in_manifest(draft_manifest, iid, snapshot_draft)
                _write_manifest_atomic(context["draft_path"], draft_manifest)

        return {
            "item_id":             iid,
            "status":              "error",
            "before_duration_sec": before_dur,
            "after_duration_sec":  before_dur,   # WAV unchanged
            "duration_delta_sec":  0.0,
            "duration_warn":       False,
            "wav_path":            None,
            "error":               error_msg,
        }


def _patch_item_in_manifest(manifest: dict, item_id: str, patch: dict) -> None:
    """Apply patch fields to the vo_item in manifest (in-place, in memory).

    Field routing:
      "text"  → vo_item["text"]  (top-level)
      others  → vo_item["tts_prompt"][field]
    """
    for it in manifest.get("vo_items", []):
        if it["item_id"] == item_id:
            for field, value in patch.items():
                if field == "text":
                    it["text"] = value
                else:
                    it.setdefault("tts_prompt", {})[field] = value
            return
    raise KeyError(f"item_id not found in manifest: {item_id}")


def _restore_item_in_manifest(manifest: dict, item_id: str, snapshot: dict) -> None:
    """Restore a vo_item's fields from a snapshot (in-place, in memory)."""
    for it in manifest.get("vo_items", []):
        if it["item_id"] == item_id:
            it["text"]       = snapshot["text"]
            it["tts_prompt"] = dict(snapshot["tts_prompt"])
            return
    raise KeyError(f"item_id not found in manifest: {item_id}")


def _snapshot_item(manifest: dict, item_id: str) -> dict:
    """Return a shallow copy of the fields that retune may modify."""
    for it in manifest.get("vo_items", []):
        if it["item_id"] == item_id:
            return {
                "text":       it.get("text", ""),
                "tts_prompt": dict(it.get("tts_prompt", {})),
            }
    raise KeyError(f"item_id not found in manifest: {item_id}")


def _write_manifest_atomic(path: str, manifest: dict) -> None:
    """Write manifest JSON atomically: write to .tmp then os.replace.

    If interrupted mid-write, the original is never partially overwritten.
    An orphaned .tmp file is benign — it can be deleted safely.
    """
    tmp     = path + ".tmp"
    content = json.dumps(manifest, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _resynth_item(context: dict, item_id: str) -> tuple[bool, Optional[str]]:
    """Invoke gen_tts_cloud.py --asset-id --force for a single item.

    Returns (success, error_message).
    """
    cmd = [
        "python3", str(CODE_DIR / "gen_tts_cloud.py"),
        "--manifest", context["manifest_path"],
        "--asset-id", item_id,
        "--force",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PIPE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return True, None
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"gen_tts_cloud.py exited with code {proc.returncode}"

    except subprocess.TimeoutExpired:
        return False, "gen_tts_cloud.py timed out after 120 s"
    except Exception as exc:
        return False, str(exc)


def _wav_path(context: dict, item_id: str) -> Optional[str]:
    """Return the expected WAV path for item_id, or None if file absent."""
    ep_dir = Path(context["manifest_path"]).parent
    locale = context["locale"]
    p      = ep_dir / "assets" / locale / "audio" / "vo" / f"{item_id}.wav"
    return str(p) if p.exists() else None


def _wav_duration(context: dict, item_id: str) -> Optional[float]:
    """Return WAV duration in seconds, or None if file absent or unreadable."""
    p = _wav_path(context, item_id)
    if not p:
        return None
    try:
        with wave.open(p) as wf:
            return round(wf.getnframes() / wf.getframerate(), 3)
    except Exception:
        return None


def _backup_manifests(context: dict) -> None:
    """Backup _merged and _draft manifest files with a timestamp suffix."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for key in ("manifest_path", "draft_path"):
        src = context.get(key)
        if src and Path(src).exists():
            shutil.copy2(src, f"{src}.bak.{ts}")


def _backup_wav(context: dict, item_id: str) -> None:
    """Backup the WAV file for item_id with a timestamp suffix."""
    p = _wav_path(context, item_id)
    if p:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(p, f"{p}.bak.{ts}")
