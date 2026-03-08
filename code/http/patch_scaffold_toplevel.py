#!/usr/bin/env python3
# =============================================================================
# patch_scaffold_toplevel.py
#
# Restores top-level pre-filled scalar fields from a scaffold JSON into the
# LLM-completed output JSON.  Called between fill_and_run and validate_scaffold
# to guard against LLMs silently dropping structural metadata fields such as
# script_ref, schema_id, shotlist_id, timing_lock_hash, etc.
#
# Only top-level scalar fields (str, int, float, bool) are patched.
# The "shots" / "items" array and any nested objects are left untouched —
# those are the LLM's work.
#
# Usage:
#   python3 patch_scaffold_toplevel.py <scaffold.json> <output.json>
#
# Exits 0 always (patching is non-fatal).  Prints what was restored.
# =============================================================================

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <scaffold.json> <output.json>",
              file=sys.stderr)
        return 1

    scaffold_path = Path(sys.argv[1])
    output_path   = Path(sys.argv[2])

    for label, p in [("scaffold", scaffold_path), ("output", output_path)]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            return 1

    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    output   = json.loads(output_path.read_text(encoding="utf-8"))

    # Scalars-only array keys to skip entirely (LLM-filled arrays).
    SKIP_KEYS = {"shots", "items", "scenes", "characters", "cast"}

    patched: list[str] = []
    for key, value in scaffold.items():
        if key in SKIP_KEYS:
            continue
        # Only restore scalar pre-filled values (not __FILL__ markers).
        if not isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, str) and value.startswith("__FILL__"):
            continue
        if output.get(key) != value:
            output[key] = value
            patched.append(key)

    if patched:
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  [patch] restored top-level field(s) from scaffold: {patched}")
    else:
        print("  [patch] no top-level drift to restore")

    return 0


if __name__ == "__main__":
    sys.exit(main())
