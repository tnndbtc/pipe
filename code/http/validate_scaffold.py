#!/usr/bin/env python3
# =============================================================================
# validate_scaffold.py — FIX2: Post-validation for the scaffold → LLM complete pattern
# =============================================================================
#
# Verifies that an LLM-completed output JSON:
#   A) Contains no residual __FILL__ tokens (the LLM filled everything in).
#   B) Has not drifted from the pre-filled values baked into the scaffold.
#
# Usage:
#   python3 validate_scaffold.py --scaffold path/to/scaffold.json \
#                                --output   path/to/output.json
#   python3 validate_scaffold.py --scaffold path/to/scaffold.json \
#                                --output   path/to/output.json \
#                                --warn-only
#
# Requirements: stdlib only (json, pathlib, argparse, re, sys)
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Path traversal helpers ────────────────────────────────────────────────────

def find_fill_tokens(obj, path: str = ""):
    """
    Recursively walk obj. Yield the JSON path of every string value that
    starts with '__FILL__'.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from find_fill_tokens(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_fill_tokens(v, f"{path}[{i}]")
    elif isinstance(obj, str) and obj.startswith("__FILL__"):
        yield path


def collect_prefilled_leaves(obj, path: str = ""):
    """
    Recursively walk obj. Yield (path, value) for every non-__FILL__ leaf.
    Leaves are strings, numbers, booleans, and None — not dicts or lists.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from collect_prefilled_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from collect_prefilled_leaves(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if not obj.startswith("__FILL__"):
            yield path, obj
    else:
        # int, float, bool, None
        yield path, obj


def resolve_path(obj, path: str):
    """
    Navigate obj using dot/bracket notation path string.

    Examples:
      "title"          → obj["title"]
      "cast[0].role"   → obj["cast"][0]["role"]
      "[2]"            → obj[2]
    """
    parts = re.split(r'\.(?![^\[]*\])', path)
    cur = obj
    for part in parts:
        if not part:
            continue
        # Handle "key[idx]" — e.g. "items[2]"
        m = re.match(r'^([^\[]+)\[(\d+)\]$', part)
        if m:
            cur = cur[m.group(1)][int(m.group(2))]
        elif '[' in part:
            # Bare "[idx]" — index only
            idx = int(re.search(r'\[(\d+)\]', part).group(1))
            cur = cur[idx]
        else:
            cur = cur[part]
    return cur


# ── Validation logic ──────────────────────────────────────────────────────────

def check_fill_tokens(output: dict) -> list[str]:
    """Part A: collect all residual __FILL__ token paths in output."""
    return list(find_fill_tokens(output))


def check_prefilled_drift(scaffold: dict, output: dict) -> list[tuple[str, object, object]]:
    """
    Part B: collect (path, expected, actual) tuples where a pre-filled scaffold
    value differs from the corresponding value in output.

    Skips paths that cannot be resolved in output (the LLM may have legitimately
    restructured a portion — those are caught by schema validation upstream).
    """
    diffs = []
    for path, expected in collect_prefilled_leaves(scaffold):
        try:
            actual = resolve_path(output, path)
        except (KeyError, IndexError, TypeError):
            # Path missing in output — treat as drift
            actual = "<MISSING>"
        if actual != expected:
            diffs.append((path, expected, actual))
    return diffs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "FIX2 — post-validation for the scaffold → LLM complete pattern. "
            "Checks for residual __FILL__ tokens and pre-filled field drift."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scaffold", required=True, metavar="PATH",
                   help="Scaffold JSON with __FILL__ markers (the LLM input).")
    p.add_argument("--output", required=True, metavar="PATH",
                   help="LLM-completed output JSON to validate.")
    p.add_argument("--warn-only", action="store_true",
                   help="Print warnings instead of exiting 1 on drift (Part B only). "
                        "Residual __FILL__ tokens (Part A) always exit 1.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    scaffold_path = Path(args.scaffold).resolve()
    output_path   = Path(args.output).resolve()

    for label, path in [("--scaffold", scaffold_path), ("--output", output_path)]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        scaffold = load_json(scaffold_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load scaffold: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        output = load_json(output_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load output: {exc}", file=sys.stderr)
        sys.exit(1)

    exit_code = 0

    # ── Part A: Residual __FILL__ check ───────────────────────────────────────
    fill_paths = check_fill_tokens(output)
    if fill_paths:
        print(f"[ERROR] {len(fill_paths)} residual __FILL__ token(s) found in output:")
        for fp in fill_paths:
            print(f"  {fp}")
        exit_code = 1
    else:
        print("✓ No __FILL__ tokens remaining")

    # ── Part B: Pre-filled field drift check ──────────────────────────────────
    diffs = check_prefilled_drift(scaffold, output)
    if diffs:
        prefix = "[WARN]" if args.warn_only else "[ERROR]"
        print(f"{prefix} {len(diffs)} pre-filled field(s) drifted from scaffold:")
        for path, expected, actual in diffs:
            print(f"  {path}: {expected!r} → {actual!r}")
        if args.warn_only:
            print("[WARN] --warn-only: continuing despite drift")
        else:
            exit_code = 1
    else:
        print("✓ No pre-filled field drift detected")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
