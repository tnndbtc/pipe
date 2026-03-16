#!/usr/bin/env python3
"""Contract verification tool — two modes in one script.

MODE 1 — Schema validation (no positional argument):
    Validates every *.v1.json file in contracts/schemas/ as a well-formed
    JSON Schema (Draft 7).  This is the original behaviour.

    python contracts/tools/verify_contracts.py
    python contracts/tools/verify_contracts.py --schemas-dir PATH

MODE 2 — Data file validation (positional argument given):
    Validates a JSON data file against its matching contract schema.
    The schema is resolved in this priority order:
      1. Explicit schema name passed as second positional arg.
      2. "schema_id" field inside the data file.
      3. First dot-separated component of the filename
         (e.g. "AssetManifest.en.json" → "AssetManifest").

    python contracts/tools/verify_contracts.py path/to/AssetManifest.en.json
    python contracts/tools/verify_contracts.py path/to/data.json AssetManifest

Exit codes:
    0 — all checks passed
    1 — one or more checks failed, or an error occurred

Called with || true in run.sh so failures are logged but never block the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

# Default schemas dir: contracts/schemas/ (sibling of this file's parent dir)
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


# ── Mode 1: schema meta-validation ────────────────────────────────────────────

def check_schema_file(schema_path: Path) -> list[str]:
    """Validate that *schema_path* is a well-formed JSON Schema (Draft 7).

    Returns a list of error strings (empty if valid).
    """
    try:
        schema = json.loads(schema_path.read_bytes())
    except json.JSONDecodeError as exc:
        return [f"JSON_PARSE_ERROR: {exc}"]

    try:
        jsonschema.Draft7Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        return [f"INVALID_SCHEMA: {exc.message}"]

    return []


def run_checks(schemas_dir: Path) -> tuple[list[str], int]:
    """Discover all *.v1.json files in schemas_dir, validate each.

    Returns (all_errors, count).  Prints per-file PASS/FAIL lines.
    """
    schema_paths = sorted(schemas_dir.glob("*.v1.json"))
    count = len(schema_paths)

    if count == 0:
        print(f"No *.v1.json files found in {schemas_dir}")
        return [], 0

    all_errors: list[str] = []

    for schema_path in schema_paths:
        name = schema_path.name
        errors = check_schema_file(schema_path)
        if errors:
            for err in errors:
                print(f"FAIL   {name}: {err}")
            all_errors.extend(errors)
        else:
            print(f"PASS   {name}")

    return all_errors, count


# ── Mode 2: data file validation ───────────────────────────────────────────────

def _detect_schema_name(data_path: Path, data: dict) -> str | None:
    """Infer the schema name for *data_path* without an explicit hint.

    Priority:
      1. "schema_id" field in the data document.
      2. First dot-separated component of the filename
         (e.g. "AssetManifest.en.json" → "AssetManifest").
    """
    schema_id = data.get("schema_id")
    if schema_id and isinstance(schema_id, str):
        return schema_id

    # e.g. "AssetManifest.en.json" → ["AssetManifest", "en", "json"]
    stem_parts = data_path.name.split(".")
    if stem_parts:
        return stem_parts[0]

    return None


def validate_data_file(
    data_path: Path,
    schema_name: str | None = None,
    schemas_dir: Path = SCHEMAS_DIR,
) -> list[str]:
    """Validate *data_path* against *schema_name*.v1.json.

    If *schema_name* is None, it is auto-detected from the data file.
    Returns a list of error strings (empty if valid).
    """
    # Load data file first (needed for auto-detect even if schema_name is given)
    try:
        data = json.loads(data_path.read_bytes())
    except FileNotFoundError:
        return [f"DATA_NOT_FOUND: {data_path}"]
    except json.JSONDecodeError as exc:
        return [f"DATA_PARSE_ERROR: {exc}"]

    # Resolve schema name
    if not schema_name:
        schema_name = _detect_schema_name(data_path, data)
    if not schema_name:
        return [f"SCHEMA_UNKNOWN: cannot detect schema for {data_path.name} "
                "(no schema_id field and filename does not start with a known name)"]

    schema_path = schemas_dir / f"{schema_name}.v1.json"
    if not schema_path.exists():
        return [f"SCHEMA_NOT_FOUND: {schema_path}"]

    try:
        schema = json.loads(schema_path.read_bytes())
    except json.JSONDecodeError as exc:
        return [f"SCHEMA_PARSE_ERROR: {exc}"]

    validator = jsonschema.Draft7Validator(schema)
    # Sort by string representation of path to avoid Python 3 mixed int/str comparisons.
    errors = sorted(validator.iter_errors(data), key=lambda e: str(list(e.path)))
    return [e.message for e in errors]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_file",
        nargs="?",
        metavar="DATA_FILE",
        help="(Mode 2) JSON data file to validate against a contract schema.",
    )
    parser.add_argument(
        "schema_name",
        nargs="?",
        metavar="SCHEMA_NAME",
        help="(Mode 2) Schema base name, e.g. 'AssetManifest'.  "
             "Auto-detected from schema_id field or filename when omitted.",
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=SCHEMAS_DIR,
        metavar="PATH",
        help="Path to schemas directory (default: auto-detected from script location).",
    )
    args = parser.parse_args()

    schemas_dir: Path = args.schemas_dir

    # ── Mode 2: data file given ────────────────────────────────────────────────
    if args.data_file:
        data_path = Path(args.data_file)
        errors = validate_data_file(data_path, args.schema_name, schemas_dir)
        schema_label = args.schema_name or "(auto)"
        label = f"{data_path.name}  schema={schema_label}"
        if errors:
            print(f"FAIL  {label}")
            for err in errors:
                print(f"  • {err}")
            sys.exit(1)
        else:
            print(f"PASS  {label}")
            sys.exit(0)

    # ── Mode 1: no data file — validate the schemas directory itself ───────────
    errors, count = run_checks(schemas_dir)
    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s) across {count} schema(s))")
        sys.exit(1)
    else:
        print(f"RESULT: PASS ({count}/{count} schemas valid)")
        sys.exit(0)


if __name__ == "__main__":
    main()
