#!/usr/bin/env python3
"""Contract verification tool — validates all JSON Schema files in contracts/schemas/.

For each *.v1.json file found, checks:
  1. Parses as valid JSON
  2. Is a valid JSON Schema (Draft 7)

Standalone script; also importable for tests.

Usage:
    python contracts/tools/verify_contracts.py [--schemas-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

# Default schemas dir is sibling of this file's parent (i.e., contracts/schemas/)
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


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
    """Discover all *.v1.json files in schemas_dir, validate each, return (errors, count).

    Prints per-file PASS/FAIL lines.
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


def main() -> None:
    """Parse --schemas-dir, call run_checks, print RESULT summary, sys.exit."""
    parser = argparse.ArgumentParser(
        description="Validate all *.v1.json schema files in contracts/schemas/."
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=SCHEMAS_DIR,
        help="Path to schemas directory (default: auto-detected from script location)",
    )
    args = parser.parse_args()

    errors, count = run_checks(args.schemas_dir)

    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s) across {count} schema(s))")
        sys.exit(1)
    else:
        print(f"RESULT: PASS ({count}/{count} schemas valid)")
        sys.exit(0)


if __name__ == "__main__":
    main()
