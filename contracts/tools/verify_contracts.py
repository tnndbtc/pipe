#!/usr/bin/env python3
"""Contract verification tool — three modes in one script.

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

MODE 3 — Behavioural self-test (--self-test flag):
    Runs a fixed set of known-good and known-bad sample documents against
    MusicPlan, SfxPlan, and MediaPlan schemas and asserts expected outcomes.
    Covers CV-M1–M7, CV-S1–S6, CV-P1/P2/P5/P6.

    python contracts/tools/verify_contracts.py --self-test
    python contracts/tools/verify_contracts.py --self-test --schemas-dir PATH

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


# ── Mode 3: behavioural self-test ─────────────────────────────────────────────

# Each entry: (case_id, schema_name, doc, expect_valid, description)
# schema_name must match the base name of a *.v1.json file in schemas_dir.
# MusicPlan/SfxPlan use Draft 7; MediaPlan uses Draft 2020-12.
_SELF_TEST_CASES: list[tuple[str, str, dict, bool, str]] = [
    # ── MusicPlan ──────────────────────────────────────────────────────────────
    ("CV-M1", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "loop_selections": {}, "track_volumes": {}, "clip_volumes": {},
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 20.0,
                             "music_asset_id": "cher2",
                             "music_clip_id": "cher2:114.0s-142.1s",
                             "clip_start_sec": 114.0, "clip_duration_sec": 28.1,
                             "duck_db": -12.0, "fade_sec": 0.15}],
    }, True, "valid MusicPlan with shot_overrides passes schema"),

    ("CV-M2", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "music_segments": [{"start_sec": 5.0, "end_sec": 20.0}],
    }, False, "MusicPlan with old music_segments field fails schema (additionalProperties)"),

    ("CV-M3", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "shot_overrides": [{"end_sec": 20.0}],
    }, False, "MusicPlan shot_override missing start_sec fails schema (required)"),

    ("CV-M4", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "shot_overrides": [{"start_sec": 5.0}],
    }, False, "MusicPlan shot_override missing end_sec fails schema (required)"),

    ("CV-M5", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 20.0,
                             "shot_id": "sc01-sh01"}],
    }, False, "MusicPlan shot_override with shot_id fails schema (additionalProperties)"),

    ("CV-M6", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 20.0,
                             "item_id": "music-sc01-sh01"}],
    }, False, "MusicPlan shot_override with item_id fails schema (additionalProperties)"),

    ("CV-M7", "MusicPlan", {
        "schema_id": "MusicPlan", "schema_version": "1.0",
        "loop_selections": {"cher1": {"start_sec": 0, "duration_sec": 30}},
        "track_volumes": {"cher1": -3},
        "clip_volumes": {"cher1:126.0s-155.6s": 2},
        "shot_overrides": [{"start_sec": 30.0, "end_sec": 55.0}],
    }, True, "MusicPlan with loop_selections/track_volumes/clip_volumes and shot_overrides passes schema"),

    # ── SfxPlan ────────────────────────────────────────────────────────────────
    ("CV-S1", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "timing_format": "episode_absolute",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 10.0,
                             "source_file": "assets/sfx/sfx-sc01-sh01-001/ai.mp3",
                             "volume_db": 0.0, "duck_db": 0.0, "fade_sec": 0.0,
                             "clip_id": None, "clip_path": None}],
        "cut_clips": [], "cut_assign": {},
    }, True, "valid SfxPlan with shot_overrides passes schema"),

    ("CV-S2", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "sfx_segments": [{"start_sec": 5.0, "end_sec": 10.0,
                           "source_file": "some.mp3"}],
    }, False, "SfxPlan with old sfx_segments field fails schema (additionalProperties)"),

    ("CV-S3", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "shot_overrides": [{"end_sec": 10.0,
                             "source_file": "assets/sfx/sfx-sc01-sh01-001/ai.mp3"}],
    }, False, "SfxPlan shot_override missing start_sec fails schema (required)"),

    ("CV-S4", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 10.0,
                             "source_file": "assets/sfx/sfx-sc01-sh01-001/ai.mp3",
                             "shot_id": "sc01-sh01"}],
    }, False, "SfxPlan shot_override with shot_id fails schema (additionalProperties)"),

    ("CV-S5", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 10.0,
                             "source_file": "assets/sfx/sfx-sc01-sh01-001/ai.mp3",
                             "item_id": "sfx-sc01-sh01-001"}],
    }, False, "SfxPlan shot_override with item_id fails schema (additionalProperties)"),

    ("CV-S6", "SfxPlan", {
        "schema_id": "SfxPlan", "schema_version": "1.0",
        "timing_format": "episode_absolute",
        "shot_overrides": [{"start_sec": 5.0, "end_sec": 10.0,
                             "source_file": "assets/sfx/sfx-sc01-sh01-001/ai.mp3",
                             "volume_db": 0.0, "duck_db": 0.0, "fade_sec": 0.0,
                             "clip_id": "sfx-sc01-sh01-001:0.0s-5.0s",
                             "clip_path": "assets/sfx/sfx-sc01-sh01-001/cut.wav"}],
        "cut_clips": [{"clip_id": "sfx-sc01-sh01-001:0.0s-5.0s",
                        "item_id": "sfx-sc01-sh01-001",
                        "start_sec": 0.0, "end_sec": 5.0,
                        "path": "assets/sfx/sfx-sc01-sh01-001/cut.wav"}],
        "cut_assign": {"sfx-sc01-sh01-001": "sfx-sc01-sh01-001:0.0s-5.0s"},
    }, True, "SfxPlan with shot_overrides, cut_clips and cut_assign passes schema"),

    # ── MediaPlan (draft 2020-12) ───────────────────────────────────────────────
    ("CV-P1", "MediaPlan", {
        "schema_id": "MediaPlan", "schema_version": "1.0",
        "shot_overrides": [{"type": "image",
                             "url": "https://example.com/bg-reactor.jpg",
                             "path": "assets/media/bg-reactor-control-room-night.jpg",
                             "clip_id": "bg-reactor-control-room-night",
                             "hold_sec": 28.0, "animation_type": "none"}],
    }, True, "valid MediaPlan with shot_overrides passes schema"),

    ("CV-P2", "MediaPlan", {
        "schema_id": "MediaPlan", "schema_version": "1.0",
        "media_segments": [{"type": "image", "hold_sec": 28.0}],
    }, False, "MediaPlan with old media_segments field fails schema (additionalProperties)"),

    ("CV-P5", "MediaPlan", {
        "schema_id": "MediaPlan", "schema_version": "1.0",
        "shot_overrides": [{"url": "https://example.com/bg-reactor.jpg",
                             "hold_sec": 28.0}],
    }, False, "MediaPlan shot_override missing type fails schema (required)"),

    ("CV-P6", "MediaPlan", {
        "schema_id": "MediaPlan", "schema_version": "1.0",
        "slug": "test-proj",
        "shot_overrides": [],
    }, False, "MediaPlan with old top-level slug field fails schema (additionalProperties)"),
]

# Schemas that use JSON Schema draft 2020-12 (all others use draft 7).
_DRAFT_2020_SCHEMAS = {"MediaPlan"}


def run_self_tests(schemas_dir: Path) -> tuple[list[str], int]:
    """Run all behavioural self-test cases.  Returns (failures, total)."""
    failures: list[str] = []

    for case_id, schema_name, doc, expect_valid, description in _SELF_TEST_CASES:
        schema_path = schemas_dir / f"{schema_name}.v1.json"
        try:
            schema = json.loads(schema_path.read_bytes())
        except FileNotFoundError:
            msg = f"{case_id}: SCHEMA_NOT_FOUND {schema_path}"
            print(f"FAIL   {msg}")
            failures.append(msg)
            continue

        if schema_name in _DRAFT_2020_SCHEMAS:
            validator_cls = jsonschema.Draft202012Validator
        else:
            validator_cls = jsonschema.Draft7Validator

        is_valid = validator_cls(schema).is_valid(doc)

        if is_valid == expect_valid:
            print(f"PASS   {case_id}: {description}")
        else:
            outcome = "valid" if is_valid else "invalid"
            expected = "valid" if expect_valid else "invalid"
            msg = f"{case_id}: expected {expected}, got {outcome} — {description}"
            print(f"FAIL   {msg}")
            failures.append(msg)

    return failures, len(_SELF_TEST_CASES)


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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="(Mode 3) Run behavioural self-tests against MusicPlan, SfxPlan, "
             "and MediaPlan schemas using hardcoded known-good/known-bad samples.",
    )
    args = parser.parse_args()

    schemas_dir: Path = args.schemas_dir

    # ── Mode 3: behavioural self-test ─────────────────────────────────────────
    if args.self_test:
        failures, total = run_self_tests(schemas_dir)
        if failures:
            print(f"RESULT: FAIL ({len(failures)} failure(s) across {total} cases)")
            sys.exit(1)
        else:
            print(f"RESULT: PASS ({total}/{total} cases)")
            sys.exit(0)

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
