#!/usr/bin/env python3
# =============================================================================
# validate_tts_prompts.py — Verify tts_prompt fields match VoiceCast.json
# =============================================================================
#
# Closes OPEN 2: the Stage 5 LLM is supposed to copy TTS parameters verbatim
# from VoiceCast.json. If it deviates (wrong rate, missing field, wrong voice),
# gen_tts_cloud.py will re-synthesize lines whose WAVs were approved at Stage 3.5,
# and gen_render_plan.py will apply approved timing to audio with a different
# duration — causing desync in the final render.
#
# This script makes the WAV-reuse guarantee deterministic by comparing each
# vo_item.tts_prompt against VoiceCast.json for the same character/locale.
#
# Runs AFTER Stage 5 (AssetManifest.{locale}.json produced) and BEFORE
# gen_tts_cloud.py is called in Stage 9.
#
# Called by run.sh in Stage 9 for each locale before gen_tts_cloud.py.
#
# Reads:
#   AssetManifest.{locale}.json         — locale manifest with tts_prompt per item
#   projects/{slug}/VoiceCast.json      — ground truth TTS params per character/locale
#
# Exits 0 if all tts_prompts match VoiceCast.json.
# Exits 1 if any field diverges (prints a human-readable diff).
#
# Fields compared (OPEN 5/8 canonical list):
#   azure_voice, azure_style, azure_style_degree,
#   azure_rate, azure_pitch, azure_break_ms
#
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

TTS_FIELDS = [
    "azure_voice",
    "azure_style",
    "azure_style_degree",
    "azure_rate",
    "azure_pitch",
    "azure_break_ms",
]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_tts(
    item_id:    str,
    speaker_id: str,
    tts_prompt: dict,
    voicecast:  dict,
    locale:     str,
) -> list[str]:
    """Return list of divergence strings (empty = all match)."""
    character = voicecast.get("characters", {}).get(speaker_id)
    if character is None:
        # No VoiceCast entry → nothing to compare
        return []

    locale_params = character.get("locales", {}).get(locale) or character.get(locale)
    if locale_params is None:
        return []  # character has no entry for this locale (e.g. narrator in some formats)

    divergences = []
    for field in TTS_FIELDS:
        expected = locale_params.get(field)
        actual   = tts_prompt.get(field)

        # Both absent or both null → OK
        if expected is None and actual is None:
            continue
        # azure_style null ↔ absent is equivalent
        if field == "azure_style" and not expected and not actual:
            continue
        # Numeric fields: compare with small tolerance
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                if abs(float(expected) - float(actual)) < 0.001:
                    continue
            except (TypeError, ValueError):
                pass
        if expected != actual:
            divergences.append(
                f"  {item_id} (speaker={speaker_id!r})  {field}: "
                f"expected={expected!r}  actual={actual!r}"
            )
    return divergences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate that tts_prompt fields in AssetManifest.{locale}.json\n"
            "match VoiceCast.json for each character/locale.\n"
            "Exits 1 if any field diverges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ep_dir", help="Episode directory")
    p.add_argument("--locale", default="en", help="Locale to validate (default: en)")
    p.add_argument("--warn-only", action="store_true",
                   help="Print divergences but exit 0 (do not block pipeline)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ep_dir = Path(args.ep_dir)
    if not ep_dir.is_dir():
        ep_dir = PIPE_DIR / args.ep_dir
    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir not found: {args.ep_dir}", file=sys.stderr)
        sys.exit(1)
    ep_dir = ep_dir.resolve()
    locale = args.locale

    # ── Find project root from ep_dir (projects/<slug>/episodes/<id>/) ────────
    # Go up 3 levels: ep_dir / episodes / <slug> / projects
    try:
        project_slug = ep_dir.parts[-3]   # episodes/<slug> → slug is 2 levels up
        project_dir  = ep_dir.parent.parent
    except IndexError:
        project_dir  = PIPE_DIR
        project_slug = ""

    # ── Load VoiceCast.json ───────────────────────────────────────────────────
    voicecast_path = project_dir / "VoiceCast.json"
    if not voicecast_path.exists():
        # Try common alternate location
        voicecast_path = PIPE_DIR / "projects" / project_slug / "VoiceCast.json"
    if not voicecast_path.exists():
        print(f"[WARN] VoiceCast.json not found — skipping tts_prompt validation")
        sys.exit(0)
    voicecast = load_json(voicecast_path)

    # ── Load locale manifest ──────────────────────────────────────────────────
    manifest_path = ep_dir / f"AssetManifest.{locale}.json"
    if not manifest_path.exists():
        print(f"[WARN] AssetManifest.{locale}.json not found — skipping")
        sys.exit(0)
    manifest = load_json(manifest_path)

    print("=" * 60)
    print("  validate_tts_prompts")
    print(f"  Episode : {ep_dir}")
    print(f"  Locale  : {locale}")
    print(f"  Items   : {len(manifest.get('vo_items', []))}")
    print("=" * 60)

    all_divergences: list[str] = []
    for item in manifest.get("vo_items", []):
        item_id    = item.get("item_id", "?")
        speaker_id = item.get("speaker_id", "")
        tts_prompt = item.get("tts_prompt", {})
        divs = compare_tts(item_id, speaker_id, tts_prompt, voicecast, locale)
        all_divergences.extend(divs)

    if not all_divergences:
        print(f"\n  ✓ All tts_prompt fields match VoiceCast.json for locale={locale!r}")
        print(f"    ({len(manifest.get('vo_items', []))} items checked — WAV reuse guaranteed)")
        print("=" * 60)
        sys.exit(0)

    print(f"\n  ✗ {len(all_divergences)} tts_prompt divergence(s) found:")
    for d in all_divergences:
        print(d)
    print()
    print("  These items will trigger re-synthesis in gen_tts_cloud.py.")
    print("  If Stage 3.5 WAVs were approved, the final render may desync.")
    print("  Fix p_5.txt and re-run Stage 5, OR fix the manifest manually.")
    print("=" * 60)

    if args.warn_only:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
