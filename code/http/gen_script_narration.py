#!/usr/bin/env python3
# =============================================================================
# gen_script_narration.py — Stage 3: Build Script.json for narration formats
# =============================================================================
#
# Replaces the LLM-based Stage 3 for the following story formats:
#   continuous_narration, illustrated_narration, documentary
#
# Reads story.txt from the episode directory, splits it into scenes and
# sentences deterministically, and produces a Script.v1.json document with a
# single "narrator" cast member whose dialogue lines are the raw story
# sentences, one action per sentence.
#
# Usage:
#   python gen_script_narration.py <ep_dir>
#
#   ep_dir — path to the episode directory (contains pipeline_vars.sh and
#             story.txt)
#
# Output:
#   ep_dir/Script.json  — validated against contracts/schemas/Script.v1.json
#
# Requirements: stdlib + jsonschema (for schema validation)
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

STAGE_LABEL_START = "[3/9] Stage 3 — gen_script_narration.py"
STAGE_LABEL_DONE  = "[3/9] Stage 3 — Script.json written"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_pipeline_vars(ep_dir: Path) -> dict:
    """Parse pipeline_vars.sh and return dict of exported variables."""
    vars_file = ep_dir / "pipeline_vars.sh"
    if not vars_file.exists():
        raise FileNotFoundError(f"pipeline_vars.sh not found: {vars_file}")
    result = {}
    for line in vars_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Match: export KEY="value" or export KEY=value or KEY="value"
        m = re.match(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)=["\']?(.*?)["\']?\s*$', line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


# ── Scene / sentence splitting ────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """
    Split text into sentences.

    Handles CJK (。！？…) and Latin (. ! ?) terminal punctuation.
    Protects common abbreviations (a.m., p.m., Mr., Dr., vs., e.g., i.e.,
    single initials, ordinals like 1st/2nd) from being treated as sentence
    boundaries.
    Returns a list of non-empty stripped sentence strings.
    """
    # Step 1: protect known abbreviations by replacing their periods with a
    # placeholder that won't be touched by the sentence splitter.
    _PLACEHOLDER = "\x00"

    # Titles and common abbreviations (case-insensitive)
    _ABBREVS = re.compile(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|dept|est|govt|corp'
        r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        r'|e\.g|i\.e|i\.e|cf|ibid|et al|op|cit)\.',
        re.IGNORECASE,
    )
    # a.m. / p.m. style — letter-dot-letter-dot
    _AM_PM = re.compile(r'\b([a-zA-Z])\.([a-zA-Z])\.')
    # Single uppercase initial followed by a period and a space then a lowercase
    # word (likely an initial, not end-of-sentence): "J. edgar" or "U.S. army"
    _INITIAL = re.compile(r'\b([A-Z])\.(?=\s+[a-z])')
    # Ordinals: 1st. 2nd. 3rd. 4th. etc.
    _ORDINAL = re.compile(r'\b(\d+(?:st|nd|rd|th))\.')

    protected = text
    protected = _ABBREVS.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), protected)
    protected = _AM_PM.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), protected)
    protected = _INITIAL.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), protected)
    protected = _ORDINAL.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), protected)

    # Step 2: split on genuine sentence-ending punctuation
    pattern = r'(?<=[。！？…\.\!\?])\s+'
    parts = [s.strip() for s in re.split(pattern, protected) if s.strip()]

    # Step 3: restore placeholders
    sentences = [p.replace(_PLACEHOLDER, '.') for p in parts]
    return sentences


def split_scenes(text: str) -> list:
    """
    Split story text into a list of scenes, where each scene is a list of
    sentence strings.

    Priority order:
      1. Paragraph breaks (2+ newlines) — most common in prose
      2. Section dividers (---, ***, ===) — used by some story templates
      3. Fallback: group sentences into ~3 equal scenes

    Returns list[list[str]].
    """
    # Priority 1: paragraph breaks
    paragraphs = re.split(r'\n{2,}', text.strip())
    if len(paragraphs) >= 2:
        return [split_sentences(p) for p in paragraphs if p.strip()]

    # Priority 2: section dividers
    sections = re.split(r'\n[-*=]{3,}\n', text)
    if len(sections) >= 2:
        return [split_sentences(s) for s in sections if s.strip()]

    # Fallback: group all sentences into ~3 equal scenes
    sents = split_sentences(text)
    group = max(1, (len(sents) + 2) // 3)
    return [sents[i:i + group] for i in range(0, len(sents), group)]


# ── Script builder ────────────────────────────────────────────────────────────

def build_script(
    project_slug: str,
    episode_id:   str,
    story_title:  str,
    series_genre: str,
    story_text:   str,
) -> dict:
    """
    Build a Script.v1.0.0 document from raw story text.

    The narrator is the only cast member. Each paragraph/section becomes a
    scene; each sentence within it becomes a 'dialogue' action.
    """
    scene_groups = split_scenes(story_text)

    scenes = []
    total_sentences = 0

    for idx, sentences in enumerate(scene_groups):
        actions = []
        for i, sentence in enumerate(sentences):
            actions.append({
                "type":        "dialogue",
                "speaker_id":  "narrator",
                "line":        sentence,
                "sentence_id": f"sent_{i + 1:03d}",
            })
        total_sentences += len(sentences)

        scenes.append({
            "scene_id":    f"sc{idx + 1:02d}",
            "location":    "unspecified",
            "time_of_day": "unspecified",
            "actions":     actions,
        })

    script = {
        "schema_id":      "Script",
        "schema_version": "1.0.0",
        "script_id":      f"{project_slug}-{episode_id}",
        "project_id":     project_slug,
        "title":          story_title or "Untitled",
        "genre":          series_genre or "narration",
        "cast": [
            {
                "character_id": "narrator",
                "gender":       "neutral",
                "role":         "off-screen narrator",
            }
        ],
        "scenes": scenes,
    }

    return script, total_sentences


# ── Schema validation ─────────────────────────────────────────────────────────

def find_schema_path(ep_dir: Path) -> Path | None:
    """Search upward from ep_dir to find contracts/schemas/Script.v1.json."""
    schema_rel = Path("contracts") / "schemas" / "Script.v1.json"
    candidates = [ep_dir]
    parent = ep_dir
    for _ in range(6):
        parent = parent.parent
        candidates.append(parent)
    for root in candidates:
        candidate = root / schema_rel
        if candidate.exists():
            return candidate
    return None


def validate_script(script: dict, ep_dir: Path) -> None:
    """Validate script against Script.v1.json schema using jsonschema."""
    if not _HAS_JSONSCHEMA:
        print("  [WARN] jsonschema not installed — skipping schema validation.")
        return

    schema_path = find_schema_path(ep_dir)
    if schema_path is None:
        print("  [WARN] Script.v1.json schema not found — skipping validation.")
        return

    schema = load_json(schema_path)
    try:
        jsonschema.validate(instance=script, schema=schema)
        print(f"  Schema validation : OK  ({schema_path.name})")
    except jsonschema.ValidationError as exc:
        print(f"[ERROR] Script.json failed schema validation:", file=sys.stderr)
        print(f"        {exc.message}", file=sys.stderr)
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: Build Script.json from story.txt for narration formats "
                    "(continuous_narration, illustrated_narration, documentary).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ep_dir", metavar="EP_DIR",
                   help="Episode directory (contains pipeline_vars.sh and story.txt).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ep_dir = Path(args.ep_dir).resolve()

    print(f"▶ {STAGE_LABEL_START}")

    # Parse pipeline_vars.sh
    try:
        pipeline_vars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_slug = pipeline_vars.get("PROJECT_SLUG", "").strip()
    episode_id   = pipeline_vars.get("EPISODE_ID", "").strip()
    story_title  = pipeline_vars.get("STORY_TITLE", "").strip()
    series_genre = pipeline_vars.get("SERIES_GENRE", "").strip()

    if not project_slug:
        print("[ERROR] PROJECT_SLUG not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)
    if not episode_id:
        print("[ERROR] EPISODE_ID not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    # Load story.txt
    story_path = ep_dir / "story.txt"
    if not story_path.exists():
        print(f"[ERROR] story.txt not found: {story_path}", file=sys.stderr)
        sys.exit(1)
    story_text = story_path.read_text(encoding="utf-8")

    print(f"  Project      : {project_slug}")
    print(f"  Episode      : {episode_id}")
    print(f"  Title        : {story_title or '(unset — will use Untitled)'}")
    print(f"  Genre        : {series_genre or '(unset — will use narration)'}")
    print(f"  Story file   : {story_path}  ({len(story_text)} chars)")

    # Build Script.json
    script, total_sentences = build_script(
        project_slug=project_slug,
        episode_id=episode_id,
        story_title=story_title,
        series_genre=series_genre,
        story_text=story_text,
    )

    scene_count = len(script["scenes"])
    print()
    print(f"  Total sentences : {total_sentences}")
    print(f"  Scene count     : {scene_count}")

    # Validate against schema
    validate_script(script, ep_dir)

    # Write output
    out_path = ep_dir / "Script.json"
    save_json(script, out_path)

    print()
    print(f"  [OK] {out_path}")
    print()
    print(f"✓ {STAGE_LABEL_DONE}")


if __name__ == "__main__":
    main()
