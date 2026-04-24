#!/usr/bin/env python3
# =============================================================================
# simple_narration_setup.py — Generate all pre-Stage-9 contracts deterministically
# =============================================================================
#
# Part of the simple_narration pipeline format.
# Reads story.md + voice_config.json → writes all 7 contract files.
# No LLM calls. No network calls. Pure file I/O.
#
# Generates:
#   <ep_dir>/meta.json
#   <ep_dir>/pipeline_vars.sh
#   projects/<slug>/VoiceCast.json
#   <ep_dir>/Script.json
#   <ep_dir>/ShotList.json
#   <ep_dir>/AssetManifest.shared.json
#   <ep_dir>/VOPlan.<locale>.json    (one per locale)
#
# Usage:
#   python3 simple_narration_setup.py <ep_dir> \
#     --story <path> --voice <path> --image <path> \
#     --title "Title" --slug <slug> --locale en \
#     --seed 12345 --profile preview_local \
#     [--skip-sections "Twist,Sources"] [--no-default-skips] [--alt N]
#
# Requirements: stdlib only
# =============================================================================

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

NOTO_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

DEFAULT_SKIP_SECTIONS = {
    "sources", "references", "bibliography", "notes", "footnotes",
    "supporting stories", "further reading", "appendix", "credits",
    "twist",
}

MIN_PARAGRAPH_CHARS = 10


# ── Story parsing ─────────────────────────────────────────────────────────────

def detect_story_format(path: Path) -> str:
    """Return 'txt' if file uses ## title / - delimiter format, else 'md'.

    Heuristic: if the first non-empty line starts with '##', treat as txt format.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            return "txt" if stripped.startswith("##") else "md"
    return "md"


def parse_story_txt(
    path: Path,
    title_override: str | None = None,
) -> tuple[str, list[dict], list[str]]:
    """Parse story.txt with '## Title / - delimiter' format.

    Format rules:
      - Each '## Heading' line starts a new story block
      - A line containing only '-' → paragraph delimiter within a block
      - ALL ## blocks are included (main story + supporting stories)
      - Each chunk between delimiters becomes one VO paragraph

    Returns:
      title         — first story's title string (used for slug/meta)
      stories       — list of {"title": str, "paragraphs": list[str]}
      skipped_log   — always [] (nothing skipped in this format)
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    stories: list[dict] = []
    current_title: str | None = None
    current_chunk: list[str] = []
    current_paragraphs: list[str] = []

    def _flush_chunk() -> None:
        nonlocal current_chunk
        chunk = "\n".join(current_chunk).strip()
        if len(chunk) >= MIN_PARAGRAPH_CHARS:
            current_paragraphs.append(chunk)
        current_chunk = []

    def _flush_story() -> None:
        nonlocal current_title, current_paragraphs
        _flush_chunk()
        if current_title is not None and current_paragraphs:
            stories.append({"title": current_title, "paragraphs": list(current_paragraphs)})
        current_paragraphs = []

    for line in lines:
        m = re.match(r"^##\s+(.+)", line)
        if m:
            _flush_story()
            current_title = m.group(1).strip()
            current_chunk = []
        elif re.match(r"^###", line):
            pass  # ### = sources/metadata marker — exclude from narration
        elif re.match(r"^\s*-\s*$", line):
            _flush_chunk()
        else:
            current_chunk.append(line)

    _flush_story()

    title = stories[0]["title"] if stories else ""
    if title_override:
        title = title_override
        if stories:
            stories[0]["title"] = title_override

    return title, stories, []


def parse_story_md(
    path: Path,
    extra_skip: set[str] | None = None,
    no_default_skips: bool = False,
    title_override: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """Parse story.md → (title, paragraphs, skipped_log).

    Steps:
      1. Extract title (H1 → first non-empty line → title_override)
      2. Build skip list (default ∪ extra_skip)
      3. Split into sections by ## headings
      4. Filter sections against skip list
      5. Strip markdown from included sections
      6. Split into paragraphs, filter short ones

    Returns:
      title         — story title string
      paragraphs    — list of plain-text paragraph strings (for TTS)
      skipped_log   — human-readable list of skipped sections
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ── Title extraction ─────────────────────────────────────────────────────
    title = ""
    for line in lines:
        m = re.match(r"^#\s+(.+)", line)
        if m:
            title = m.group(1).strip()
            break
    if not title:
        for line in lines:
            stripped = line.strip()
            if stripped:
                title = stripped
                break
    if title_override:
        title = title_override

    # ── Build skip list ──────────────────────────────────────────────────────
    final_skip: set[str] = set()
    if not no_default_skips:
        final_skip = set(DEFAULT_SKIP_SECTIONS)
    if extra_skip:
        final_skip |= {s.lower().strip() for s in extra_skip}

    # ── Split into sections by ## headings ───────────────────────────────────
    sections: list[tuple[str | None, list[str]]] = []  # (heading, content_lines)
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = re.match(r"^##\s+(.+)", line)
        if m:
            sections.append((current_heading, current_lines))
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))

    # ── Filter sections ──────────────────────────────────────────────────────
    included_content: list[str] = []
    skipped_log: list[str] = []

    for heading, content_lines in sections:
        if heading is None:
            # Root section (before first ##) — always included
            included_content.extend(content_lines)
        elif heading.lower().strip() in final_skip:
            # Skip entire section
            word_count = len(" ".join(content_lines).split())
            skipped_log.append(f'"{heading}" ({word_count} words)')
        else:
            included_content.extend(content_lines)

    # ── Strip markdown ────────────────────────────────────────────────────────
    cleaned = _strip_markdown("\n".join(included_content))

    # ── Split into paragraphs ─────────────────────────────────────────────────
    raw_paragraphs = re.split(r"\n{2,}", cleaned)
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if len(p) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(p)

    return title, paragraphs, skipped_log


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting, return plain text."""
    # Fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`{3,}[\s\S]*?`{3,}", "", text)
    # ATX headings (### text) — remove entirely
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    # Setext headings (=== / ---) — remove underline only
    text = re.sub(r"^[=\-]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Unordered lists
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Bold/italic (order matters: *** before ** before *)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Images (remove entirely)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Links (keep text)
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)
    # HTML tags (strip tags, keep inner text)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse 3+ blank lines → 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── Voice config loading ──────────────────────────────────────────────────────

def load_voice_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_voice_params(voice_cfg: dict, locale: str, alt_index: int | None = None) -> dict:
    """Return the effective voice params for a locale (primary or alternative)."""
    narrator = voice_cfg.get("narrator", {})
    locale_block = narrator.get(locale, narrator.get("en", {}))
    if alt_index is not None:
        alts = locale_block.get("alternatives", [])
        if alt_index < len(alts):
            # Merge: base locale block + alternative overrides
            merged = dict(locale_block)
            merged.update(alts[alt_index])
            # Remove 'alternatives' from merged result
            merged.pop("alternatives", None)
            return merged
        else:
            print(f"[WARN] --alt {alt_index} out of range "
                  f"(alternatives has {len(alts)} entries); using primary", file=sys.stderr)
    return {k: v for k, v in locale_block.items() if k != "alternatives"}


# ── Slugify ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert title to ASCII-only lowercase-hyphenated slug safe for filenames.

    Non-ASCII characters (e.g. CJK, Arabic) are stripped via NFKD normalization.
    If the entire title is non-ASCII and the result is empty, a short MD5 hash
    of the original text is used as a deterministic fallback.
    """
    import hashlib
    import unicodedata
    original = text
    # Normalise to NFKD and drop non-ASCII code points (CJK has no ASCII decomp)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)       # strip non-word chars (keep -)
    text = re.sub(r"[\s_]+", "-", text)         # spaces/underscores → dashes
    text = re.sub(r"-{2,}", "-", text)          # collapse multiple dashes
    text = text.strip("-")
    if not text:
        # Title was entirely non-ASCII — use a 6-char hex hash as unique id
        text = "story-" + hashlib.md5(original.encode()).hexdigest()[:6]
    return text


# ── Validate voice + style against azure_tts index ───────────────────────────

def validate_voice_style(
    voice_cfg: dict,
    locale: str,
    alt_index: int | None,
    azure_index_path: Path,
) -> None:
    """Warn if azure_voice or azure_style not found in azure_tts/index.json."""
    if not azure_index_path.exists():
        print("[WARN] azure_tts/index.json not found — skipping voice validation",
              file=sys.stderr)
        return

    with open(azure_index_path, encoding="utf-8") as f:
        index = json.load(f)
    voices = index.get("voices", {})

    params = get_voice_params(voice_cfg, locale, alt_index)
    azure_voice = params.get("azure_voice", "")
    azure_style = params.get("azure_style", "")

    if azure_voice not in voices:
        print(f"[WARN] Voice '{azure_voice}' not found in azure_tts/index.json",
              file=sys.stderr)
        return

    available_styles = set(voices[azure_voice].get("clips", {}).keys())
    if azure_style and azure_style not in available_styles:
        print(f"[ERROR] Style '{azure_style}' not available for voice '{azure_voice}'.",
              file=sys.stderr)
        print(f"        Available styles: {sorted(available_styles)}", file=sys.stderr)
        sys.exit(1)

    print(f"[validate] Voice '{azure_voice}' style '{azure_style or '(default)'}' ✓")


# ── Contract builders ─────────────────────────────────────────────────────────

def build_meta(
    slug: str,
    title: str,
    locale: str,
    seed: int,
    profile: str,
    episode_id: str = "s01e01",
) -> dict:
    return {
        "schema_id":       "EpisodeMeta",
        "story_title":     title,
        "project_slug":    slug,
        "episode_id":      episode_id,
        "episode_number":  episode_id[4:],      # e.g. "s01e01" → "01"
        "series_genre":    "narration",
        "story_format":    "simple_narration",
        "locales":         locale,
        "generation_seed": seed,
        "render_profile":  profile,
        "no_music":        True,
        "purge_cache":     False,
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }


def build_pipeline_vars(
    slug: str,
    title: str,
    locale: str,
    seed: int,
    profile: str,
    episode_id: str = "s01e01",
    episode_number: str = "01",
    subtitles: bool = False,
) -> str:
    """Return pipeline_vars.sh content, matching gen_pipeline_vars.py format."""
    primary = locale.split(",")[0].strip()
    lines = [
        f'export STORY_TITLE="{title}"',
        f'export EPISODE_NUMBER="{episode_number}"',
        f'export EPISODE_ID="{episode_id}"',
        f'export PRIMARY_LOCALE="{primary}"',
        f'export LOCALES="{locale}"',
        f'export PROJECT_SLUG="{slug}"',
        f'export SERIES_GENRE="narration"',
        f'export GENERATION_SEED="{seed}"',
        f'export RENDER_PROFILE="{profile}"',
        f'export STORY_FORMAT="simple_narration"',
        f'export PROJECT_DIR="projects/{slug}"',
        f'export EPISODE_DIR="projects/{slug}/episodes/{episode_id}"',
        f'export VOICE_CAST_FILE="projects/{slug}/VoiceCast.json"',
        f'export NO_MUSIC="1"',
    ]
    if subtitles:
        lines.append('export SIMPLE_NARRATION_SUBTITLES="1"')
    return "\n".join(lines) + "\n"


def build_voice_cast(slug: str, voice_cfg: dict, locales: list[str]) -> dict:
    """Build VoiceCast.json from voice_config.json."""
    narrator = voice_cfg.get("narrator", {})
    character: dict = {
        "character_id": "narrator",
        "role":         "narrator",
        "gender":       narrator.get("gender", "neutral"),
        "personality":  narrator.get("personality", ""),
    }

    for locale in locales:
        params = get_voice_params(voice_cfg, locale)
        character[locale] = {
            "azure_voice":        params.get("azure_voice", ""),
            "available_styles":   [],
            "azure_style":        params.get("azure_style", None),
            "azure_pitch":        params.get("azure_pitch", "0%"),
            "azure_rate":         params.get("azure_rate", "0%"),
            "azure_break_ms":     params.get("azure_break_ms", 600),
            "azure_style_degree": params.get("azure_style_degree", 1.0),
        }

    return {
        "schema_id":      "VoiceCast",
        "schema_version": "1.0.0",
        "project_id":     slug,
        "characters":     [character],
    }


def build_script(
    slug: str,
    title: str,
    stories: list[dict],
    gender: str,
    episode_id: str = "s01e01",
) -> dict:
    """Build Script.json from stories. One scene per story."""
    scenes = []
    for i, story in enumerate(stories, 1):
        actions = [
            {"type": "dialogue", "speaker_id": "narrator", "line": p}
            for p in story["paragraphs"]
        ]
        scenes.append({
            "scene_id":    f"sc{i:02d}",
            "location":    "narration",
            "time_of_day": "unspecified",
            "actions":     actions,
        })

    return {
        "schema_id":      "Script",
        "schema_version": "1.0.0",
        "script_id":      f"{slug}-{episode_id}",
        "project_id":     slug,
        "title":          title,
        "genre":          "narration",
        "cast": [{"character_id": "narrator", "gender": gender, "role": "narrator"}],
        "scenes":         scenes,
    }


def build_shotlist(
    slug: str,
    stories: list[dict],
    episode_id: str = "s01e01",
) -> dict:
    """Build ShotList.json — one shot per story."""
    now = datetime.now(timezone.utc).isoformat()
    shots = []
    for i, story in enumerate(stories, 1):
        scene_id = f"sc{i:02d}"
        vo_ids = [f"vo-{scene_id}-{j:03d}" for j in range(1, len(story["paragraphs"]) + 1)]
        shots.append({
            "shot_id":         f"{scene_id}-sh01",
            "scene_id":        scene_id,
            "duration_sec":    0,
            "characters":      [],
            "background_id":   "bg-provided",
            "camera_framing":  "wide",
            "camera_movement": "slow_push",
            "audio_intent": {
                "sfx_tags":      [],
                "music_mood":    "none",
                "vo_speaker_id": "narrator",
                "vo_item_ids":   vo_ids,
            },
        })

    return {
        "schema_id":          "ShotList",
        "schema_version":     "1.0.0",
        "created_at":         now,
        "script_id":          f"{slug}-{episode_id}",
        "shotlist_id":        f"{slug}-{episode_id}",
        "timing_lock_hash":   "",
        "total_duration_sec": 0,
        "shots":              shots,
    }


def build_asset_manifest_shared(
    slug: str,
    episode_id: str = "s01e01",
) -> dict:
    """Build AssetManifest.shared.json — one background, no character packs."""
    return {
        "schema_id":       "AssetManifest",
        "schema_version":  "1.0.0",
        "manifest_id":     f"{slug}-{episode_id}-shared",
        "project_id":      slug,
        "episode_id":      episode_id,
        "shotlist_ref":    "ShotList.json",
        "locale_scope":    "shared",
        "vo_items":        [],
        "character_packs": [],
        "backgrounds": [
            {
                "asset_id":      "bg-provided",
                "type":          "background",
                "license_type":  "proprietary_cleared",
                "motion_level":  "low",
                "cinematic_role": "hold",
            }
        ],
    }


def build_voplan(
    slug: str,
    locale: str,
    stories: list[dict],
    voice_cfg: dict,
    alt_index: int | None,
    episode_id: str = "s01e01",
) -> dict:
    """Build VOPlan.{locale}.json — fully pre-populated vo_items with tts_prompt.

    Each story gets its own scene_id (sc01, sc02, …).  The story_segments field
    records story titles and first_item_id so render_video.py can overlay story
    badges without needing a MediaPlan.
    """
    params = get_voice_params(voice_cfg, locale, alt_index)
    story_count = len(stories)
    vo_items: list[dict] = []
    story_segments: list[dict] = []

    for story_idx, story in enumerate(stories, 1):
        scene_id = f"sc{story_idx:02d}"
        first_item_id: str | None = None

        for para_idx, para in enumerate(story["paragraphs"], 1):
            item_id = f"vo-{scene_id}-{para_idx:03d}"
            if first_item_id is None:
                first_item_id = item_id

            tts_prompt: dict = {
                "locale":             locale,
                "azure_voice":        params.get("azure_voice", ""),
                "azure_rate":         params.get("azure_rate", "0%"),
                "azure_pitch":        params.get("azure_pitch", "0%"),
                "azure_break_ms":     params.get("azure_break_ms", 600),
                "azure_style_degree": params.get("azure_style_degree", 1.0),
            }
            style = params.get("azure_style")
            if style:
                tts_prompt["azure_style"] = style

            vo_items.append({
                "item_id":      item_id,
                "speaker_id":   "narrator",
                "text":         para,
                "license_type": "commercial_reusable",
                "tts_prompt":   tts_prompt,
            })

        story_segments.append({
            "story_index":   story_idx,
            "story_count":   story_count,
            "title":         story["title"],
            "first_item_id": first_item_id,
        })

    return {
        "schema_id":       "VOPlan",
        "schema_version":  "1.0.0",
        "manifest_id":     f"{slug}-{episode_id}-{locale}-manifest",
        "project_id":      slug,
        "episode_id":      episode_id,
        "locale":          locale,
        "locale_scope":    "merged",
        "shared_ref":      "AssetManifest.shared.json",
        "shotlist_ref":    "ShotList.json",
        "character_packs": [],
        "backgrounds": [{
            "asset_id":       "bg-provided",
            "license_type":   "proprietary_cleared",
            "motion_level":   "low",
            "cinematic_role": "hold",
        }],
        "resolved_assets":  [],   # filled by resolve_assets.py at Stage 9
        "story_segments":   story_segments,
        "vo_items":         vo_items,
    }


# ── File I/O ──────────────────────────────────────────────────────────────────

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  ✓ {path}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all simple_narration contracts deterministically."
    )
    p.add_argument("ep_dir",            help="Episode directory path")
    p.add_argument("--story",           required=True,  help="Path to story.md")
    p.add_argument("--voice",           required=True,  help="Path to voice_config.json")
    p.add_argument("--image",           required=True,  help="Copied background image path in ep_dir/assets/")
    p.add_argument("--title",           default=None,   help="Override story title")
    p.add_argument("--slug",            default=None,   help="Project slug (derived from title if omitted)")
    p.add_argument("--locale",          default="en",   help="Comma-separated BCP-47 locales (default: en)")
    p.add_argument("--seed",            type=int, default=None, help="Generation seed (random if omitted)")
    p.add_argument("--profile",         default="preview_local", help="Render profile (default: preview_local)")
    p.add_argument("--episode",         default="s01e01", help="Episode ID (default: s01e01)")
    p.add_argument("--skip-sections",   default="",     help="Comma-separated section names to skip")
    p.add_argument("--no-default-skips", action="store_true",
                   help="Disable built-in default skip list")
    p.add_argument("--alt",             type=int, default=None,
                   help="Use alternatives[N] instead of primary voice (0-based)")
    p.add_argument("--subtitles",       action="store_true",
                   help="Burn subtitles into output (persisted to pipeline_vars.sh)")
    return p.parse_args()


def main() -> None:
    import random
    args = parse_args()

    ep_dir   = Path(args.ep_dir).resolve()
    story    = Path(args.story).resolve()
    voice    = Path(args.voice).resolve()
    image    = Path(args.image).resolve()
    episode_id = args.episode
    episode_number = episode_id[4:]  # "s01e01" → "01"

    # Validate inputs
    for p, label in [(story, "--story"), (voice, "--voice"), (image, "--image")]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    seed = args.seed if args.seed is not None else random.randint(100_000_000, 999_999_999)

    # ── Build extra skip set ─────────────────────────────────────────────────
    extra_skip: set[str] = set()
    if args.skip_sections:
        extra_skip = {s.strip() for s in args.skip_sections.split(",") if s.strip()}

    print("=" * 60)
    print("  simple_narration_setup.py")
    print(f"  Story  : {story.name}")
    print(f"  Voice  : {voice.name}")
    print(f"  Image  : {image.name}")
    print("=" * 60)
    print()

    # ── Load voice config ────────────────────────────────────────────────────
    print("[1] Loading voice config…")
    voice_cfg = load_voice_config(voice)
    narrator  = voice_cfg.get("narrator", {})
    gender    = narrator.get("gender", "neutral")
    # Merge skip_sections from voice_config
    vc_skip = {s.lower().strip() for s in voice_cfg.get("skip_sections", [])}
    extra_skip |= vc_skip
    print(f"  Gender        : {gender}")
    print(f"  Extra skip    : {sorted(extra_skip) or '(none)'}")
    print()

    # ── Parse story ──────────────────────────────────────────────────────────
    fmt = detect_story_format(story)
    if fmt == "txt":
        print("[2] Parsing story (## / - delimiter format)…")
        title, stories, skipped_log = parse_story_txt(
            story,
            title_override=args.title,
        )
        # Flatten paragraphs for backward-compat validators
        paragraphs = [p for s in stories for p in s["paragraphs"]]
    else:
        print("[2] Parsing story (markdown format)…")
        title, paragraphs, skipped_log = parse_story_md(
            story,
            extra_skip=extra_skip,
            no_default_skips=args.no_default_skips,
            title_override=args.title,
        )
        # Wrap single-story markdown as a stories list
        stories = [{"title": title, "paragraphs": paragraphs}]

    slug = args.slug or slugify(title)
    total_vo = sum(len(s["paragraphs"]) for s in stories)
    print(f"  Title         : {title}")
    print(f"  Slug          : {slug}")
    print(f"  Stories       : {len(stories)}")
    print(f"  Total VO items: {total_vo}")
    if skipped_log:
        print(f"  Skipped       : {', '.join(skipped_log)}")
    print()

    if not paragraphs:
        print("[ERROR] No paragraphs extracted from story — nothing to narrate.",
              file=sys.stderr)
        sys.exit(1)

    # ── Validate voice+style ─────────────────────────────────────────────────
    print("[3] Validating voice configuration…")
    locales = [l.strip() for l in args.locale.split(",") if l.strip()]
    azure_index = Path(__file__).parent.parent.parent / "projects" / "resources" / "azure_tts" / "index.json"
    for locale in locales:
        validate_voice_style(voice_cfg, locale, args.alt, azure_index)
    print()

    # ── Write contracts ──────────────────────────────────────────────────────
    print("[4] Writing contracts…")
    locale_str = args.locale

    # 4a. meta.json
    meta = build_meta(slug, title, locale_str, seed, args.profile, episode_id)
    write_json(ep_dir / "meta.json", meta)

    # 4b. pipeline_vars.sh
    vars_content = build_pipeline_vars(
        slug, title, locale_str, seed, args.profile, episode_id, episode_number,
        subtitles=args.subtitles,
    )
    write_text(ep_dir / "pipeline_vars.sh", vars_content)

    # 4c. VoiceCast.json (project level — sibling of episodes/)
    project_dir = ep_dir.parent.parent   # ep_dir = projects/<slug>/episodes/<ep_id>
    voice_cast = build_voice_cast(slug, voice_cfg, locales)
    write_json(project_dir / "VoiceCast.json", voice_cast)

    # 4d. Script.json
    script = build_script(slug, title, stories, gender, episode_id)
    write_json(ep_dir / "Script.json", script)

    # 4e. ShotList.json
    shotlist = build_shotlist(slug, stories, episode_id)
    write_json(ep_dir / "ShotList.json", shotlist)

    # 4f. AssetManifest.shared.json
    manifest = build_asset_manifest_shared(slug, episode_id)
    write_json(ep_dir / "AssetManifest.shared.json", manifest)

    # 4g. VOPlan.{locale}.json — one per locale
    for locale in locales:
        voplan = build_voplan(slug, locale, stories, voice_cfg, args.alt, episode_id)
        write_json(ep_dir / f"VOPlan.{locale}.json", voplan)

    # 4h. story.txt — required by run.sh
    write_text(ep_dir / "story.txt", story.read_text(encoding="utf-8"))

    print()
    print(f"[5] Summary")
    print(f"  Seed          : {seed}")
    print(f"  Episode dir   : {ep_dir}")
    print(f"  Stories       : {len(stories)}")
    print(f"  Total VO items: {total_vo}")
    print(f"  Locales       : {locale_str}")
    _last_story = stories[-1]
    _last_scene = f"sc{len(stories):02d}"
    _last_para  = len(_last_story["paragraphs"])
    print(f"  VO range      : vo-sc01-001 … vo-{_last_scene}-{_last_para:03d}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
