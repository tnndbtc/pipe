#!/usr/bin/env python3
# =============================================================================
# ssml_preprocess.py  —  SSML Narration Preprocessor
#
# Parses raw SSML narration (story.txt) and episode metadata (meta.json) to
# produce pipeline-compatible artifacts, bypassing the LLM-driven stages that
# are unnecessary when the user has already authored structured SSML content.
#
# Outputs:
#   1. $EP_DIR/Script.json          Script.v1.json-valid script
#   2. projects/{slug}/VoiceCast.json   append-only voice cast
#   3. $EP_DIR/ssml_inner.xml       inner SSML content (immutable after creation)
#   4. $EP_DIR/pipeline_vars.sh     shell variables for downstream stages
#   5. $EP_DIR/NarrationText.txt    plain prose (all XML tags stripped)
#
# Usage:
#   python3 code/http/ssml_preprocess.py "$EP_DIR"
#
# Prerequisites:
#   $EP_DIR/story.txt   raw SSML pasted by the user
#   $EP_DIR/meta.json   episode metadata
# =============================================================================

from __future__ import annotations

import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPE_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="  %(levelname)s  %(message)s",
)
log = logging.getLogger("ssml_preprocess")

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------
NS_SYNTH = "http://www.w3.org/2001/10/synthesis"
NS_MSTTS = "http://www.w3.org/2001/mstts"

NS_MAP = {
    "": NS_SYNTH,
    "mstts": NS_MSTTS,
}

# Register namespaces so ET.tostring preserves prefixes.
for prefix, uri in NS_MAP.items():
    ET.register_namespace(prefix, uri)


# ---------------------------------------------------------------------------
# Locale mapping
# ---------------------------------------------------------------------------
LOCALE_MAP: dict[str, str] = {
    "zh-CN":   "zh-Hans",
    "zh-TW":   "zh-Hant",
    "en-US":   "en",
    "en-GB":   "en",
    "en-AU":   "en",
    "en-IN":   "en",
    "ja-JP":   "ja",
    "ko-KR":   "ko",
    "fr-FR":   "fr",
    "de-DE":   "de",
    "es-ES":   "es",
    "es-MX":   "es",
    "pt-BR":   "pt",
    "it-IT":   "it",
}


def map_locale(xml_lang: str) -> str:
    """Map an xml:lang value (e.g. 'zh-CN') to pipeline locale (e.g. 'zh-Hans')."""
    if xml_lang in LOCALE_MAP:
        return LOCALE_MAP[xml_lang]
    # Try prefix match (e.g. 'en-...' -> 'en')
    prefix = xml_lang.split("-")[0]
    return prefix


# ---------------------------------------------------------------------------
# Rate normalisation
# ---------------------------------------------------------------------------
def normalise_rate(raw: str) -> str:
    """Normalise SSML prosody rate to a percentage string.

    - Numeric rates (e.g. '0.86') -> round((val - 1.0) * 100) -> '-14%'
    - Percentage rates (e.g. '-14%') -> passthrough
    - Named rates (e.g. 'slow') -> passthrough
    """
    raw = raw.strip()
    if not raw:
        return "0%"
    if raw.endswith("%"):
        return raw
    try:
        val = float(raw)
        pct = round((val - 1.0) * 100)
        return f"{pct:+d}%".replace("+0%", "0%")
    except ValueError:
        return raw


# ---------------------------------------------------------------------------
# SSML parsing helpers
# ---------------------------------------------------------------------------
def _tag(ns: str, local: str) -> str:
    """Build a Clark-notation tag name."""
    return f"{{{ns}}}{local}" if ns else local


def _parse_break_ms(elem: ET.Element) -> int:
    """Extract break duration in ms from a <break time='Nms'/> element."""
    t = elem.get("time", "")
    m = re.match(r"(\d+)\s*ms", t, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"([\d.]+)\s*s", t, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1000)
    return 0


def _iter_text_and_breaks(root: ET.Element):
    """Depth-first walk yielding ('text', str) or ('break', ms) tuples.

    We walk all elements recursively, collecting text/tail and <break> nodes.
    Outer wrapper elements (<speak>, <voice>, <mstts:express-as>, <prosody>)
    are traversed transparently; only their text content is captured.
    """
    # Yield root text (if any)
    if root.text:
        yield ("text", root.text)
    for child in root:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "break":
            ms = _parse_break_ms(child)
            yield ("break", ms)
        else:
            # Recurse into nested elements (voice, express-as, prosody, etc.)
            yield from _iter_text_and_breaks(child)
        # Tail text after this child element
        if child.tail:
            yield ("text", child.tail)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# Regex patterns for sentence-ending punctuation by language family.
_ZH_SENT_RE = re.compile(r"(?<=[。！？…])")
_EN_SENT_RE = re.compile(r'(?<=[.!?…])\s+')


def _split_sentences(text: str, locale: str) -> list[str]:
    """Split text into sentences based on locale-aware punctuation rules."""
    if locale.startswith("zh"):
        parts = _ZH_SENT_RE.split(text)
    else:
        parts = _EN_SENT_RE.split(text)
    # Strip and filter empties
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Core SSML extraction
# ---------------------------------------------------------------------------
def parse_ssml(ssml_text: str) -> dict:
    """Parse SSML content and return structured data.

    Returns dict with keys:
        locale          pipeline locale string
        voice_name      Azure voice name
        style           express-as style (or '')
        style_degree    express-as styledegree (or '')
        rate            prosody rate (raw)
        pitch           prosody pitch (or '')
        fragments       list of ('text', str) | ('break', int) tuples
        inner_xml       inner SSML content string
    """
    # Parse the SSML document
    root = ET.fromstring(ssml_text)

    # Detect locale from <speak xml:lang="...">
    xml_lang = root.get(f"{{{NS_SYNTH}}}lang") or root.get("xml:lang") or root.get("lang", "")
    # ET may have parsed the xml:lang as {http://www.w3.org/XML/1998/namespace}lang
    if not xml_lang:
        xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang", "en-US")
    locale = map_locale(xml_lang)

    # Find <voice name="...">
    _ve = root.find(f".//{_tag(NS_SYNTH, 'voice')}")
    voice_elem = _ve if _ve is not None else root.find(".//voice")
    voice_name = voice_elem.get("name", "") if voice_elem is not None else ""

    # Find <mstts:express-as>
    express_elem = (
        root.find(f".//{_tag(NS_MSTTS, 'express-as')}")
    )
    style = ""
    style_degree = ""
    if express_elem is not None:
        style = express_elem.get("style", "")
        style_degree = express_elem.get("styledegree", "")

    # Find <prosody>
    _pe = root.find(f".//{_tag(NS_SYNTH, 'prosody')}")
    prosody_elem = _pe if _pe is not None else root.find(".//prosody")
    rate_raw = ""
    pitch = ""
    if prosody_elem is not None:
        rate_raw = prosody_elem.get("rate", "")
        pitch = prosody_elem.get("pitch", "")

    # Find the innermost content container (prosody > express-as > voice > speak)
    inner_root = (prosody_elem if prosody_elem is not None else
                  express_elem if express_elem is not None else
                  voice_elem if voice_elem is not None else root)
    if inner_root is None:
        inner_root = root

    # Collect text + break fragments from the inner content
    fragments = list(_iter_text_and_breaks(inner_root))

    # Build inner XML string: serialise inner_root children + text
    inner_parts: list[str] = []
    if inner_root.text:
        inner_parts.append(inner_root.text)
    for child in inner_root:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "break":
            time_attr = child.get("time", "0ms")
            inner_parts.append(f'<break time="{time_attr}"/>')
        else:
            # For any other child element, serialise it
            inner_parts.append(
                ET.tostring(child, encoding="unicode", short_empty_elements=True)
            )
        if child.tail:
            inner_parts.append(child.tail)
    inner_xml = "".join(inner_parts).strip()

    return {
        "locale": locale,
        "voice_name": voice_name,
        "style": style,
        "style_degree": style_degree,
        "rate": rate_raw,
        "pitch": pitch,
        "fragments": fragments,
        "inner_xml": inner_xml,
    }


# ---------------------------------------------------------------------------
# Output generators
# ---------------------------------------------------------------------------
def build_script(meta: dict, fragments: list, locale: str) -> dict:
    """Build Script.v1.json from parsed SSML fragments."""
    slug = meta["project_slug"]
    ep_id = meta["episode_id"]
    title = meta.get("story_title", slug)
    genre = meta.get("series_genre", "")

    # Phase 1: aggregate text fragments into sentences with trailing pauses.
    # Walk fragments: accumulate text, and when we see a break, attach its
    # duration as a trailing pause on whatever text has been accumulated so far.
    raw_chunks: list[dict] = []  # {"text": str, "pause_ms": int}
    current_text = ""

    for kind, value in fragments:
        if kind == "text":
            current_text += value
        elif kind == "break":
            if current_text.strip():
                raw_chunks.append({"text": current_text.strip(), "pause_ms": value})
                current_text = ""
            elif raw_chunks:
                # Break with no preceding new text — add to last chunk's pause
                raw_chunks[-1]["pause_ms"] += value
            else:
                # Leading break before any text — skip
                pass

    # Flush remaining text
    if current_text.strip():
        raw_chunks.append({"text": current_text.strip(), "pause_ms": 800})

    # Phase 2: split text chunks into individual sentences.
    sentences: list[dict] = []  # {"text": str, "pause_ms": int}
    for chunk in raw_chunks:
        sents = _split_sentences(chunk["text"], locale)
        for i, sent_text in enumerate(sents):
            if i < len(sents) - 1:
                # Non-terminal sentence within a chunk: default inter-sentence pause
                sentences.append({"text": sent_text, "pause_ms": 800})
            else:
                # Last sentence in this chunk inherits the chunk's trailing pause
                sentences.append({"text": sent_text, "pause_ms": chunk["pause_ms"]})

    if not sentences:
        log.warning("No sentences found in SSML content")

    # Phase 3: group sentences into scenes based on pause boundaries.
    # Pauses >= 2000ms define scene boundaries.
    SCENE_BREAK_MS = 2000
    scenes: list[dict] = []
    current_actions: list[dict] = []
    sent_counter = 0

    for sent in sentences:
        sent_counter += 1
        sent_id = f"sent_{sent_counter:03d}"
        action = {
            "type": "dialogue",
            "speaker_id": "narrator",
            "line": sent["text"],
            "pause_after_ms": sent["pause_ms"],
            "sentence_id": sent_id,
        }
        current_actions.append(action)

        # If this sentence has a long pause, start a new scene after it
        if sent["pause_ms"] >= SCENE_BREAK_MS and sent is not sentences[-1]:
            scene_num = len(scenes) + 1
            scenes.append({
                "scene_id": f"{ep_id}_sc{scene_num:02d}",
                "location": "unspecified",
                "time_of_day": "unspecified",
                "actions": current_actions,
            })
            current_actions = []

    # Flush remaining actions into the last scene
    if current_actions:
        scene_num = len(scenes) + 1
        scenes.append({
            "scene_id": f"{ep_id}_sc{scene_num:02d}",
            "location": "unspecified",
            "time_of_day": "unspecified",
            "actions": current_actions,
        })

    script = {
        "schema_id": "Script",
        "schema_version": "1.0.0",
        "script_id": f"{slug}-{ep_id}",
        "project_id": slug,
        "title": f"{title} \u2014 {ep_id}",
        "genre": genre,
        "cast": [
            {
                "character_id": "narrator",
                "gender": "neutral",
                "role": "off-screen narrator",
            }
        ],
        "scenes": scenes,
    }
    return script


def build_voicecast_entry(parsed: dict) -> dict:
    """Build a narrator VoiceCast entry from parsed SSML voice data."""
    locale = parsed["locale"]
    rate_normalised = normalise_rate(parsed["rate"])

    # Build available_styles list from the express-as style
    available_styles: list[str] = []
    if parsed["style"]:
        available_styles.append(parsed["style"])
    # Add common narration styles if not already present
    for s in ["narration-professional", "newscast"]:
        if s not in available_styles:
            available_styles.append(s)

    locale_block: dict = {
        "azure_voice": parsed["voice_name"],
        "available_styles": available_styles,
    }
    if parsed["pitch"]:
        locale_block["azure_pitch"] = parsed["pitch"]
    else:
        locale_block["azure_pitch"] = "-5%"

    locale_block["azure_break_ms"] = 600

    if parsed["style_degree"]:
        try:
            locale_block["azure_style_degree"] = float(parsed["style_degree"])
        except ValueError:
            locale_block["azure_style_degree"] = 1.3
    else:
        locale_block["azure_style_degree"] = 1.3

    locale_block["azure_rate"] = rate_normalised

    # Preserve the original numeric rate if it was converted
    if parsed["rate"] and not parsed["rate"].endswith("%"):
        try:
            float(parsed["rate"])
            locale_block["ssml_rate_raw"] = parsed["rate"]
        except ValueError:
            pass

    entry = {
        "character_id": "narrator",
        "role": "narrator",
        "gender": "neutral",
        "personality": "SSML-authored narrator",
        locale: locale_block,
    }
    return entry


def update_voicecast(voicecast_path: Path, entry: dict, slug: str) -> None:
    """Read existing VoiceCast.json (if any) and append narrator entry if absent."""
    if voicecast_path.exists():
        try:
            vc = json.loads(voicecast_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read existing VoiceCast.json (%s); creating fresh", exc)
            vc = None
    else:
        vc = None

    if vc is None:
        vc = {
            "schema_id": "VoiceCast",
            "schema_version": "1.0.0",
            "project_id": slug,
            "characters": [],
        }

    # Check if narrator already exists
    characters = vc.get("characters", [])
    narrator_exists = any(c.get("character_id") == "narrator" for c in characters)

    if narrator_exists:
        log.info("VoiceCast.json already contains narrator entry; skipping")
    else:
        characters.append(entry)
        vc["characters"] = characters
        log.info("Added narrator entry to VoiceCast.json")

    # Ensure parent directory exists
    voicecast_path.parent.mkdir(parents=True, exist_ok=True)
    voicecast_path.write_text(
        json.dumps(vc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_pipeline_vars(meta: dict, detected_locale: str = "") -> str:
    """Build pipeline_vars.sh content from meta.json fields.

    *detected_locale* is the locale parsed from the SSML xml:lang attribute.
    It becomes PRIMARY_LOCALE — the authoritative source locale for the episode.
    """
    slug = meta["project_slug"]
    ep_id = meta["episode_id"]
    primary = detected_locale or meta.get("primary_locale", "en")

    lines = [
        f'export STORY_TITLE="{meta.get("story_title", "")}"',
        f'export EPISODE_NUMBER="{meta.get("episode_number", "")}"',
        f'export EPISODE_ID="{ep_id}"',
        f'export PRIMARY_LOCALE="{primary}"',
        f'export LOCALES="{meta.get("locales", primary)}"',
        f'export PROJECT_SLUG="{slug}"',
        f'export SERIES_GENRE="{meta.get("series_genre", "")}"',
        f'export GENERATION_SEED="{meta.get("generation_seed", "")}"',
        f'export RENDER_PROFILE="{meta.get("render_profile", "")}"',
        f'export STORY_FORMAT="{meta.get("story_format", "")}"',
        f'export PROJECT_DIR="projects/{slug}"',
        f'export EPISODE_DIR="projects/{slug}/episodes/{ep_id}"',
        f'export VOICE_CAST_FILE="projects/{slug}/VoiceCast.json"',
    ]
    return "\n".join(lines) + "\n"


def build_narration_text(fragments: list) -> str:
    """Build plain narration text by stripping all XML, restoring spacing."""
    parts: list[str] = []
    for kind, value in fragments:
        if kind == "text":
            parts.append(value)
        elif kind == "break":
            # Replace breaks with a space to maintain word boundaries
            if parts and not parts[-1].endswith((" ", "\n")):
                parts.append(" ")

    raw = "".join(parts)
    # Collapse multiple whitespace runs into single spaces
    raw = re.sub(r"[ \t]+", " ", raw)
    # Normalise line breaks
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 code/http/ssml_preprocess.py <EP_DIR>", file=sys.stderr)
        return 1

    ep_dir = Path(sys.argv[1]).resolve()

    # Validate inputs
    story_path = ep_dir / "story.txt"
    meta_path = ep_dir / "meta.json"

    if not ep_dir.is_dir():
        log.error("EP_DIR does not exist: %s", ep_dir)
        return 1
    if not story_path.is_file():
        log.error("story.txt not found in %s", ep_dir)
        return 1
    if not meta_path.is_file():
        log.error("meta.json not found in %s", ep_dir)
        return 1

    # Load inputs
    ssml_text = story_path.read_text(encoding="utf-8")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("Invalid JSON in meta.json: %s", exc)
        return 1

    slug = meta.get("project_slug", "")
    ep_id = meta.get("episode_id", "")
    if not slug or not ep_id:
        log.error("meta.json must contain 'project_slug' and 'episode_id'")
        return 1

    log.info("ssml_preprocess: %s/%s", slug, ep_id)

    # Parse SSML
    try:
        parsed = parse_ssml(ssml_text)
    except ET.ParseError as exc:
        log.error("Failed to parse SSML in story.txt: %s", exc)
        return 1

    locale = parsed["locale"]
    log.info("Detected locale: %s (from xml:lang)", locale)
    log.info("Voice: %s", parsed["voice_name"] or "(none)")
    log.info("Style: %s  degree: %s", parsed["style"] or "(none)", parsed["style_degree"] or "(none)")
    log.info("Rate: %s  pitch: %s", parsed["rate"] or "(none)", parsed["pitch"] or "(none)")

    # 1. Script.json
    script = build_script(meta, parsed["fragments"], locale)
    script_path = ep_dir / "Script.json"
    script_path.write_text(
        json.dumps(script, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    total_sents = sum(len(sc["actions"]) for sc in script["scenes"])
    log.info("Script.json: %d scenes, %d sentences -> %s", len(script["scenes"]), total_sents, script_path)

    # 2. VoiceCast.json (append-only)
    vc_entry = build_voicecast_entry(parsed)
    voicecast_path = PIPE_DIR / "projects" / slug / "VoiceCast.json"
    update_voicecast(voicecast_path, vc_entry, slug)
    log.info("VoiceCast.json -> %s", voicecast_path)

    # 3. ssml_inner.xml (immutable after creation)
    inner_path = ep_dir / "ssml_inner.xml"
    if inner_path.exists():
        log.info("ssml_inner.xml already exists; skipping (immutable)")
    else:
        inner_path.write_text(parsed["inner_xml"] + "\n", encoding="utf-8")
        log.info("ssml_inner.xml -> %s", inner_path)

    # 4. pipeline_vars.sh
    vars_content = build_pipeline_vars(meta, detected_locale=locale)
    vars_path = ep_dir / "pipeline_vars.sh"
    vars_path.write_text(vars_content, encoding="utf-8")
    log.info("pipeline_vars.sh -> %s", vars_path)

    # 5. NarrationText.txt
    narration = build_narration_text(parsed["fragments"])
    narration_path = ep_dir / "NarrationText.txt"
    narration_path.write_text(narration, encoding="utf-8")
    log.info("NarrationText.txt -> %s", narration_path)

    log.info("ssml_preprocess complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
