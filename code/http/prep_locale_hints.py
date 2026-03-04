#!/usr/bin/env python3
# =============================================================================
# prep_locale_hints.py  —  Phase 0: upstream character-count hints for Stage 8.
#
# Runs BEFORE Stage 8 (the translation LLM stage).
# Reads actual EN WAV durations from the previous TTS run and computes a
# target character count for each VO item in the target locale, using
# calibrated chars/sec rates from prompts/tts_calibration.{locale}.json.
#
# Writes: {ep_dir}/vo_hints.{locale}.json
#
# On first run (no EN WAVs synthesised yet) the script exits gracefully
# with a warning — Stage 8 proceeds using its own word-count estimation.
# From the second episode onward the calibration converges automatically.
#
# Usage:
#   python prep_locale_hints.py \
#       --manifest  projects/{slug}/{ep}/AssetManifest_draft.en.json \
#       --locale    zh-Hans
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

PIPE_DIR        = Path(__file__).resolve().parent.parent.parent
SHORT_THRESHOLD = 1.5   # seconds — below this use short_cps rate
MIN_CHARS       = 4     # floor for any target character count
GLOBAL_DEFAULTS = {"normal_cps": 3.2, "short_cps": 2.3}


# ── WAV helper ────────────────────────────────────────────────────────────────

def wav_duration_sec(path: Path) -> float | None:
    """Return WAV duration in seconds, or None if file is missing/corrupt."""
    try:
        with wave.open(str(path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


# ── Calibration helpers ───────────────────────────────────────────────────────

def load_calibration(locale: str) -> dict:
    p = PIPE_DIR / "prompts" / f"tts_calibration.{locale}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 2, "locale": locale, "history": [], "defaults": GLOBAL_DEFAULTS}


def _weighted_avg(entries: list) -> tuple[float, float]:
    nn = sum(e.get("normal_n", 0) for e in entries)
    sn = sum(e.get("short_n",  0) for e in entries)
    nc = sum(e.get("normal_cps", 3.5) * e.get("normal_n", 0) for e in entries)
    sc = sum(e.get("short_cps",  2.5) * e.get("short_n",  0) for e in entries)
    return (nc / nn if nn else 3.5), (sc / sn if sn else 2.5)


def get_cps(cal: dict, preset_hash: str | None,
            voice: str, style: str, rate: str,
            en_voice: str = "", en_style: str = "", en_rate: str = "") -> tuple[float, float]:
    """Return (normal_cps, short_cps) using the priority lookup ladder.

    cps = ZH_chars / EN_duration_sec, so it encodes a specific ZH+EN voice
    PAIR.  When en_voice is provided, priorities 2–3 require the entry to
    match BOTH voices.  Entries that predate the en_voice field (which have
    en_voice == "") fall through to priorities 4–6 for backward compatibility.
    """
    hist = cal.get("history", [])
    defs = cal.get("defaults", GLOBAL_DEFAULTS)

    def _match(pred, min_n: int = 1):
        m = [e for e in hist if pred(e)]
        return _weighted_avg(m) if len(m) >= min_n else None

    # Priority 1: exact preset_hash
    if preset_hash:
        r = _match(lambda e: e.get("preset_hash") == preset_hash)
        if r:
            return r

    if en_voice:
        # Priority 2: ZH voice+style+rate  AND  EN voice+style+rate  (full pair)
        r = _match(lambda e: e.get("voice") == voice
                   and e.get("style") == style and e.get("rate") == rate
                   and e.get("en_voice") == en_voice
                   and e.get("en_style") == en_style
                   and e.get("en_rate")  == en_rate)
        if r:
            return r
        # Priority 3: ZH voice+style+rate  AND  EN voice  (EN rate/style may differ)
        r = _match(lambda e: e.get("voice") == voice
                   and e.get("style") == style and e.get("rate") == rate
                   and e.get("en_voice") == en_voice)
        if r:
            return r

    # Priority 4: ZH voice+style+rate  (no EN — backward-compat for old entries)
    r = _match(lambda e: e.get("voice") == voice
               and e.get("style") == style and e.get("rate") == rate)
    if r:
        return r
    # Priority 5: ZH voice+style
    r = _match(lambda e: e.get("voice") == voice and e.get("style") == style)
    if r:
        return r
    # Priority 6: ZH voice only
    r = _match(lambda e: e.get("voice") == voice)
    if r:
        return r
    # Priority 7: global defaults
    return defs.get("normal_cps", 3.5), defs.get("short_cps", 2.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute locale character-count hints from EN WAV durations.")
    ap.add_argument("--manifest", required=True,
                    help="Path to AssetManifest_draft.en.json")
    ap.add_argument("--locale",   required=True,
                    help="Target locale, e.g. zh-Hans")
    ap.add_argument("--primary-locale", default="en",
                    help="Primary locale (source of truth for timing)")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        print(f"  ⚠  prep_locale_hints: {mpath.name} not found — skipping hints")
        sys.exit(0)

    manifest       = json.loads(mpath.read_text(encoding="utf-8"))
    locale         = args.locale
    primary_locale = args.primary_locale
    ep_dir         = mpath.parent
    proj_dir       = ep_dir.parent.parent
    wav_dir        = ep_dir / "assets" / primary_locale / "audio" / "vo"

    # Load VoiceCast.json for calibration lookup (voice/style/rate per character)
    vc_map: dict[str, dict] = {}
    vc_path = proj_dir / "VoiceCast.json"
    if vc_path.exists():
        for ch in json.loads(vc_path.read_text(encoding="utf-8")).get("characters", []):
            vc_map[ch["character_id"]] = ch

    cal = load_calibration(locale)
    hints: list[dict] = []
    skipped_no_wav = 0

    for item in manifest.get("vo_items", []):
        item_id  = item.get("item_id", "")
        speaker  = item.get("speaker_id", "narrator")
        wav_path = wav_dir / f"{item_id}.wav"
        en_dur   = wav_duration_sec(wav_path)
        if not en_dur:
            skipped_no_wav += 1
            continue

        # Look up ZH voice params for calibration
        ch       = vc_map.get(speaker, {})
        loc_data = ch.get(locale, {})
        voice    = loc_data.get("azure_voice", "")
        style    = loc_data.get("azure_style", "")
        rate     = loc_data.get("azure_rate", "0%")
        ph       = loc_data.get("preset_hash")      # set if user assigned a preset

        # Look up primary-locale reference voice params — cps is locale_chars/primary_sec
        # so it is pair-specific: changing primary voice speed changes the measured ratio.
        en_data  = ch.get(primary_locale, {})
        en_voice = en_data.get("azure_voice", "")
        en_style = en_data.get("azure_style", "")
        en_rate  = en_data.get("azure_rate", "0%")

        nc, sc  = get_cps(cal, ph, voice, style, rate, en_voice, en_style, en_rate)
        cps     = sc if en_dur <= SHORT_THRESHOLD else nc
        target  = max(MIN_CHARS, round(en_dur * cps))

        hints.append({
            "item_id":         item_id,
            "en_duration_sec": round(en_dur, 3),
            "target_chars":    target,
            "cps_used":        round(cps, 2),
            "fragment_type":   "short" if en_dur <= SHORT_THRESHOLD else "normal",
        })

    out_path = ep_dir / f"vo_hints.{locale}.json"
    out_path.write_text(
        json.dumps({"locale": locale, "hints": hints}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    skip_note = f"  ({skipped_no_wav} skipped — no EN wav yet)" if skipped_no_wav else ""
    print(f"  ✓ prep_locale_hints: {len(hints)} hints → {out_path.name}{skip_note}")


if __name__ == "__main__":
    main()
