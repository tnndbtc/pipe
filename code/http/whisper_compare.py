#!/usr/bin/env python3
# whisper_compare.py — Run Whisper on TTS WAVs and compare to intended text.
#
# Usage:
#   python3 whisper_compare.py <ep_dir> <locale>
#
# Reads:  {ep_dir}/AssetManifest.{locale}.json   → vo_items[].{item_id, text}
#         {ep_dir}/assets/{locale}/audio/vo/{item_id}.wav
# Writes: {ep_dir}/assets/{locale}/whisper_compare.json
#
# Runs after Stage 3.5 (gen_tts + post_tts_analysis).
# Called with || true in run.sh — must never crash the pipeline.

import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

# Whisper --language code mapping (BCP-47 locale → Whisper language tag)
_LANG_MAP = {
    "en":      "en",
    "zh-Hans": "zh",
    "zh-Hant": "zh",
}


def _lang_code(locale: str) -> str:
    """Return the Whisper --language code for a locale tag."""
    return _LANG_MAP.get(locale, locale.split("-")[0])


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace → word token list."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _wer(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """
    Word Error Rate via stdlib difflib.SequenceMatcher.

    WER = (insertions + deletions + substitutions) / len(ref_tokens)

    Returns 0.0 when ref_tokens is empty (nothing to compare).
    """
    if not ref_tokens:
        return 0.0

    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    ins = dels = subs = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            ins += j2 - j1
        elif tag == "delete":
            dels += i2 - i1
        elif tag == "replace":
            # replace of n ref words with m hyp words:
            # min(n,m) substitutions + abs(n-m) ins or dels
            n, m = i2 - i1, j2 - j1
            subs += min(n, m)
            if n > m:
                dels += n - m
            else:
                ins += m - n
    return (ins + dels + subs) / len(ref_tokens)


def _status(wer: float) -> str:
    if wer <= 0.10:
        return "ok"
    if wer <= 0.25:
        return "warn"
    return "fail"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: whisper_compare.py <ep_dir> <locale>", file=sys.stderr)
        sys.exit(1)

    ep_dir = Path(sys.argv[1]).resolve()
    locale = sys.argv[2].strip()
    lang   = _lang_code(locale)

    # Load manifest
    manifest_path = ep_dir / f"AssetManifest.{locale}.json"
    if not manifest_path.exists():
        print(f"[whisper_compare] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    vo_items = manifest.get("vo_items", [])
    if not vo_items:
        print("[whisper_compare] No vo_items in manifest — nothing to compare.")
        sys.exit(0)

    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"
    out_dir = ep_dir / "assets" / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    results   = []
    counts    = {"ok": 0, "warn": 0, "fail": 0, "missing": 0}

    tmp_dir = tempfile.mkdtemp(prefix="whisper_compare_")
    try:
        for item in vo_items:
            item_id  = item.get("item_id", "")
            intended = item.get("text", "")
            wav_path = vo_dir / f"{item_id}.wav"

            if not wav_path.exists():
                results.append({
                    "item_id":    item_id,
                    "intended":   intended,
                    "transcript": "",
                    "wer":        None,
                    "status":     "missing",
                })
                counts["missing"] += 1
                print(f"  [MISS] {item_id} — WAV not found")
                continue

            # Run Whisper
            try:
                subprocess.run(
                    [
                        "whisper", str(wav_path),
                        "--model",         "base",
                        "--language",      lang,
                        "--output_format", "json",
                        "--output_dir",    tmp_dir,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                out_json = Path(tmp_dir) / f"{wav_path.stem}.json"
                with open(out_json, encoding="utf-8") as fj:
                    whisper_data = json.load(fj)
                transcript = whisper_data.get("text", "").strip()
            except Exception as exc:
                print(f"  [WARN] Whisper failed for {item_id}: {exc}", file=sys.stderr)
                results.append({
                    "item_id":    item_id,
                    "intended":   intended,
                    "transcript": "",
                    "wer":        None,
                    "status":     "missing",
                })
                counts["missing"] += 1
                continue

            ref_tokens = _normalise(intended)
            hyp_tokens = _normalise(transcript)
            wer_val    = _wer(ref_tokens, hyp_tokens)
            st         = _status(wer_val)
            counts[st] += 1

            results.append({
                "item_id":    item_id,
                "intended":   intended,
                "transcript": transcript,
                "wer":        round(wer_val, 4),
                "status":     st,
            })
            icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[st]
            print(f"  [{icon}] {item_id}  WER={wer_val:.2%}  ({st})")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {
        "total":   len(results),
        "ok":      counts["ok"],
        "warn":    counts["warn"],
        "fail":    counts["fail"],
        "missing": counts["missing"],
    }

    output = {
        "locale":       locale,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items":        results,
        "summary":      summary,
    }

    out_path = out_dir / "whisper_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"\n  [whisper_compare] Done: {summary['total']} items — "
        f"✓ {summary['ok']} ok  ⚠ {summary['warn']} warn  "
        f"✗ {summary['fail']} fail  ○ {summary['missing']} missing"
    )
    print(f"  [whisper_compare] Written: {out_path}")


if __name__ == "__main__":
    main()
