#!/usr/bin/env python3
# =============================================================================
# polish_locale_vo.py  —  Phase 1: convergence loop for locale VO duration.
#
# Runs AFTER gen_tts_cloud.py [3/8] and BEFORE post_tts_analysis.py [4/8].
#
# Algorithm:
#   1. Measure actual ZH and EN WAV durations directly from files.
#   2. Flag lines with ratio = zh_dur / en_dur outside [THRESHOLD, THRESHOLD_HIGH].
#   3. Call Claude (sonnet) to rewrite flagged Chinese text to target_chars.
#   4. Patch AssetManifest_merged.{locale}.json + AssetManifest_draft.{locale}.json
#      with the revised text (so re-runs from Stage 10 also use corrected text).
#   5. Re-synthesize flagged items only via gen_tts_cloud.py --asset-id.
#   6. Re-measure. Repeat up to MAX_ITERS times.
#   7. Write observed chars/sec to prompts/tts_calibration.{locale}.json.
#
# Usage:
#   python polish_locale_vo.py \
#       --manifest  {ep_dir}/AssetManifest_merged.{locale}.json \
#       --locale    zh-Hans \
#       --ep-dir    {ep_dir}
# =============================================================================

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import wave
from pathlib import Path

PIPE_DIR        = Path(__file__).resolve().parent.parent.parent
CODE_DIR        = Path(__file__).resolve().parent
THRESHOLD       = 0.90   # zh/en ratio below this → expand
THRESHOLD_HIGH  = 1.10   # zh/en ratio above this → shorten
MIN_CHARS       = 4
SHORT_THRESHOLD = 1.5    # seconds
MAX_ITERS       = 3
GLOBAL_DEFAULTS = {"normal_cps": 3.2, "short_cps": 2.3}


# ── WAV + text helpers ────────────────────────────────────────────────────────

def wav_duration_sec(path: Path) -> float | None:
    try:
        with wave.open(str(path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def char_count(text: str) -> int:
    """Count non-punctuation, non-space characters (content chars only)."""
    return sum(1 for c in text
               if unicodedata.category(c)[0] not in ("P", "Z", "C"))


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration(locale: str) -> dict:
    p = PIPE_DIR / "prompts" / f"tts_calibration.{locale}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 2, "locale": locale, "history": [], "defaults": GLOBAL_DEFAULTS}


def save_calibration(locale: str, cal: dict) -> None:
    p = PIPE_DIR / "prompts" / f"tts_calibration.{locale}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2, ensure_ascii=False), encoding="utf-8")


def append_calibration_entry(
    locale: str,
    run_id: str,
    voice: str,
    style: str,
    rate: str,
    preset_hash: str | None,
    en_voice: str,
    en_style: str,
    en_rate: str,
    converged_items: list[dict],
) -> None:
    """Compute observed ZH-chars/EN-sec rate and append to history.

    The cps value = total_zh_chars / total_en_duration_sec, so it encodes
    the relationship between a SPECIFIC ZH voice setting and a SPECIFIC EN
    voice setting.  Both are stored in the entry so the lookup ladder can
    match on the exact pair rather than the ZH voice alone.
    """
    normal = [it for it in converged_items if it["en_dur"] > SHORT_THRESHOLD]
    short  = [it for it in converged_items if it["en_dur"] <= SHORT_THRESHOLD]

    def _cps(items: list) -> tuple[float | None, int]:
        if not items:
            return None, 0
        total_chars = sum(char_count(it["final_text"]) for it in items)
        total_dur   = sum(it["en_dur"] for it in items)
        return (round(total_chars / total_dur, 3) if total_dur else None), len(items)

    nc, nn = _cps(normal)
    sc, sn = _cps(short)
    if not nc and not sc:
        return

    cal   = load_calibration(locale)
    entry: dict = {
        "run_id":   run_id,
        "date":     datetime.date.today().isoformat(),
        # ZH voice (the locale being calibrated)
        "voice":    voice,
        "style":    style,
        "rate":     rate,
        # EN reference voice (the source whose duration is used as denominator)
        "en_voice": en_voice,
        "en_style": en_style,
        "en_rate":  en_rate,
    }
    if preset_hash:
        entry["preset_hash"] = preset_hash
    if nc:
        entry["normal_cps"] = nc
        entry["normal_n"]   = nn
    if sc:
        entry["short_cps"]  = sc
        entry["short_n"]    = sn

    cal.setdefault("history", []).append(entry)
    if len(cal["history"]) > 20:
        cal["history"] = cal["history"][-20:]
    save_calibration(locale, cal)
    print(f"  ✓ calibration updated — locale={locale}  "
          f"normal_cps={nc}(n={nn})  short_cps={sc}(n={sn})")


# ── Claude rewrite ────────────────────────────────────────────────────────────

BATCH_SIZE = 15  # max lines per Claude call to avoid timeout

def _call_claude_batch(batch: list[dict], locale: str) -> dict[str, str]:
    """Single Claude call for one batch of lines. Returns {item_id: revised_text}."""
    items_block = "\n".join(
        f'  - item_id: "{it["item_id"]}"\n'
        f'    direction: "{it["direction"]}"\n'
        f'    current_text: {json.dumps(it["text"], ensure_ascii=False)}\n'
        f'    current_chars: {it["current_chars"]}\n'
        f'    target_chars: {it["target_chars"]}\n'
        f'    en_duration_sec: {it["en_dur"]:.2f}\n'
        f'    zh_duration_sec: {it["zh_dur"]:.2f}\n'
        f'    ratio_zh_en: {it["ratio"]:.2f}'
        for it in batch
    )
    prompt = f"""You are a literary Chinese translator and voice-over adapter.

The following Chinese narration lines need adjustment so that the spoken duration matches
the English source line within a ±10% window (target ratio 0.90–1.10).

Each line has a direction field:
  "expand"  — line is too short; add sensory detail, atmosphere, or emotional texture
  "shorten" — line is too long;  trim or simplify while keeping the meaning and tone

Rules:
- Write natural, literary Chinese (classical or refined literary register suits narration)
- Expansions: add detail already implied by the scene — no meaningless filler
- Shortenings: cut redundant phrases, consolidate, do NOT lose the core image
- Target is approximate: ±2 characters from target_chars is acceptable
- Preserve the original meaning and emotional tone
- Reply with a JSON array only — no text outside the array

Lines to rewrite (locale: {locale}):
{items_block}

Required JSON format:
[{{"item_id": "...", "revised_text": "..."}}, ...]"""

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(prompt)
        tf_path = tf.name

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)   # allow nested claude invocation
        proc = subprocess.run(
            ["claude", "-p", tf_path,
             "--model", "sonnet",
             "--dangerously-skip-permissions",
             "--no-session-persistence"],
            capture_output=True, timeout=300, cwd=str(PIPE_DIR),
            env=env,
        )
        raw = proc.stdout.decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        print(f"  ✗ polish_locale_vo: Claude timed out on batch of {len(batch)}")
        return {}
    finally:
        os.unlink(tf_path)

    j0 = raw.find("["); j1 = raw.rfind("]") + 1
    if j0 < 0 or j1 <= j0:
        print(f"  ✗ polish_locale_vo: Claude returned no JSON array")
        return {}
    try:
        revisions = json.loads(raw[j0:j1])
        return {r["item_id"]: r["revised_text"]
                for r in revisions if "item_id" in r and "revised_text" in r}
    except Exception as exc:
        print(f"  ✗ polish_locale_vo: JSON parse error — {exc}")
        return {}


def call_claude_rewrite(flagged: list[dict], locale: str) -> dict[str, str]:
    """Ask Claude to rewrite flagged lines in batches of BATCH_SIZE.
    Returns {item_id: revised_text}."""
    if not flagged:
        return {}
    result = {}
    for i in range(0, len(flagged), BATCH_SIZE):
        batch = flagged[i:i + BATCH_SIZE]
        print(f"    claude batch {i//BATCH_SIZE + 1}/{(len(flagged)-1)//BATCH_SIZE + 1}"
              f" ({len(batch)} lines)…")
        result.update(_call_claude_batch(batch, locale))
    return result


# ── TTS re-synthesis ──────────────────────────────────────────────────────────

def resynthesize_items(manifest_path: Path, item_ids: list[str]) -> None:
    """Re-synthesize specific VO items via gen_tts_cloud.py --asset-id."""
    for iid in item_ids:
        print(f"    re-synthesising {iid}…")
        subprocess.run(
            ["python3", str(CODE_DIR / "gen_tts_cloud.py"),
             "--manifest", str(manifest_path),
             "--asset-id", iid],
            cwd=str(PIPE_DIR),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convergence loop: expand short locale VO lines to match EN timing.")
    ap.add_argument("--manifest", required=True,
                    help="AssetManifest_merged.{locale}.json")
    ap.add_argument("--locale",   required=True,
                    help="Target locale, e.g. zh-Hans")
    ap.add_argument("--ep-dir",   required=True,
                    help="Episode directory path")
    args = ap.parse_args()

    manifest_path     = Path(args.manifest)
    ep_dir            = Path(args.ep_dir)
    locale            = args.locale
    draft_locale_path = ep_dir / f"AssetManifest_draft.{locale}.json"

    if not manifest_path.exists():
        print(f"  ✗ polish_locale_vo: {manifest_path} not found")
        sys.exit(1)

    wav_zh = ep_dir / "assets" / locale / "audio" / "vo"
    wav_en = ep_dir / "assets" / "en"   / "audio" / "vo"

    if not wav_en.exists():
        print(f"  ⚠  polish_locale_vo: EN WAV dir not found ({wav_en}) — skipping")
        sys.exit(0)

    # Load manifests
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    draft    = (json.loads(draft_locale_path.read_text(encoding="utf-8"))
                if draft_locale_path.exists() else None)

    # In-memory index for fast lookup
    merged_idx: dict[str, dict] = {
        it["item_id"]: it for it in manifest.get("vo_items", [])}
    draft_idx: dict[str, dict] = {
        it["item_id"]: it for it in draft.get("vo_items", [])} if draft else {}

    # VoiceCast for calibration metadata
    proj_dir  = ep_dir.parent.parent
    vc_map: dict[str, dict] = {}
    vc_path = proj_dir / "VoiceCast.json"
    if vc_path.exists():
        for ch in json.loads(vc_path.read_text(encoding="utf-8")).get("characters", []):
            vc_map[ch["character_id"]] = ch

    def _voice_info(speaker_id: str, loc_key: str | None = None) -> tuple[str, str, str, str | None]:
        """Return (voice, style, rate, preset_hash) for speaker in the given locale."""
        ch = vc_map.get(speaker_id) or vc_map.get("narrator") or {}
        loc = ch.get(loc_key or locale, {})
        return (loc.get("azure_voice", ""),
                loc.get("azure_style", ""),
                loc.get("azure_rate", "0%"),
                loc.get("preset_hash"))

    slug   = proj_dir.name
    ep_id  = ep_dir.name
    run_id = f"{slug}/{ep_id}"

    all_converged: list[dict] = []
    representative_speaker    = "narrator"

    print(f"\n  ── polish_locale_vo: locale={locale}  "
          f"threshold={THRESHOLD}  max_iters={MAX_ITERS}")

    # ── Snapshot initial ZH/EN ratios before any rewriting ───────────────────
    initial_snapshot: dict[str, dict] = {}
    for item in manifest.get("vo_items", []):
        iid    = item.get("item_id", "")
        zh_dur = wav_duration_sec(wav_zh / f"{iid}.wav")
        en_dur = wav_duration_sec(wav_en / f"{iid}.wav")
        if zh_dur and en_dur and en_dur > 0:
            initial_snapshot[iid] = {
                "ratio": round(zh_dur / en_dur, 3),
                "chars": char_count(item.get("text", "")),
                "text":  item.get("text", ""),
            }

    # Track original text before first rewrite per item
    original_text: dict[str, str] = {}

    for iteration in range(1, MAX_ITERS + 1):
        # ── Measure all ZH and EN durations ──────────────────────────────────
        flagged: list[dict] = []
        for item in manifest.get("vo_items", []):
            iid    = item.get("item_id", "")
            text   = item.get("text", "")
            zh_dur = wav_duration_sec(wav_zh / f"{iid}.wav")
            en_dur = wav_duration_sec(wav_en / f"{iid}.wav")
            if not zh_dur or not en_dur or en_dur <= 0:
                continue
            ratio = zh_dur / en_dur
            if ratio < THRESHOLD or ratio > THRESHOLD_HIGH:
                direction    = "expand" if ratio < THRESHOLD else "shorten"
                target_chars = max(MIN_CHARS,
                                   round(char_count(text) * (en_dur / zh_dur)))
                flagged.append({
                    "item_id":       iid,
                    "text":          text,
                    "direction":     direction,
                    "current_chars": char_count(text),
                    "target_chars":  target_chars,
                    "zh_dur":        zh_dur,
                    "en_dur":        en_dur,
                    "ratio":         ratio,
                    "speaker_id":    item.get("speaker_id", "narrator"),
                })

        if not flagged:
            print(f"  ✓ iter {iteration}: all lines in [{THRESHOLD}, {THRESHOLD_HIGH}] — converged")
            break

        n_low  = sum(1 for f in flagged if f["direction"] == "expand")
        n_high = sum(1 for f in flagged if f["direction"] == "shorten")
        worst  = min(f["ratio"] for f in flagged if f["direction"] == "expand") if n_low  else None
        best   = max(f["ratio"] for f in flagged if f["direction"] == "shorten") if n_high else None
        parts  = []
        if n_low:  parts.append(f"{n_low} too-short (worst {worst:.2f})")
        if n_high: parts.append(f"{n_high} too-long (worst {best:.2f})")
        print(f"  iter {iteration}: {len(flagged)} lines out of range — {', '.join(parts)}")
        for f in sorted(flagged, key=lambda x: x["ratio"])[:3]:
            print(f"    {f['item_id']:30s}  ratio={f['ratio']:.2f}  "
                  f"chars {f['current_chars']}→{f['target_chars']}")

        # ── Ask Claude to rewrite ─────────────────────────────────────────────
        revisions = call_claude_rewrite(flagged, locale)
        if not revisions:
            print("  ✗ no revisions returned — stopping loop")
            break

        # ── Patch manifests in memory ─────────────────────────────────────────
        patched_ids: list[str] = []
        for item in flagged:
            iid      = item["item_id"]
            new_text = revisions.get(iid)
            if not new_text:
                continue
            # Record original text before first rewrite
            if iid not in original_text:
                original_text[iid] = item["text"]
            if iid in merged_idx:
                merged_idx[iid]["text"] = new_text
            if iid in draft_idx:
                draft_idx[iid]["text"] = new_text
            patched_ids.append(iid)

        if not patched_ids:
            break

        # Write updated manifests to disk before re-synthesis
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if draft and draft_locale_path.exists():
            draft_locale_path.write_text(
                json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")

        # ── Re-synthesize patched items ───────────────────────────────────────
        print(f"  re-synthesising {len(patched_ids)} item(s)…")
        resynthesize_items(manifest_path, patched_ids)

    # ── Collect converged items for calibration ───────────────────────────────
    final_snapshot: dict[str, dict] = {}
    for item in manifest.get("vo_items", []):
        iid    = item.get("item_id", "")
        zh_dur = wav_duration_sec(wav_zh / f"{iid}.wav")
        en_dur = wav_duration_sec(wav_en / f"{iid}.wav")
        if zh_dur and en_dur and en_dur > 0:
            ratio = zh_dur / en_dur
            final_snapshot[iid] = {
                "ratio": round(ratio, 3),
                "chars": char_count(item.get("text", "")),
                "text":  item.get("text", ""),
            }
            if THRESHOLD <= ratio <= THRESHOLD_HIGH:
                all_converged.append({
                    "item_id":    iid,
                    "final_text": item.get("text", ""),
                    "zh_dur":     zh_dur,
                    "en_dur":     en_dur,
                })
                representative_speaker = item.get("speaker_id", "narrator")

    total = len(list(manifest.get("vo_items", [])))
    print(f"  ✓ polish_locale_vo complete: "
          f"{len(all_converged)}/{total} lines converged "
          f"(ratio in [{THRESHOLD}, {THRESHOLD_HIGH}])")

    # ── Write alignment report ────────────────────────────────────────────────
    flagged_ids = {iid for iid, s in initial_snapshot.items()
                   if s["ratio"] < THRESHOLD or s["ratio"] > THRESHOLD_HIGH}
    al_lines = []
    for iid in sorted(flagged_ids, key=lambda x: initial_snapshot[x]["ratio"]):
        init  = initial_snapshot[iid]
        final = final_snapshot.get(iid, init)
        al_lines.append({
            "item_id":      iid,
            "ratio_before": init["ratio"],
            "ratio_after":  final["ratio"],
            "chars_before": init["chars"],
            "chars_after":  final["chars"],
            "text_before":  original_text.get(iid, init["text"]),
            "text_after":   final["text"],
            "rewritten":    iid in original_text,
        })
    alignment = {
        "locale":              locale,
        "updated":             datetime.datetime.now().isoformat(timespec="seconds"),
        "total_lines":         len(initial_snapshot),
        "flagged_count":       len(flagged_ids),
        "converged_count":     sum(1 for s in final_snapshot.values()
                                   if THRESHOLD <= s["ratio"] <= THRESHOLD_HIGH),
        "worst_ratio_before":  round(min((s["ratio"] for s in initial_snapshot.values()),
                                         default=1.0), 3),
        "worst_ratio_after":   round(min((s["ratio"] for s in final_snapshot.values()),
                                         default=1.0), 3),
        "best_ratio_before":   round(max((s["ratio"] for s in initial_snapshot.values()),
                                         default=1.0), 3),
        "best_ratio_after":    round(max((s["ratio"] for s in final_snapshot.values()),
                                         default=1.0), 3),
        "lines":               al_lines,
    }
    align_path = ep_dir / f"vo_alignment.{locale}.json"
    align_path.write_text(
        json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ alignment report → {align_path.name}")

    # ── Update calibration history ────────────────────────────────────────────
    if all_converged:
        zh_voice, zh_style, zh_rate, ph = _voice_info(representative_speaker)
        en_voice, en_style, en_rate, _  = _voice_info(representative_speaker, "en")
        append_calibration_entry(
            locale, run_id, zh_voice, zh_style, zh_rate, ph,
            en_voice, en_style, en_rate, all_converged)


if __name__ == "__main__":
    main()
