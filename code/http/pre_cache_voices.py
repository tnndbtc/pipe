#!/usr/bin/env python3
# =============================================================================
# pre_cache_voices.py
# Pre-download Azure TTS sample clips for all listed voices × styles.
#
# Usage:
#   python code/http/pre_cache_voices.py              # full run (~44 min)
#   python code/http/pre_cache_voices.py --dry-run    # print queue, no API calls
#   python code/http/pre_cache_voices.py --voice en-US-AriaNeural
#   python code/http/pre_cache_voices.py --voice en-US-AriaNeural --style whispering
#
# Env vars (same as gen_tts_cloud.py):
#   AZURE_SPEECH_KEY     Azure subscription key  (required)
#   AZURE_SPEECH_REGION  e.g. "eastus"           (required unless AZURE_ENDPOINT set)
#   AZURE_ENDPOINT       Custom endpoint URL      (optional)
#
# Cache key formula (must match Voice Cast Editor exactly):
#   sha256(json.dumps({"v":voice,"s":style or "","d":style_degree,"r":rate,
#                      "p":pitch or "","b":break_ms,"t":text}, sort_keys=True))
#   Defaults: style_degree=1.0  rate="0%"  pitch=""  break_ms=0
#   Only styled clips are cached — no-style clips are generated on first
#   [▶ Sample] click in the editor and cached there.
#
# Files:  projects/resources/azure_tts/{voice_dir}/{hash[:16]}.mp3
# Index:  projects/resources/azure_tts/index.json
# =============================================================================

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PIPE_DIR   = Path(__file__).resolve().parent.parent.parent
CACHE_DIR  = PIPE_DIR / "projects" / "resources" / "azure_tts"
INDEX_FILE = CACHE_DIR / "index.json"

DELAY_MIN    = 5
DELAY_MAX    = 10
RETRY_DELAYS = [5, 10, 20]   # seconds on 429

DEFAULT_STYLE_DEGREE = 1.0
DEFAULT_RATE         = "0%"
DEFAULT_PITCH        = ""    # empty = omitted from SSML and hash
DEFAULT_BREAK_MS     = 0

# ── Path helpers ──────────────────────────────────────────────────────────────

def voice_to_dir(voice: str) -> str:
    """Sanitise voice name for filesystem use. Replaces ':' with '_'.
    Dragon HD voice names contain colons which are illegal on Windows.
    NOTE: cache_key() always uses the raw voice name, not this form.
    The Voice Cast Editor must apply the same function when building paths.
    """
    return voice.replace(":", "_")


def cache_key(voice: str, style: str) -> str:
    """style is always a non-empty string here — build_work_queue() only
    enqueues styled clips (Option A, no baselines pre-cached)."""
    text = sentence_for(style, VOICES[voice]["locale_group"])
    key  = {"v": voice, "s": style, "d": DEFAULT_STYLE_DEGREE,
            "r": DEFAULT_RATE, "p": DEFAULT_PITCH or "", "b": DEFAULT_BREAK_MS, "t": text}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def clip_path(voice: str, key: str) -> Path:
    return CACHE_DIR / voice_to_dir(voice) / f"{key[:16]}.mp3"

# ── Voices ────────────────────────────────────────────────────────────────────

VOICES: dict[str, dict] = {
    # EN MALE
    "en-US-DavisNeural":               {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["chat","angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-US-GuyNeural":                 {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["newscast","angry","cheerful","sad","excited","friendly","terrified","shouting","unfriendly","whispering","hopeful"]},
    "en-US-JasonNeural":               {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-US-TonyNeural":                {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-GB-RyanNeural":                {"azure_locale": "en-GB", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["cheerful","chat","whispering","sad"]},
    "en-US-DerekMultilingualNeural":   {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["empathetic","excited","relieved","shy"]},
    "en-US-DavisMultilingualNeural":   {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["empathetic","funny","relieved"]},
    "en-US-AndrewMultilingualNeural":  {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["empathetic","relieved"]},
    "en-US-KaiNeural":                 {"azure_locale": "en-US", "locale_group": "en", "gender": "Male",   "type": "STANDARD", "styles": ["conversation"]},
    # EN FEMALE
    "en-US-AriaNeural":                {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["chat","customerservice","narration-professional","newscast-casual","newscast-formal","cheerful","empathetic","angry","sad","excited","friendly","terrified","shouting","unfriendly","whispering","hopeful"]},
    "en-US-JennyNeural":               {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["assistant","chat","customerservice","newscast","angry","cheerful","sad","excited","friendly","terrified","shouting","unfriendly","whispering","hopeful"]},
    "en-US-JaneNeural":                {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-US-NancyNeural":               {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-US-SaraNeural":                {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["angry","cheerful","excited","friendly","hopeful","sad","shouting","terrified","unfriendly","whispering"]},
    "en-US-SerenaMultilingualNeural":  {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["empathetic","excited","friendly","shy","serious","relieved","sad"]},
    "en-US-NancyMultilingualNeural":   {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["excited","friendly","funny","relieved","shy"]},
    "en-US-AvaNeural":                 {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["angry","fearful","sad"]},
    "en-US-PhoebeMultilingualNeural":  {"azure_locale": "en-US", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["empathetic","sad","serious"]},
    "en-IN-NeerjaNeural":              {"azure_locale": "en-IN", "locale_group": "en", "gender": "Female", "type": "STANDARD", "styles": ["newscast","cheerful","empathetic"]},
    # ZH MALE
    "zh-CN-YunxiNeural":               {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["narration-relaxed","embarrassed","fearful","cheerful","disgruntled","serious","angry","sad","depressed","chat","assistant","newscast"]},
    "zh-CN-YunjianNeural":             {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["narration-relaxed","sports-commentary","sports-commentary-excited","angry","disgruntled","cheerful","sad","serious","depressed","documentary-narration"]},
    "zh-CN-YunzeNeural":               {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["calm","fearful","cheerful","disgruntled","serious","angry","sad","depressed","documentary-narration"]},
    "zh-CN-Yunxia:DragonHDFlashLatestNeural": {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "DRAGON",   "styles": ["affectionate","angry","comfort","cheerful","encourage","excited","fearful","sad","surprised"]},
    "zh-CN-YunyeNeural":               {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["embarrassed","calm","fearful","cheerful","disgruntled","serious","angry","sad"]},
    "zh-CN-YunfengNeural":             {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["angry","disgruntled","cheerful","fearful","sad","serious","depressed"]},
    "zh-CN-Yunyi:DragonHDFlashLatestNeural":  {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "DRAGON",   "styles": ["assassin","captain","cavalier","drake","gamenarrator","geomancer","poet"]},
    "zh-CN-YunxiaNeural":              {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["calm","fearful","cheerful","angry","sad"]},
    "zh-CN-YunyangNeural":             {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["customerservice","narration-professional","newscast-casual"]},
    "zh-CN-YunhaoNeural":              {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Male",   "type": "STANDARD", "styles": ["advertisement-upbeat"]},
    # ZH FEMALE
    "zh-CN-Xiaoxiao2:DragonHDFlashLatestNeural": {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "DRAGON",   "styles": ["affectionate","angry","anxious","cheerful","curious","disappointed","empathetic","encouragement","excited","fearful","guilty","lonely","poetry-reading","sad","surprised","sentiment","sorry","story","whisper","tired"]},
    "zh-CN-XiaoxiaoNeural":            {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["assistant","chat","customerservice","newscast","affectionate","angry","calm","cheerful","disgruntled","fearful","gentle","lyrical","sad","serious","poetry-reading","friendly","chat-casual","whispering","sorry","excited"]},
    "zh-CN-XiaomoNeural":              {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["embarrassed","calm","fearful","cheerful","disgruntled","serious","angry","sad","depressed","affectionate","gentle","envious"]},
    "zh-CN-XiaohanNeural":             {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["calm","fearful","cheerful","disgruntled","serious","angry","sad","gentle","affectionate","embarrassed"]},
    "zh-CN-XiaoyiNeural":              {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["angry","disgruntled","affectionate","cheerful","fearful","sad","embarrassed","serious","gentle"]},
    "zh-CN-Xiaoyi:DragonHDFlashLatestNeural":  {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "DRAGON",   "styles": ["angry","cheerful","complaining","cutesy","gentle","nervous","sad","shy","strict"]},
    "zh-CN-Xiaoxiao:DragonHDFlashLatestNeural": {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "DRAGON",   "styles": ["angry","chat","cheerful","excited","fearful","sad","voiceassistant","customerservice"]},
    "zh-CN-XiaoxiaoMultilingualNeural":{"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["affectionate","cheerful","empathetic","excited","poetry-reading","sorry","story"]},
    "zh-CN-Xiaoyou:DragonHDFlashLatestNeural": {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "DRAGON",   "styles": ["chat","angry","cheerful","poetry-reading","sad","story","cute"]},
    "zh-CN-XiaoyouMultilingualNeural": {"azure_locale": "zh-CN", "locale_group": "zh", "gender": "Female", "type": "STANDARD", "styles": ["chat","angry","cheerful","poetry-reading","sad","story","cute"]},
}

# ── Sentence bank ─────────────────────────────────────────────────────────────

SENTENCES: dict[str, dict[str, str]] = {
    # "baseline" removed — sentence_for() now uses "bedtime" as the fallback
    # for both None style and unknown styles, matching the editor's vcSampleText().
    "bedtime":        {"en": "Breathe in\u2026 and let your shoulders soften. The night is quiet, and you are safe.",
                       "zh": "慢慢吸一口气，把肩膀放松。夜很安静，你很安全。"},
    "narrator":       {"en": "It was a quiet evening — the kind that makes you forget how loud the world can be.",
                       "zh": "那是个安静的傍晚——那种让人忘记世界有多嘈杂的夜晚。"},
    "documentary":    {"en": "For millions of years, these mountains have stood as silent witnesses to all that lives below.",
                       "zh": "数百万年来，这些山脉默默伫立，见证着脚下一切生命的来去。"},
    "epic":           {"en": "Under a cold moon, the ancient stones remember every name they have ever known.",
                       "zh": "冷月之下，古老的石墙记得每一个曾经存在过的名字。"},
    "poet":           {"en": "The river does not remember the rain that made it, yet carries all things to the sea.",
                       "zh": "河流不记得造就它的雨水，却将万物带向大海。"},
    "angry":          {"en": "This will not stand. I have given everything, and still it is never enough.",
                       "zh": "这不能接受。我已经付出了一切，却永远都不够。"},
    "fear":           {"en": "Something is wrong. I can feel it — the silence where there should be sound.",
                       "zh": "有什么不对劲。我能感觉到——本该有声音的地方，却一片寂静。"},
    "sad":            {"en": "Some things, once broken, can never truly be made whole again.",
                       "zh": "有些事情，一旦破碎，就再也无法复原了。"},
    "warm":           {"en": "Even in the darkest hour, a single light is enough to find the way home.",
                       "zh": "即使在最黑暗的时刻，一点点光芒也足以找到回家的路。"},
    "curious":        {"en": "Wait — what is that? I have never seen anything like it.",
                       "zh": "等等——这是什么？我从来没见过这样的东西。"},
    "social":         {"en": "I honestly don't know what to say. I should have handled that better.",
                       "zh": "我真的不知道该说什么。我本应该处理得更好的。"},
    "professional":   {"en": "Reporting from the capital — tonight's session concluded with a unanimous vote.",
                       "zh": "来自首都的报道——今晚的会议以全票通过结束。"},
    "sports":         {"en": "And he drives forward — the crowd is on their feet — can he make it across the line?",
                       "zh": "他向前冲去——观众全站了起来——他能做到吗？"},
    "sports-excited": {"en": "UNBELIEVABLE! What a finish! Nobody saw that coming!",
                       "zh": "难以置信！什么样的结局！没有任何人预料到这一幕！"},
    "commercial":     {"en": "Limited time only — get the best price of the year, today.",
                       "zh": "限时特惠！现在购买，享受全年最低价！"},
}

STYLE_CATEGORY: dict[str, str] = {
    "calm": "bedtime", "gentle": "bedtime", "whispering": "bedtime", "whisper": "bedtime",
    "narration-relaxed": "narrator", "narration-professional": "narrator",
    "documentary-narration": "documentary", "newscast-formal": "documentary",
    "drake": "epic", "geomancer": "epic", "cavalier": "epic",
    "captain": "epic", "assassin": "epic", "gamenarrator": "epic",
    "poet": "poet", "lyrical": "poet", "poetry-reading": "poet", "story": "poet", "sentiment": "poet",
    "angry": "angry", "shouting": "angry", "disgruntled": "angry",
    "complaining": "angry", "unfriendly": "angry", "strict": "angry",
    "terrified": "fear", "fearful": "fear", "anxious": "fear", "nervous": "fear",
    "sad": "sad", "depressed": "sad", "disappointed": "sad", "tired": "sad",
    "lonely": "sad", "sorry": "sad", "guilty": "sad",
    "cheerful": "warm", "excited": "warm", "friendly": "warm", "hopeful": "warm",
    "affectionate": "warm", "encouragement": "warm", "encourage": "warm",
    "comfort": "warm", "cute": "warm", "cutesy": "warm",
    "curious": "curious", "surprised": "curious",
    "shy": "social", "embarrassed": "social", "envious": "social",
    "empathetic": "social", "relieved": "social", "funny": "social",
    "newscast": "professional", "newscast-casual": "professional",
    "chat": "professional", "chat-casual": "professional",
    "conversation": "professional", "assistant": "professional",
    "customerservice": "professional", "serious": "professional", "voiceassistant": "professional",
    "sports-commentary": "sports",
    "sports-commentary-excited": "sports-excited",
    "advertisement-upbeat": "commercial", "livecommercial": "commercial",
}


def sentence_for(style: str | None, locale_group: str) -> str:
    # style=None → "bedtime" (matches editor Sample button which calls
    # vcSampleText(locale, null) → always returns bedtime category).
    # Unknown style → "bedtime" (safe default, same as editor).
    cat  = STYLE_CATEGORY.get(style, "bedtime") if style else "bedtime"
    lang = "zh" if locale_group == "zh" else "en"
    return SENTENCES[cat][lang]

# ── Index ─────────────────────────────────────────────────────────────────────

def load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"schema_version": "1.0", "voices": {}}


def save_index(index: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX_FILE)


def already_cached(index: dict, voice: str, style: str) -> bool:
    clip = index.get("voices", {}).get(voice, {}).get("clips", {}).get(style)
    return bool(clip) and clip_path(voice, clip["hash"]).exists()

# ── SSML ──────────────────────────────────────────────────────────────────────

def build_ssml(text: str, voice: str, azure_lang: str, style: str | None) -> str:
    escaped = (text
               .replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;").replace('"', "&quot;")
               .replace("'", "&apos;"))
    spoken = escaped
    if DEFAULT_BREAK_MS:
        spoken = f'{spoken}<break time="{DEFAULT_BREAK_MS}ms"/>'
    rate_attr  = f' rate="{DEFAULT_RATE}"'   if DEFAULT_RATE  and DEFAULT_RATE  != "0%" else ""
    pitch_attr = f' pitch="{DEFAULT_PITCH}"' if DEFAULT_PITCH and DEFAULT_PITCH != "0%" else ""
    if rate_attr or pitch_attr:
        spoken = f'<prosody{rate_attr}{pitch_attr}>{spoken}</prosody>'
    if style:
        spoken = (f'<mstts:express-as style="{style}" styledegree="{DEFAULT_STYLE_DEGREE}">'
                  f"{spoken}</mstts:express-as>")
    return (f"<speak version='1.0' xml:lang='{azure_lang}' "
            f"xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{voice}'>{spoken}</voice></speak>")

# ── Azure ─────────────────────────────────────────────────────────────────────

def load_synthesizer():
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise SystemExit("Run: pip install azure-cognitiveservices-speech>=1.38.0")

    key      = os.environ.get("AZURE_SPEECH_KEY", "")
    region   = os.environ.get("AZURE_SPEECH_REGION", "")
    endpoint = os.environ.get("AZURE_ENDPOINT", "")
    if not key:
        raise SystemExit("[ERROR] AZURE_SPEECH_KEY not set.")
    if not region and not endpoint:
        raise SystemExit("[ERROR] AZURE_SPEECH_REGION not set.")

    cfg = (speechsdk.SpeechConfig(endpoint=endpoint, subscription=key)
           if endpoint else
           speechsdk.SpeechConfig(subscription=key, region=region))
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3)
    return speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)


def synthesize(synth, ssml: str) -> bytes:
    import azure.cognitiveservices.speech as speechsdk
    result = synth.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    d = result.cancellation_details
    raise RuntimeError(f"Azure TTS cancelled: {d.reason} — {d.error_details}")

# ── Write ─────────────────────────────────────────────────────────────────────

def write_clip(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)

# ── Process one ───────────────────────────────────────────────────────────────

def process_one(voice: str, style: str, index: dict, synth, dry_run: bool) -> bool:
    """Generate and cache one clip. Returns True if an API call was made.
    Index is updated only after write_clip's atomic rename succeeds.
    A failed write leaves the index unchanged; clip retries on next run.
    """
    info  = VOICES[voice]
    text  = sentence_for(style, info["locale_group"])
    key   = cache_key(voice, style)
    path  = clip_path(voice, key)
    label = style
    cat   = STYLE_CATEGORY.get(style, "bedtime")

    if dry_run:
        print(f"  [DRY]  {voice:<45}  {label}")
        return False

    ssml = build_ssml(text, voice, info["azure_locale"], style)

    # Synthesize with 429 retry
    data = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            print(f"    retry in {delay}s...", flush=True)
            time.sleep(delay)
        try:
            data = synthesize(synth, ssml)
            break
        except RuntimeError as e:
            if "429" in str(e) or "TooManyRequests" in str(e):
                if attempt < len(RETRY_DELAYS):
                    continue
            print(f"\n  [FAIL] {voice}  {label}: {e}")
            return True   # made_call=True so caller saves index & sleeps

    write_clip(path, data)

    # Update index only after successful write
    ve = index["voices"].setdefault(voice, {
        "locale": info["azure_locale"], "locale_group": info["locale_group"],
        "gender": info["gender"], "type": info["type"], "clips": {},
    })
    ve["clips"][label] = {
        "hash":     key,
        "file":     f"{voice_to_dir(voice)}/{key[:16]}.mp3",
        "text":     text,
        "category": cat,
        "params":   {"style": style, "style_degree": DEFAULT_STYLE_DEGREE,
                     "rate": DEFAULT_RATE, "pitch": DEFAULT_PITCH, "break_ms": DEFAULT_BREAK_MS},
    }
    return True

# ── Work queue ────────────────────────────────────────────────────────────────

def build_work_queue(index: dict, voice_filter: str | None,
                     style_filter: str | None) -> list[tuple]:
    queue, skipped = [], 0
    for voice, info in VOICES.items():
        if voice_filter and voice != voice_filter:
            continue
        # Styled clips only — no-style (baseline) clips are NOT pre-cached.
        # The editor's [▶ Sample] button synthesises its own cache entry on
        # first click and reuses it thereafter, so pre-caching baselines would
        # waste F0 quota on clips the editor never looks up.
        for style in info["styles"]:
            if style_filter and style != style_filter:
                continue
            if already_cached(index, voice, style):
                skipped += 1
            else:
                queue.append((voice, style))
    print(f"  {len(queue)} to generate, {skipped} already cached.")
    return queue

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-cache Azure TTS samples for all voices × styles.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print work queue without making API calls.")
    p.add_argument("--voice",   default=None, help="Process only this voice.")
    p.add_argument("--style",   default=None, help="Process only this style (requires --voice).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.style and not args.voice:
        sys.exit("[ERROR] --style requires --voice.")
    if args.voice and args.voice not in VOICES:
        sys.exit(f"[ERROR] Unknown voice: {args.voice!r}\nAvailable: {', '.join(VOICES)}")

    index = load_index()
    queue = build_work_queue(index, args.voice, args.style)
    if not queue:
        print("Nothing to do.")
        return

    if not args.dry_run:
        eta = len(queue) * 7.5 / 60
        print(f"  Estimated time: ~{eta:.0f} min  (avg 7.5 s/clip, resumable)")
        synth = load_synthesizer()
    else:
        synth = None

    done = failed = 0
    for i, (voice, style) in enumerate(queue, 1):
        label = style
        if not args.dry_run:
            print(f"  [{i}/{len(queue)}] {voice}  {label}", end="  ", flush=True)
        made_call = process_one(voice, style, index, synth, args.dry_run)
        if made_call:
            save_index(index)
            path = clip_path(voice, cache_key(voice, style))
            if path.exists():
                done += 1
                print("✓")
            else:
                failed += 1   # [FAIL] already printed with leading \n
            if i < len(queue):
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if not args.dry_run:
        print(f"\nDone. {done} new clip(s) → {CACHE_DIR}")
        if failed:
            print(f"  ⚠ {failed} clip(s) failed — re-run to retry.")


if __name__ == "__main__":
    main()
