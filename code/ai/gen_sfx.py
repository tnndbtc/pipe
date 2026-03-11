# =============================================================================
# gen_sfx.py
# Generate short sound-effect audio clips from text descriptions.
# STATUS: VALIDATED
# Supports two backends selected via --backend:
#   audiogen       -- Meta AudioGen (default, MIT licence, 4-8 GB VRAM)
#   stable_audio_2 -- Stability AI Stable Audio 2.0 (>=16 GB VRAM recommended)
# =============================================================================
#
# requirements.txt (pip install before running):
#   audiocraft>=1.3.0        # Meta AudioCraft -- includes AudioGen
#   torch>=2.1.0
#   torchaudio>=2.1.0
#   soundfile>=0.12.0
#   huggingface_hub>=0.21.0
#   diffusers>=0.30.0        # Stable Audio 2.0 backend only
#   transformers>=4.40.0     # Stable Audio 2.0 backend only
#
# ---------------------------------------------------------------------------
# Hardware targets:
#   audiogen       -- NVIDIA RTX 4060 8 GB VRAM (medium), 4 GB (small)
#   stable_audio_2 -- REQUIRES RTX 4090 OR BETTER (>=16 GB VRAM)
#                    RTX 4060 (8 GB) is insufficient for this backend.
#                    Uses torch.float16 on CUDA automatically.
# ---------------------------------------------------------------------------
# Memory-saving techniques (AudioGen):
#   AudioGen medium (~8 GB VRAM) is used when available; if it OOMs, the
#   script catches the exception and reloads AudioGen small (~4 GB).
#
#   Key optimisations:
#     - Model is kept loaded across all SFX jobs and only freed at the end.
#     - Each job is generated as a single batch item to avoid batch-size
#       memory overhead.
#     - torch.cuda.empty_cache() + gc.collect() after each generation.
#     - If the GPU is insufficient, set DEVICE = "cpu" via --device flag;
#       AudioGen runs on CPU at roughly 0.3x real-time.
#
# NOTE: AudioCraft models are downloaded automatically from HuggingFace
#   (facebook/audiogen-medium or facebook/audiogen-small, MIT licence).
#   Stable Audio 2.0 requires a HuggingFace account with model access granted
#   at https://huggingface.co/stabilityai/stable-audio-2
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import re
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS -- fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_sfx"

SFX_JOBS = [
    {
        "shot_id": "s01-sh01", "duration": 6.0,
        "tags": [
            "stone scraping in an ancient stone chamber",
            "distant desert wind howling",
            "oil lamp flame flickering",
        ],
    },
    {
        "shot_id": "s01-sh02", "duration": 10.0,
        "tags": ["oil lamp flame flickering softly"],
    },
    {
        "shot_id": "s01-sh03", "duration": 8.0,
        "tags": [
            "multiple oil lamps flickering violently",
            "deep stone hum resonance",
        ],
    },
    {
        "shot_id": "s02-sh01", "duration": 5.0,
        "tags": [
            "papyrus scroll rustling quietly",
            "distant temple ritual chanting",
        ],
    },
    {
        "shot_id": "s02-sh02", "duration": 8.0,
        "tags": ["writing stylus stopping mid-stroke on papyrus"],
    },
    {
        "shot_id": "s03-sh01", "duration": 6.0,
        "tags": [
            "heavy stone blocks being dragged across sand",
            "dry desert wind",
            "distant shouted orders echoing",
        ],
    },
    {
        "shot_id": "s03-sh02", "duration": 7.0,
        "tags": ["footsteps fading away on stone and sand"],
    },
    {
        "shot_id": "s04-sh01", "duration": 5.0,
        "tags": [
            "deep stone resonance booming in a large underground chamber",
            "distant low chanting beginning",
        ],
    },
    {
        "shot_id": "s04-sh02", "duration": 8.0,
        "tags": [
            "bronze rods humming with supernatural energy",
            "ritual chanting building",
            "crackling blue energy ignition",
        ],
    },
    {
        "shot_id": "s04-sh03", "duration": 6.0,
        "tags": [
            "man screaming in terror",
            "sudden total silence",
        ],
    },
    {
        "shot_id": "s04-sh04", "duration": 6.0,
        "tags": [
            "liquid stone surface undulating like black water",
            "whispers building from inside stone",
        ],
    },
    {
        "shot_id": "s04-sh06", "duration": 6.0,
        "tags": ["single massive low-frequency impact cut to silence"],
    },
    {
        "shot_id": "s05-sh01", "duration": 6.0,
        "tags": ["oil lamp crackling in a quiet stone room"],
    },
    {
        "shot_id": "s05-sh02", "duration": 10.0,
        "tags": [
            "oil lamp flickering",
            "deep silence",
        ],
    },
]

AUDIOGEN_MEDIUM = "facebook/audiogen-medium"
AUDIOGEN_SMALL  = "facebook/audiogen-small"
STABLE_AUDIO_2  = "stabilityai/stable-audio-open-1.0"


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """
    Load SFX job list from AssetManifest JSON (section: sfx_items).
    Returns None if the sfx_items section is absent (caller should use hardcoded SFX_JOBS).
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sfx_items = manifest.get("sfx_items")
    if sfx_items is None:
        return None  # section absent -- caller falls back to hardcoded
    if asset_id_filter:
        sfx_items = [j for j in sfx_items if j.get("asset_id") == asset_id_filter or j.get("shot_id") == asset_id_filter]
    return sfx_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tag_to_slug(tag: str) -> str:
    """Convert a descriptive tag into a safe filename slug (max 50 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return slug[:50]


def build_output_filename(shot_id: str, tag: str) -> str:
    """sfx_{shot_id}_{tag_slug}.wav  e.g. sfx_s01-sh01_stone-scraping.wav"""
    return f"sfx_{shot_id}_{tag_to_slug(tag)}.wav"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SFX WAV files using AudioGen or Stable Audio 2.0."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu).",
    )
    parser.add_argument(
        "--backend",
        choices=["audiogen", "stable_audio_2"],
        default="audiogen",
        help="TTS backend. 'audiogen' (default) or 'stable_audio_2'.",
    )
    # AudioGen-specific
    parser.add_argument(
        "--model",
        choices=["medium", "small", "auto"],
        default="auto",
        help="AudioGen model size. 'auto' tries medium, falls back to small. (audiogen backend only)",
    )
    # Stable Audio 2.0-specific
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Default clip duration in seconds. (stable_audio_2 backend; per-job duration overrides this)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion inference steps. (stable_audio_2 backend only)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale. (stable_audio_2 backend only)",
    )
    # Shared
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. Uses sfx_items section if present; "
             "otherwise falls back to hardcoded SFX_JOBS.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only the sfx job matching this shot_id (requires --manifest).",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None, metavar="PROMPT",
        help=(
            "One or more text prompts to generate directly, bypassing the manifest "
            "and hardcoded job list. Each prompt produces one WAV file. "
            "Example: --prompts \"volcanic eruption\" \"distant thunder\""
        ),
    )
    parser.add_argument(
        "--prompts-duration", type=float, default=10.0, dest="prompts_duration",
        help="Clip duration in seconds for --prompts jobs. (default: 10.0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_audiogen(preference: str, device: str):
    """Load AudioGen medium (only publicly available model)."""
    from audiocraft.models import AudioGen

    print(f"[MODEL] Loading AudioGen medium...")
    model = AudioGen.get_pretrained(AUDIOGEN_MEDIUM, device=device)
    return model


def load_stable_audio_model(device: str):
    """Load Stable Audio 2.0 pipeline. Requires >=16 GB VRAM on CUDA."""
    from diffusers import StableAudioPipeline  # noqa: PLC0415

    dtype = torch.float16 if "cuda" in device else torch.float32
    print(f"[MODEL] Loading Stable Audio 2.0 ({STABLE_AUDIO_2}) on {device} dtype={dtype}...")
    try:
        pipe = StableAudioPipeline.from_pretrained(STABLE_AUDIO_2, torch_dtype=dtype)
        pipe = pipe.to(device)
        print("[MODEL] Stable Audio 2.0 loaded.")
        return pipe
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        raise RuntimeError(
            f"Stable Audio 2.0 requires >=16 GB VRAM. Original error: {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not load {STABLE_AUDIO_2}. "
            "If the model is gated, log in first: huggingface-cli login\n"
            f"Original error: {exc}"
        ) from exc


def locale_from_manifest_path(path: str) -> str:
    """Extract locale from manifest filename.
    'AssetManifest_draft.zh-Hans.json' -> 'zh-Hans'
    'AssetManifest_draft.json'          -> 'en'
    """
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


# ---------------------------------------------------------------------------
# Shared iteration helper
# ---------------------------------------------------------------------------
def _iter_sfx_tags(sfx_jobs, default_duration: float = 10.0):
    """
    Yield (shot_id, tag, duration_sec) for every clip to generate.
    Handles both formats:
      hardcoded SFX_JOBS -- {shot_id, duration, tags: [...]}
      manifest sfx_items -- {shot_id, tag, duration_sec}
    """
    for job in sfx_jobs:
        shot_id  = job.get("asset_id") or job.get("shot_id") or "unknown"
        duration = job.get("duration") or job.get("duration_sec") or default_duration
        tags     = job.get("tags") or ([job["tag"]] if "tag" in job else None) or ([job["ai_prompt"]] if "ai_prompt" in job else [])
        for tag in tags:
            yield shot_id, tag, float(duration)


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------
def run_audiogen(sfx_jobs: list, out_dir: Path, model, args) -> list[dict]:
    """Generate SFX clips with AudioGen. Returns list of result dicts."""
    import soundfile as sf  # noqa: PLC0415

    all_tags  = list(_iter_sfx_tags(sfx_jobs, default_duration=args.duration))
    total     = len(all_tags)
    results   = []
    model_id  = AUDIOGEN_MEDIUM  # label only; actual loaded id varies

    for counter, (shot_id, tag, duration) in enumerate(all_tags, start=1):
        filename = build_output_filename(shot_id, tag)
        out_path = out_dir / filename
        print(f"\n[{counter}/{total}] {shot_id} -- \"{tag}\"")

        if out_path.exists():
            print(f"  [SKIP] {filename} already exists")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": out_path.stat().st_size, "duration_sec": duration,
                "model": model_id, "status": "skipped",
            })
            continue

        try:
            model.set_generation_params(duration=duration)
            wav      = model.generate([tag])   # [1, channels, samples]
            audio_np = wav[0, 0].cpu().numpy() # mono: channel 0
            sf.write(str(out_path), audio_np, model.sample_rate, subtype="PCM_16")
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({duration}s, {size:,} bytes)")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": size, "duration_sec": duration,
                "model": model_id, "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {filename}: {exc}")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": 0, "duration_sec": duration,
                "model": model_id, "status": "failed", "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    return results


def run_stable_audio_2(sfx_jobs: list, out_dir: Path, pipe, args) -> list[dict]:
    """Generate SFX clips with Stable Audio 2.0. Returns list of result dicts."""
    import soundfile as sf  # noqa: PLC0415

    sample_rate = pipe.vae.config.sampling_rate
    all_tags    = list(_iter_sfx_tags(sfx_jobs, default_duration=args.duration))
    total       = len(all_tags)
    results     = []

    for counter, (shot_id, tag, duration) in enumerate(all_tags, start=1):
        filename = build_output_filename(shot_id, tag)
        out_path = out_dir / filename
        print(f"\n[{counter}/{total}] {shot_id} -- \"{tag}\"")
        print(f"  steps={args.steps}  guidance={args.guidance}  duration={duration}s")

        if out_path.exists():
            print(f"  [SKIP] {filename} already exists")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": out_path.stat().st_size, "duration_sec": duration,
                "model": STABLE_AUDIO_2, "status": "skipped",
            })
            continue

        try:
            audio = pipe(
                prompt=tag,
                audio_length_in_s=duration,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            ).audios[0]                          # numpy [channels, samples]
            sf.write(str(out_path), audio.T, samplerate=sample_rate)
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({duration}s, {size:,} bytes)")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": size, "duration_sec": duration,
                "model": STABLE_AUDIO_2, "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {filename}: {exc}")
            results.append({
                "shot_id": shot_id, "tag": tag,
                "output": str(out_path), "output_path": str(out_path),
                "size_bytes": 0, "duration_sec": duration,
                "model": STABLE_AUDIO_2, "status": "failed", "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    locale  = locale_from_manifest_path(args.manifest) if args.manifest else 'en'
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine job list: --prompts > manifest sfx_items > hardcoded
    sfx_jobs = SFX_JOBS
    if args.prompts:
        sfx_jobs = [
            {"shot_id": "oneoff", "duration": args.prompts_duration, "tags": args.prompts}
        ]
        print(f"[INFO] --prompts mode: {len(args.prompts)} prompt(s), duration={args.prompts_duration}s")
    elif args.manifest:
        manifest_jobs = load_from_manifest(args.manifest, args.asset_id)
        if manifest_jobs is None:
            print("[INFO] No sfx_items section in manifest -- using hardcoded SFX_JOBS.")
        elif not manifest_jobs:
            print("[WARN] sfx_items section is empty (or no match for --asset-id). Nothing to do.")
            return
        else:
            sfx_jobs = manifest_jobs

    torch.manual_seed(args.seed)

    print(f"[BACKEND] {args.backend}  device={args.device}")

    if args.backend == "audiogen":
        model   = load_audiogen(args.model, args.device)
        results = run_audiogen(sfx_jobs, out_dir, model, args)

    elif args.backend == "stable_audio_2":
        try:
            pipe = load_stable_audio_model(args.device)
        except RuntimeError as exc:
            raise SystemExit(f"[ERROR] {exc}") from exc
        results = run_stable_audio_2(sfx_jobs, out_dir, pipe, args)

    else:
        raise SystemExit(f"[ERROR] Unknown backend: {args.backend!r}")

    # Write results manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    total       = len(results)
    ok_count    = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print("\n" + "=" * 60)
    print(f"SUMMARY -- gen_sfx ({args.backend})")
    print("=" * 60)
    for r in results:
        label = "OK" if r["status"] == "success" else r["status"].upper()
        print(f"  [{label}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
