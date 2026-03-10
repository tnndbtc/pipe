# =============================================================================
# plan_assets.py  --  AI Asset Pipeline Orchestrator
#
# Reads AssetManifest_draft.json and prints:
#   - Which scripts are RUNNABLE on this machine (with exact commands)
#   - Which scripts are BLOCKED (need a larger GPU)
#   - For each script: models used, VRAM estimate, recommended GPU
#
# Usage (from repo root):
#   python code/ai/plan_assets.py                        # hardware report (default)
#   python code/ai/plan_assets.py --run                  # execute runnable stages
#   python code/ai/plan_assets.py --only tts             # report / run one stage
#   python code/ai/plan_assets.py --ignore-vram          # skip VRAM budget checks
#   python code/ai/plan_assets.py --manifest <path>
#   python code/ai/plan_assets.py --output_dir <path>
# =============================================================================

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR        = Path(__file__).parent           # code/ai/
DEFAULT_MANIFEST   = _SCRIPT_DIR / "AssetManifest_draft.json"
DEFAULT_OUTPUT_DIR = Path("projects/the-pharaoh-who-defied-death/episodes/s01e01/assets")

W = 76   # report line width

# ---------------------------------------------------------------------------
# RUNNABLE STAGES
# All 10 gen_*.py scripts fit on RTX 4060 8 GB with the listed model choices.
#
# Fields:
#   name        -- stage identifier (used with --only)
#   script      -- filename in the same directory as this file
#   vram_gb     -- peak VRAM estimate (0 = CPU-only)
#   description -- one-line description
#   models      -- list of {name, hf_id, precision}  (primary first, fallbacks after)
#   rec_gpu     -- minimum GPU to run comfortably
#   check       -- fn(manifest) -> bool: True if manifest has data for this stage
#   manifest_fn -- fn(manifest) -> list[str]: asset/item IDs from manifest
# ---------------------------------------------------------------------------
STAGES = [
    {
        "name":        "character_images",
        "script":      "gen_character_images.py",
        "vram_gb":     7,
        "description": "Generate character portrait PNGs",
        "models": [
            {"name": "FLUX.1-schnell",
             "hf_id": "black-forest-labs/FLUX.1-schnell",
             "precision": "bitsandbytes int4  (~6-7 GB)"},
            {"name": "SDXL 1.0  (fallback if FLUX unavailable)",
             "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
             "precision": "FP16  (~4-5 GB)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("character_packs")),
        "manifest_fn": lambda m: [p["asset_id"] for p in m.get("character_packs", [])],
    },
    {
        "name":        "background_images",
        "script":      "gen_background_images.py",
        "vram_gb":     7,
        "description": "Generate static background plate images",
        "models": [
            {"name": "FLUX.1-schnell",
             "hf_id": "black-forest-labs/FLUX.1-schnell",
             "precision": "bitsandbytes int4  (~6-7 GB)"},
            {"name": "SDXL 1.0  (fallback if FLUX unavailable)",
             "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
             "precision": "FP16  (~4-5 GB)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: any(
            b.get("motion") is None for b in m.get("backgrounds", [])
        ),
        "manifest_fn": lambda m: [
            b["asset_id"] for b in m.get("backgrounds", []) if b.get("motion") is None
        ],
    },
    {
        "name":        "background_video",
        "script":      "gen_background_video.py",
        "vram_gb":     6,
        "description": "Generate animated background video clips",
        "models": [
            {"name": "CogVideoX-2b",
             "hf_id": "THUDM/CogVideoX-2b",
             "precision": "FP16 + enable_model_cpu_offload  (~5-6 GB)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: any(
            b.get("motion") and b["motion"].get("type") == "camera"
            for b in m.get("backgrounds", [])
        ),
        "manifest_fn": lambda m: [
            b["asset_id"] for b in m.get("backgrounds", [])
            if b.get("motion") and b["motion"].get("type") == "camera"
        ],
    },
    {
        "name":        "character_mattes",
        "script":      "gen_character_mattes.py",
        "vram_gb":     2,
        "description": "Remove portrait backgrounds -> RGBA PNGs",
        "models": [
            {"name": "RMBG-1.4",
             "hf_id": "briaai/RMBG-1.4",
             "precision": "FP32  (~175 MB, ~1-2 GB peak)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("character_packs")),
        "manifest_fn": lambda m: [p["asset_id"] for p in m.get("character_packs", [])],
    },
    {
        "name":        "upscale",
        "script":      "gen_upscale.py",
        "vram_gb":     2,
        "description": "2x upscale character RGBA PNGs",
        "models": [
            {"name": "Real-ESRGAN x4plus",
             "hf_id": "xinntao/Real-ESRGAN  [file: RealESRGAN_x4plus.pth]",
             "precision": "FP32  (~67 MB, ~2 GB peak with tile=512)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("character_packs")),
        "manifest_fn": lambda m: [p["asset_id"] for p in m.get("character_packs", [])],
    },
    {
        "name":        "character_animation",
        "script":      "gen_character_animation.py",
        "vram_gb":     6,
        "description": "Animate character portraits -> MP4",
        "models": [
            {"name": "AnimateDiff v3 motion adapter",
             "hf_id": "guoyww/animatediff-motion-adapter-v1-5-3",
             "precision": "FP16"},
            {"name": "Stable Diffusion v1.5  (base model)",
             "hf_id": "runwayml/stable-diffusion-v1-5",
             "precision": "FP16 + enable_model_cpu_offload  (~4-6 GB peak)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: any(
            p.get("motion") is not None for p in m.get("character_packs", [])
        ),
        "manifest_fn": lambda m: [
            p["asset_id"] for p in m.get("character_packs", []) if p.get("motion")
        ],
    },
    {
        "name":        "lipsync",
        "script":      "gen_lipsync.py",
        "vram_gb":     1,
        "description": "Drive lip-sync on character videos from VO audio",
        "models": [
            {"name": "Wav2Lip-GAN",
             "hf_id": "Rudrabha/Wav2Lip  [file: wav2lip_gan.pth]",
             "precision": "FP32  (~500 MB, ~1 GB peak)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("vo_items")),
        "manifest_fn": lambda m: [
            v["item_id"] for v in m.get("vo_items", []) if v.get("visual", True)
        ],
    },
    {
        "name":        "tts",
        "script":      "gen_tts.py",
        "vram_gb":     0,
        "description": "Generate voice-over WAVs  (CPU-only)",
        "models": [
            {"name": "Kokoro-82M",
             "hf_id": "hexgrad/Kokoro-82M",
             "precision": "CPU-only, FP32  (82 M params -- no GPU needed)"},
        ],
        "rec_gpu":     "CPU  (no GPU needed)",
        "check":       lambda m: bool(m.get("vo_items")),
        "manifest_fn": lambda m: [v["item_id"] for v in m.get("vo_items", [])],
    },
    {
        "name":        "sfx",
        "script":      "gen_sfx.py",
        "vram_gb":     4,
        "description": "Generate sound effect WAVs",
        "models": [
            {"name": "AudioGen small",
             "hf_id": "facebook/audiogen-small",
             "precision": "FP16  (auto-fallback from audiogen-medium on OOM, ~2-4 GB)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("sfx_items")),
        "manifest_fn": lambda m: [s["shot_id"] for s in m.get("sfx_items", [])],
    },
    {
        "name":        "music",
        "script":      "gen_music.py",
        "vram_gb":     6,
        "description": "Generate background music WAVs",
        "models": [
            {"name": "MusicGen small",
             "hf_id": "facebook/musicgen-small",
             "precision": "FP16  (300 M params, ~4-6 GB peak)"},
        ],
        "rec_gpu":     "RTX 4060 8 GB",
        "check":       lambda m: bool(m.get("music_items")),
        "manifest_fn": lambda m: [s["shot_id"] for s in m.get("music_items", [])],
    },
]

# ---------------------------------------------------------------------------
# BLOCKED SCRIPTS
# placeholder_*.py files -- quality-upgrade models that do NOT fit on 8 GB VRAM.
# Run these on a larger GPU for higher-quality output.
# ---------------------------------------------------------------------------
BLOCKED_SCRIPTS = [
    {
        "script":         "placeholder_for_background_video.py",
        "description":    "Higher-quality animated BGs  (upgrade from CogVideoX-2b)",
        "upgrades":       "gen_background_video.py",
        "models": [
            {"name": "LTX-Video 0.9.8-13B distilled",
             "hf_id": "Lightricks/LTX-Video-0.9.8-13B-distilled",
             "precision": "FP8"},
        ],
        "vram_needed_gb": 24,
        "vram_note":      "13 B params x 1 byte (FP8) = 13 GB weights + ~10 GB peak activations",
        "rec_gpu":        "RTX 3090 24 GB  or  RTX 4090 24 GB",
        "manifest_fn": lambda m: [
            b["asset_id"] for b in m.get("backgrounds", [])
            if b.get("motion") and b["motion"].get("type") == "camera"
        ],
    },
    {
        "script":         "placeholder_for_lipsync.py",
        "description":    "Higher-quality lip-sync  (upgrade from Wav2Lip-GAN)",
        "upgrades":       "gen_lipsync.py",
        "models": [
            {"name": "LatentSync 1.5",
             "hf_id": "ByteDance/LatentSync-1.5",
             "precision": "FP16"},
        ],
        "vram_needed_gb": 16,
        "vram_note":      "UNet 4-6 GB + Whisper 1-2 GB + denoising activations 6-8 GB",
        "rec_gpu":        "RTX 4080 16 GB",
        "manifest_fn": lambda m: [
            v["item_id"] for v in m.get("vo_items", []) if v.get("visual", True)
        ],
    },
    {
        "script":         "placeholder_for_sfx_medium.py",
        "description":    "Higher-quality SFX  (upgrade from AudioGen small)",
        "upgrades":       "gen_sfx.py",
        "models": [
            {"name": "AudioGen medium",
             "hf_id": "facebook/audiogen-medium",
             "precision": "FP16  (1.5 B params)"},
        ],
        "vram_needed_gb": 12,
        "vram_note":      "model weights ~6 GB + KV-cache / activations 4-6 GB",
        "rec_gpu":        "RTX 4080 16 GB",
        "manifest_fn":    lambda m: [s["shot_id"] for s in m.get("sfx_items", [])],
    },
    {
        "script":         "placeholder_for_music_medium.py",
        "description":    "Higher-quality music  (upgrade from MusicGen small)",
        "upgrades":       "gen_music.py",
        "models": [
            {"name": "MusicGen medium",
             "hf_id": "facebook/musicgen-medium",
             "precision": "FP16  (1.5 B params, ~10-12 GB)"},
            {"name": "MusicGen large",
             "hf_id": "facebook/musicgen-large",
             "precision": "FP16  (3.3 B params, ~20+ GB)"},
        ],
        "vram_needed_gb": 12,
        "vram_note":      "medium ~10-12 GB  |  large ~20+ GB",
        "rec_gpu":        "RTX 4080 16 GB  (medium)  /  RTX 3090 24 GB  (large)",
        "manifest_fn":    lambda m: [s["shot_id"] for s in m.get("music_items", [])],
    },
]

STAGE_NAMES = [s["name"] for s in STAGES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
    except Exception:
        pass
    return ""


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_relative(path: Path) -> str:
    """Return path relative to cwd (forward slashes). Falls back to absolute."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_command_str(script_name: str, manifest_path: Path, output_dir: Path) -> str:
    """Human-readable command string using relative paths."""
    script = _SCRIPT_DIR / script_name
    return (
        f"python {fmt_relative(script)}"
        f" --manifest {fmt_relative(manifest_path)}"
        f" --output_dir {str(output_dir).replace(chr(92), '/')}"
    )


def build_subprocess_argv(script_name: str, manifest_path: Path, output_dir: Path) -> list:
    """Argv list for subprocess.run (absolute paths)."""
    return [
        sys.executable,
        str(_SCRIPT_DIR / script_name),
        "--manifest", str(manifest_path),
        "--output_dir", str(output_dir),
    ]


def wrap_ids(prefix: str, ids: list, cont_indent: int) -> str:
    """Wrap a list of IDs after prefix; continuation lines indented by cont_indent spaces."""
    if not ids:
        return prefix + "(none)"
    lines = []
    current = prefix + ids[0]
    for item in ids[1:]:
        candidate = current + ", " + item
        if len(candidate) > W:
            lines.append(current + ",")
            current = " " * cont_indent + item
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def rule(char="=") -> str:
    return char * W


# ---------------------------------------------------------------------------
# Hardware compatibility report
# ---------------------------------------------------------------------------
def print_hardware_report(
    manifest: dict,
    manifest_path: Path,
    output_dir: Path,
    vram_gb: float,
    gpu_name: str,
    only_names: set = None,
):
    project_id = manifest.get("project_id", "unknown")

    if gpu_name and vram_gb > 0:
        gpu_str = f"{gpu_name}  ({vram_gb:.1f} GB VRAM detected)"
    elif vram_gb > 0:
        gpu_str = f"{vram_gb:.1f} GB VRAM detected"
    else:
        gpu_str = "not detected  (torch.cuda unavailable -- install torch with CUDA)"

    # Asset counts
    chars        = manifest.get("character_packs", [])
    bgs          = manifest.get("backgrounds", [])
    static_bgs   = [b for b in bgs if not b.get("motion")]
    animated_bgs = [b for b in bgs if b.get("motion") and b["motion"].get("type") == "camera"]
    vo_items     = manifest.get("vo_items", [])
    sfx_items    = manifest.get("sfx_items", [])
    music_items  = manifest.get("music_items", [])

    print(rule())
    print("HARDWARE COMPATIBILITY REPORT")
    print(rule())
    print(f"  Project   : {project_id}")
    print(f"  Manifest  : {fmt_relative(manifest_path)}")
    print(f"  GPU       : {gpu_str}")
    print(f"  Output    : {str(output_dir).replace(chr(92), '/')}")
    print()
    print("  Assets in manifest:")

    char_ids = [p["asset_id"] for p in chars]
    print(wrap_ids(f"    Characters  : {len(chars)}  ", char_ids, cont_indent=20))

    print(f"    Backgrounds : {len(bgs)}"
          f"  ({len(static_bgs)} static image{'s' if len(static_bgs) != 1 else ''},"
          f"  {len(animated_bgs)} animated video{'s' if len(animated_bgs) != 1 else ''})")

    print(f"    VO lines    : {len(vo_items)}")

    sfx_note   = str(len(sfx_items))   if sfx_items   else "0  (not in manifest -- gen_sfx.py uses hardcoded defaults)"
    music_note = str(len(music_items)) if music_items else "0  (not in manifest -- gen_music.py uses hardcoded defaults)"
    print(f"    SFX items   : {sfx_note}")
    print(f"    Music items : {music_note}")

    # Filter to --only if requested
    active_stages = STAGES if not only_names else [s for s in STAGES if s["name"] in only_names]
    n_run     = len(active_stages)
    n_blocked = len(BLOCKED_SCRIPTS)

    # -----------------------------------------------------------------------
    # RUNNABLE
    # -----------------------------------------------------------------------
    print()
    print(rule())
    print(f"RUNNABLE ON THIS MACHINE  ({n_run} / {n_run + n_blocked} scripts)")
    print(rule())
    print("  All gen_*.py scripts fit on RTX 4060 8 GB with the model choices below.")

    for i, stage in enumerate(active_stages, start=1):
        has_data  = stage["check"](manifest)
        asset_ids = stage["manifest_fn"](manifest)
        vram_note = f"~{stage['vram_gb']} GB" if stage["vram_gb"] > 0 else "0 GB (CPU-only)"

        print()
        header = f"  [{i:02d}/{n_run}]  {stage['script']}"
        right  = f"VRAM {vram_note}"
        print(f"{header}{right:>{W - len(header)}}")
        print(f"          {stage['description']}")
        print()

        # Assets
        if has_data and asset_ids:
            print(wrap_ids("          Assets    : ", asset_ids, cont_indent=22))
        elif has_data:
            print("          Assets    : (manifest matched but returned no IDs)")
        else:
            print("          Assets    : (not in manifest -- uses hardcoded defaults when run standalone)")

        # Models
        for j, mdl in enumerate(stage["models"]):
            label = "Model      :" if j == 0 else "            "
            print(f"          {label} {mdl['name']}")
            print(f"                      HF       : {mdl['hf_id']}")
            print(f"                      Precision: {mdl['precision']}")

        ok_tag = "  [OK -- current GPU]" if vram_gb >= stage["vram_gb"] or stage["vram_gb"] == 0 else "  [verify -- VRAM marginal]"
        print(f"          Rec GPU    : {stage['rec_gpu']}{ok_tag}")
        print(f"          Command    : {build_command_str(stage['script'], manifest_path, output_dir)}")

    # -----------------------------------------------------------------------
    # BLOCKED
    # -----------------------------------------------------------------------
    print()
    print(rule())
    print(f"BLOCKED ON THIS MACHINE  ({n_blocked} / {n_run + n_blocked} scripts)  --  need a larger GPU")
    print(rule())
    print("  These are quality-upgrade models. The placeholder_*.py scripts print")
    print("  requirements info only; run the gen_*.py equivalent on the larger GPU.")

    for i, bs in enumerate(BLOCKED_SCRIPTS, start=1):
        asset_ids = bs["manifest_fn"](manifest)

        print()
        header = f"  [{i:02d}/{n_blocked}]  {bs['script']}"
        right  = f"NEED ~{bs['vram_needed_gb']} GB  [BLOCKED]"
        print(f"{header}{right:>{W - len(header)}}")
        print(f"          {bs['description']}")
        print(f"          Upgrades   : {bs['upgrades']}")
        print()

        # Assets
        if asset_ids:
            print(wrap_ids("          Assets    : ", asset_ids, cont_indent=22))
        else:
            print("          Assets    : (no matching data in manifest)")

        # Models
        for j, mdl in enumerate(bs["models"]):
            label = "Model      :" if j == 0 else "            "
            print(f"          {label} {mdl['name']}")
            print(f"                      HF       : {mdl['hf_id']}")
            print(f"                      Precision: {mdl['precision']}")

        print(f"          VRAM       : ~{bs['vram_needed_gb']} GB  ({bs['vram_note']})")
        print(f"          Rec GPU    : {bs['rec_gpu']}")
        # Show the command you'd run on a bigger machine (using the real gen_ script)
        real_script = bs["upgrades"]
        print(f"          Command    : {build_command_str(real_script, manifest_path, output_dir)}")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print()
    print(rule())
    print("SUMMARY")
    print(rule())
    gpu_label = gpu_name if gpu_name else "current GPU"
    print(f"  Runnable on {gpu_label:<32}: {n_run:2d} scripts")
    print(f"  Blocked (need larger GPU){'':<19}: {n_blocked:2d} scripts")
    print()
    print("  GPU upgrade path:")
    print("    RTX 4080  16 GB  ->  unlocks: LatentSync 1.5 (lipsync),")
    print("                                  AudioGen medium (SFX)")
    print("    RTX 3090 / RTX 4090  24 GB  ->  unlocks: LTX-Video 13B (bg video),")
    print("                                             MusicGen large (music)")
    print()
    print("  Run pipeline:  python code/ai/plan_assets.py --run")
    print(rule())


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print a hardware compatibility report for the AI asset pipeline,\n"
            "then optionally execute the runnable stages.\n"
            "Default (no flags): print report only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"Path to AssetManifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Asset output directory passed to each gen_*.py (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Execute runnable stages sequentially after printing the report.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="STAGE",
        choices=STAGE_NAMES,
        help="Limit to these stage(s): " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument(
        "--ignore-vram", action="store_true",
        help="Skip VRAM budget checks in --run mode (attempt all stages).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    manifest_path = args.manifest
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    vram_gb  = detect_vram_gb()
    gpu_name = detect_gpu_name()
    only_names = set(args.only) if args.only else None

    # Always print the hardware report
    print_hardware_report(manifest, manifest_path, args.output_dir, vram_gb, gpu_name, only_names)

    if not args.run:
        return

    # -----------------------------------------------------------------------
    # Execute runnable stages
    # -----------------------------------------------------------------------
    active_stages = STAGES if not only_names else [s for s in STAGES if s["name"] in only_names]
    plan = []
    for stage in active_stages:
        if not stage["check"](manifest):
            print(f"\n[SKIP] {stage['script']}  -- no matching data in manifest")
            continue
        if not args.ignore_vram and vram_gb > 0 and stage["vram_gb"] > vram_gb:
            print(
                f"\n[SKIP] {stage['script']}  -- VRAM insufficient "
                f"({vram_gb:.1f} GB available, {stage['vram_gb']} GB required). "
                f"Pass --ignore-vram to override."
            )
            continue
        plan.append(stage)

    if not plan:
        print("\nNothing to run.")
        return

    print()
    print(rule())
    print(f"RUNNING  ({len(plan)} stage{'s' if len(plan) != 1 else ''})")
    print(rule())

    results = []
    for i, stage in enumerate(plan, start=1):
        argv = build_subprocess_argv(stage["script"], manifest_path, args.output_dir)
        print(f"\n[{i}/{len(plan)}] {stage['script']}")
        print(f"CMD: {build_command_str(stage['script'], manifest_path, args.output_dir)}\n")
        try:
            subprocess.run(argv, check=True)
            results.append({"stage": stage["name"], "status": "success"})
        except subprocess.CalledProcessError as exc:
            print(f"\n[ERROR] {stage['script']} exited with code {exc.returncode}. Aborting.")
            results.append({"stage": stage["name"], "status": "failed"})
            break
        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
            results.append({"stage": stage["name"], "status": "interrupted"})
            break

    print()
    print(rule())
    print("RUN RESULTS")
    print(rule())
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        print(f"  [{tag:^12}]  {r['stage']}")
    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n  {ok}/{len(plan)} stages completed.")
    print(rule())


if __name__ == "__main__":
    main()
