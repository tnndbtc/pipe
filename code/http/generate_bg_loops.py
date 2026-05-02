#!/usr/bin/env python3
"""
generate_bg_loops.py — Pre-render 14-second looping background videos.

The geq sine-wave light filter has a period of exactly 14 seconds.
Pre-rendering one full period per background image means render_video.py
can stream-copy that clip to any duration instead of running geq on every
frame of every video.

Speedup: ~80 min background render → ~5 seconds (stream copy).

Usage:
    cd ~/data/code/pipe/code/http
    python3 generate_bg_loops.py            # render all missing loops
    python3 generate_bg_loops.py --force    # re-render even if loop exists

Output:
    ../../projects/resources/news/news_ai_loop.mp4
    ../../projects/resources/news/news_business_loop.mp4
    ... (one per news_*.png)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# script lives in  pipe/code/http/ — go up two levels to reach pipe root
RESOURCES   = Path(__file__).parent.parent.parent / "projects/resources/news"
RENDER_CODE = Path(__file__).parent   # render_video.py is in the same directory

FPS         = 24
LOOP_DUR    = 14        # exactly one sine period: sin(2π·T/14)
W, H        = 1280, 720
SAMPLE_RATE = 44100
CRF         = 18
PRESET      = "slow"   # match the quality of per-video renders

sys.path.insert(0, str(RENDER_CODE))
try:
    from render_video import _build_light_streak_vf  # type: ignore
except ImportError as e:
    print(f"[ERROR] Cannot import render_video: {e}", file=sys.stderr)
    sys.exit(1)

_SCALE_PAD = (
    f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
)


def render_loop(png: Path, out: Path) -> bool:
    """Render a 14-second looping .mp4 with geq streak effect for one PNG."""
    streak_vf = _build_light_streak_vf(str(png), mode="auto")
    vf = _SCALE_PAD + "," + streak_vf

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(png),
        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-vf", vf,
        "-r", str(FPS), "-pix_fmt", "yuv420p",
        "-t", str(LOOP_DUR),
        "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
        # Keyframe every second so stream-copy loops are always clean.
        "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
        "-c:a", "aac", "-shortest",
        str(out),
    ]

    print(f"  Rendering {png.name} → {out.name}  ({LOOP_DUR}s @ {FPS}fps) ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        size_mb = out.stat().st_size / 1_048_576
        print(f"  ✓  {out.name}  ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗  FAILED: {out.name}")
        print(r.stderr[-600:], file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if _loop.mp4 already exists")
    args = parser.parse_args()

    pngs = sorted(RESOURCES.glob("news_*.png"))
    if not pngs:
        print(f"[ERROR] No news_*.png found in {RESOURCES}", file=sys.stderr)
        sys.exit(1)

    print(f"Background images : {len(pngs)}")
    print(f"Resources dir     : {RESOURCES}")
    print(f"Loop duration     : {LOOP_DUR}s  ({LOOP_DUR * FPS} frames @ {FPS} fps)")
    print(f"Encode settings   : libx264  CRF={CRF}  preset={PRESET}")
    print()

    ok, skipped, failed = 0, 0, 0
    for png in pngs:
        loop = RESOURCES / (png.stem + "_loop.mp4")
        if loop.exists() and not args.force:
            size_mb = loop.stat().st_size / 1_048_576
            print(f"  ✓  {loop.name}  ({size_mb:.1f} MB) already exists — skipping")
            skipped += 1
            ok += 1
            continue
        if render_loop(png, loop):
            ok += 1
        else:
            failed += 1

    print()
    print(f"Done: {ok}/{len(pngs)} ready  ({skipped} skipped, {failed} failed)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
