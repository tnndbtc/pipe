"""
export_youtube_dubbed.py
------------------------
Extract the mixed audio track from a locale render as a standalone AAC file
suitable for upload to YouTube Studio as a dubbed audio track.

YouTube's "Dubbed audio" feature requires a separate audio file upload —
it does NOT read multiple audio tracks from an uploaded MP4.  This script
produces that file from the locale render that the pipeline already created.

Output:  <episode_dir>/renders/<locale>/youtube_dubbed.aac
Format:  AAC, 192 kbps, 48 kHz, stereo  (matches render_video.py settings)
Content: Full mixed audio — Chinese VO + music + SFX, loudness-normalised to
         -16 LUFS exactly as rendered.  YouTube replaces the primary audio
         track for viewers who select this language.

Usage:
    python code/http/export_youtube_dubbed.py <episode_dir> [locale]

    episode_dir  Path to the episode directory,
                 e.g.  projects/pharaoh/episodes/s01e02
    locale       BCP-47 locale code (default: zh-Hans)

Examples:
    python code/http/export_youtube_dubbed.py projects/pharaoh/episodes/s01e02
    python code/http/export_youtube_dubbed.py projects/pharaoh/episodes/s01e02 zh-Hans
"""

import argparse
import json
import subprocess
from pathlib import Path


def export_dubbed_audio(episode_dir: str, locale: str = "zh-Hans") -> Path:
    """
    Extract the mixed audio track from a locale render and write it as a
    standalone AAC file ready for YouTube Studio dubbed-audio upload.

    Args:
        episode_dir: Path to the episode directory
                     e.g. "projects/pharaoh/episodes/s01e02"
        locale:      BCP-47 locale code, e.g. "zh-Hans"

    Returns:
        Path to the output AAC file.

    Raises:
        SystemExit: if the source render MP4 is missing or ffmpeg fails.
    """
    source = Path(episode_dir) / "renders" / locale / "output.mp4"
    output = Path(episode_dir) / "renders" / locale / "youtube_dubbed.aac"

    # Pre-flight: verify source render exists before starting any work
    if not source.exists():
        raise SystemExit(
            f"[ERROR] Source render not found: {source}\n"
            f"        Run the render pipeline for locale '{locale}' first."
        )

    print(f"▶  Exporting dubbed audio  locale={locale}")
    print(f"   Source : {source}")
    print(f"   Output : {output}")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(source),
            "-map", "0:a:0",        # first (only) audio track from the render
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",         # 48 kHz — matches render_video.py
            str(output),
        ],
        check=True,
    )

    # Verify and report the output
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)
    duration_sec = float(info["format"]["duration"])
    size_mb = int(info["format"]["size"]) / (1024 * 1024)

    print(f"   Duration : {duration_sec:.1f}s")
    print(f"   Size     : {size_mb:.1f} MB")
    print(f"✓  youtube_dubbed.aac ready — upload via YouTube Studio > Dubbing")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="export_youtube_dubbed.py",
        description=(
            "Extract the mixed audio track from a locale render as a standalone\n"
            "AAC file for upload to YouTube Studio as a dubbed audio track."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python code/http/export_youtube_dubbed.py projects/pharaoh/episodes/s01e02\n"
            "  python code/http/export_youtube_dubbed.py projects/pharaoh/episodes/s01e02 zh-Hans"
        ),
    )
    parser.add_argument(
        "episode_dir",
        help="path to the episode directory, e.g. projects/pharaoh/episodes/s01e02",
    )
    parser.add_argument(
        "locale",
        nargs="?",
        default="zh-Hans",
        help="BCP-47 locale code (default: zh-Hans)",
    )
    args = parser.parse_args()
    export_dubbed_audio(args.episode_dir, args.locale)


if __name__ == "__main__":
    main()
