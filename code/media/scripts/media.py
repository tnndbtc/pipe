#!/usr/bin/env python3
"""
media — local asset library CLI.

Usage:
  media ask --project SLUG --episode EP_ID "natural language query"

Subcommands:
  ask   — describe in plain English what media you want for a shot;
           claude -p reasons over the episode's AssetManifest, checks
           existing batches, and triggers a CC-only search on the media server.
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: media <subcommand> [args]\nSubcommands: ask")
        sys.exit(2)

    subcommand = sys.argv[1]
    args       = sys.argv[1:]

    # ── ask subcommand ────────────────────────────────────────────────────────
    if subcommand == "ask":
        import argparse
        import shutil
        import subprocess

        ap = argparse.ArgumentParser(
            prog="media ask",
            description="Search for media using natural language via claude -p.",
        )
        ap.add_argument("--project",  required=True, help="project slug (e.g. pompeii-vesuvius-eruption)")
        ap.add_argument("--episode",  required=True, help="episode id (e.g. s01e01)")
        ap.add_argument("query",      nargs="+",     help="natural language search query")
        a = ap.parse_args(args[1:])
        query = " ".join(a.query)

        # Import build_prompt from sibling script
        sys.path.insert(0, str(Path(__file__).parent))
        from media_ask import build_prompt

        if not shutil.which("claude"):
            sys.exit(
                "Error: 'claude' not found in PATH.\n"
                "Install Claude Code: https://claude.ai/code"
            )

        prompt = build_prompt(query, a.project, a.episode)

        result = subprocess.run(
            ["claude", "-p", "--allowedTools", "Bash", "--output-format", "text"],
            input=prompt,
            capture_output=False,
            text=True,
        )
        sys.exit(result.returncode)

    # ── unknown subcommand ────────────────────────────────────────────────────
    else:
        print(f"Unknown subcommand: {subcommand!r}")
        print("Usage: media <subcommand> [args]\nSubcommands: ask")
        sys.exit(2)


if __name__ == "__main__":
    main()
