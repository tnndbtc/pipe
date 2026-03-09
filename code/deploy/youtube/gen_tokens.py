#!/usr/bin/env python3
"""
gen_tokens.py — Generate and verify OAuth tokens for YouTube multi-channel upload.

Run once per environment to create per-channel token files.

Usage:
    python code/deploy/youtube/gen_tokens.py

Reads:
    projects/client_secret.json

Writes:
    ~/.config/pipe/token_en.json
    ~/.config/pipe/token_zh.json
    ~/.config/pipe/youtube_profiles.json
"""

import json
import sys
import urllib.parse
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ── Constants ────────────────────────────────────────────────────────────────

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for captions.insert
]

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
CLIENT_SECRET = PROJECT_ROOT / "projects" / "client_secret.json"
CONFIG_DIR    = Path.home() / ".config" / "pipe"
PROFILES_PATH = CONFIG_DIR / "youtube_profiles.json"

# Profiles to generate in order
PROFILES = ["en", "zh"]

# ── OAuth helpers ─────────────────────────────────────────────────────────────

def run_oauth(profile: str) -> Credentials:
    """
    Headless OAuth flow — prints a URL for the operator to visit in any browser.

    After granting access, Google redirects the browser to http://localhost
    (which fails to load — nothing is listening there).  The operator copies
    the full redirect URL from the address bar and pastes it back here.
    We extract the authorization code from that URL.

    Uses redirect_uri=http://localhost (registered for 'installed' type apps).
    Forces account/channel picker via prompt=select_account.
    """
    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET),
        scopes=SCOPES,
    )
    flow.redirect_uri = "http://localhost"

    auth_url, _ = flow.authorization_url(
        prompt="select_account",   # force account/channel chooser every time
        access_type="offline",     # request refresh token
    )

    print(f"\n{'='*60}")
    print(f"  Profile : {profile!r}")
    print(f"{'='*60}")
    print(f"\n  Step 1 — Open this URL in your browser:\n")
    print(f"  {auth_url}\n")
    print(f"  Step 2 — Switch YouTube to the {profile.upper()} channel,")
    print(f"           then click Allow.")
    print(f"\n  Step 3 — Your browser will redirect to a page that fails to load.")
    print(f"           Copy the FULL URL from the address bar and paste it below.")
    print(f"           (It starts with http://localhost/?code=...)\n")

    pasted = input("  Paste redirect URL: ").strip()

    # Accept either the full redirect URL or just the raw code
    if pasted.startswith("http"):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
        code = params.get("code", [None])[0]
        if not code:
            print("ERROR: no 'code' parameter found in the pasted URL.", file=sys.stderr)
            sys.exit(1)
    else:
        code = pasted

    flow.fetch_token(code=code)
    return flow.credentials


def save_token(credentials: Credentials, token_path: Path) -> None:
    """Persist credentials to disk in google-auth standard format."""
    token_path.write_text(credentials.to_json(), encoding="utf-8")


def get_channels(credentials: Credentials) -> list[dict]:
    """
    Return all channels accessible via this token.
    Each entry: {"id": "UCxxx", "title": "Channel Name"}
    """
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
    resp = youtube.channels().list(part="id,snippet", mine=True).execute()
    return [
        {"id": item["id"], "title": item["snippet"]["title"]}
        for item in resp.get("items", [])
    ]


def pick_channel(channels: list[dict], profile: str) -> dict:
    """
    If exactly one channel returned, return it automatically.
    If multiple, ask operator to select the correct one.
    """
    if len(channels) == 1:
        return channels[0]

    print(f"\n  Multiple channels found — which is the {profile.upper()} channel?")
    for i, ch in enumerate(channels, 1):
        print(f"    [{i}] {ch['title']}  ({ch['id']})")
    while True:
        raw = input(f"  Enter number (1–{len(channels)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(channels):
            return channels[int(raw) - 1]
        print("  Invalid choice — try again.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Pre-flight checks
    if not CLIENT_SECRET.exists():
        print(f"ERROR: client_secret.json not found at {CLIENT_SECRET}", file=sys.stderr)
        sys.exit(1)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Restrict config directory permissions to owner only
    CONFIG_DIR.chmod(0o700)

    profiles: dict[str, dict] = {}

    for profile in PROFILES:
        token_path = CONFIG_DIR / f"token_{profile}.json"

        # ── OAuth flow ────────────────────────────────────────────────────
        credentials = run_oauth(profile)
        save_token(credentials, token_path)
        token_path.chmod(0o600)  # owner read/write only
        print(f"\n  Token saved: {token_path}")

        # ── Discover channel(s) ───────────────────────────────────────────
        channels = get_channels(credentials)

        if not channels:
            print(f"\n  ⚠ No channels found for token_{profile}.json.")
            print(f"    Check that the YouTube Data API is enabled in your")
            print(f"    Google Cloud project and that this account owns a channel.")
            profiles[profile] = {
                "token_path": str(token_path),
                "channel_id": None,
                "channel_name": None,
            }
            continue

        print(f"\n  token_{profile}.json resolved to:")
        for ch in channels:
            print(f"    {ch['title']}  ({ch['id']})")

        chosen = pick_channel(channels, profile)
        print(f"\n  ✓ Selected: {chosen['title']}  ({chosen['id']})")

        print(f"\n  Which render locale does this channel serve?")
        print(f"    (e.g. 'en' for English, 'zh-Hans' for Simplified Chinese)")
        render_locale = input("  Render locale: ").strip() or profile

        profiles[profile] = {
            "locale":       render_locale,
            "token_path":   str(token_path),
            "channel_id":   chosen["id"],
            "channel_name": chosen["title"],
        }

    # ── Write youtube_profiles.json ───────────────────────────────────────────
    PROFILES_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    PROFILES_PATH.chmod(0o600)  # owner read/write only
    print(f"\n  Profiles saved: {PROFILES_PATH}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for profile, info in profiles.items():
        ch_id   = info.get("channel_id")   or "UNKNOWN"
        ch_name = info.get("channel_name") or "UNKNOWN"
        print(f"  token_{profile}.json → {ch_name}  ({ch_id})")

    resolved_ids = [
        info["channel_id"]
        for info in profiles.values()
        if info.get("channel_id")
    ]
    all_present  = len(resolved_ids) == len(PROFILES)
    all_distinct = len(set(resolved_ids)) == len(resolved_ids)

    if all_present and all_distinct:
        print("\n  Channel IDs are different ✓")
        print("  Run prepare_upload.py when ready to upload an episode.")
    else:
        print("\n  ⚠ Both tokens resolve to the same channel.")
        print("    The OAuth flow did not select different Brand Account channels.")
        print("    Options:")
        print("      1. Re-run and switch to the correct channel before clicking Allow.")
        print("      2. In YouTube settings, change your default channel, then re-run.")
        print("      3. Use two separate Google accounts (one per channel).")


if __name__ == "__main__":
    main()
