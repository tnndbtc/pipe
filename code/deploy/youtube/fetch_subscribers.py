#!/usr/bin/env python3
"""
fetch_subscribers.py — Fetch public subscribers from YouTube and write to DB.

Run manually whenever you want to refresh the subscriber list:
  cd /home/tnnd/data/code/pipe
  source /home/tnnd/.virtualenvs/pipe/bin/activate
  python code/deploy/youtube/fetch_subscribers.py

Or add to cron for daily refresh:
  0 8 * * * cd /home/tnnd/data/code/pipe && \\
    source /home/tnnd/.virtualenvs/pipe/bin/activate && \\
    python code/deploy/youtube/fetch_subscribers.py \\
    >> /home/tnnd/data/code/http/logs/subscribers.log 2>&1

Note: Only subscribers with PUBLIC subscriptions are returned by the API.
Subscribers who set their subscriptions to private will not appear.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── story_engine path ──────────────────────────────────────────────────────────
_SE_SRC = Path(__file__).resolve().parent.parent.parent.parent.parent / "story_engine" / "src"
if str(_SE_SRC) not in sys.path:
    sys.path.insert(0, str(_SE_SRC))
try:
    from db.models import init_db, upsert_subscriber
except ImportError as _e:
    print(f"ERROR: cannot import story_engine db.models: {_e}", file=sys.stderr)
    sys.exit(1)

PROFILES_PATH = Path.home() / ".config" / "pipe" / "youtube_profiles.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def _fetch_channel_details(yt, channel_id: str) -> dict:
    """
    Fetch public channel metadata for a subscriber via Data API v3.
    Returns a dict with snippet + statistics fields.
    """
    resp = yt.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id,
    ).execute()
    items = resp.get("items", [])
    if not items:
        return {}
    return items[0]


def _fetch_public_playlists(yt, channel_id: str) -> list[dict]:
    """
    Fetch all public playlists for a subscriber.
    Returns list of {id, title, item_count, created_at}.
    """
    try:
        resp = yt.playlists().list(
            part="snippet,contentDetails",
            channelId=channel_id,
            maxResults=50,
        ).execute()
        playlists = []
        for pl in resp.get("items", []):
            playlists.append({
                "id":         pl["id"],
                "title":      pl["snippet"].get("title", ""),
                "item_count": pl.get("contentDetails", {}).get("itemCount", 0),
                "created_at": pl["snippet"].get("publishedAt"),
            })
        return playlists
    except Exception as exc:
        print(f"  ⚠  Could not fetch playlists for {channel_id}: {exc}")
        return []


def main() -> None:
    start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  fetch_subscribers  [{start_ts}]")
    print(f"{'='*60}\n")

    # Ensure DB schema is up to date
    init_db()

    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)
    profiles = json.load(open(PROFILES_PATH, encoding="utf-8"))

    # Track all seen subscriber channel_ids → which profiles they follow
    # A subscriber may follow multiple of our channels
    seen: dict[str, dict] = {}   # channel_id → {info, subscribed_to: [profile_key, ...]}

    # ── Step 1: collect subscribers from all profiles ──────────────────────────
    for profile_key, profile in profiles.items():
        token_path = Path(profile["token_path"]).expanduser()
        if not token_path.is_file():
            print(f"  ⚠  Token not found for profile '{profile_key}' — skip")
            continue

        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")

        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

        # Fetch our own channel name for this profile
        our_channel_id   = profile.get("channel_id", "")
        our_channel_name = profile_key.upper()   # fallback
        try:
            ch_resp = yt.channels().list(part="snippet", mine=True).execute()
            ch_items = ch_resp.get("items", [])
            if ch_items:
                our_channel_name = ch_items[0]["snippet"].get("title", our_channel_name)
                our_channel_id   = our_channel_id or ch_items[0]["id"]
        except Exception as exc:
            print(f"  ⚠  Could not fetch channel name for '{profile_key}': {exc}")

        print(f"Profile '{profile_key}' ({our_channel_name}): fetching subscribers...")

        page_token = None
        sub_count = 0
        while True:
            kwargs = dict(
                part="subscriberSnippet,snippet",  # snippet.publishedAt = when they subscribed
                mySubscribers=True,
                maxResults=50,
            )
            if page_token:
                kwargs["pageToken"] = page_token

            resp = yt.subscriptions().list(**kwargs).execute()
            for item in resp.get("items", []):
                sub_snip  = item.get("subscriberSnippet", {})
                meta_snip = item.get("snippet", {})          # contains publishedAt
                cid = sub_snip.get("channelId", "")
                if not cid:
                    continue
                subscribed_at = meta_snip.get("publishedAt")  # ISO datetime of subscription
                if cid not in seen:
                    seen[cid] = {"snippet": sub_snip, "subscribed_to": [], "yt": yt}
                seen[cid]["subscribed_to"].append({
                    "profile":       profile_key,
                    "channel_id":    our_channel_id,
                    "channel_name":  our_channel_name,
                    "subscribed_at": subscribed_at,
                })
                sub_count += 1

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        print(f"  Found {sub_count} public subscriber(s) for '{profile_key}'")

    if not seen:
        print("\nNo public subscribers found across all profiles.")
        print("(Subscribers with private subscriptions are not visible via API.)")
        return

    # ── Step 2: enrich each unique subscriber with channel details + playlists ─
    print(f"\nEnriching {len(seen)} unique subscriber(s)...\n")
    for channel_id, info in seen.items():
        yt    = info["yt"]
        snip  = info["snippet"]
        name  = snip.get("title", "Unknown")

        print(f"  {name} ({channel_id})")

        # Channel details (stats, creation date, country)
        details = _fetch_channel_details(yt, channel_id)
        ch_snip  = details.get("snippet", {})
        ch_stats = details.get("statistics", {})

        description      = ch_snip.get("description", "") or ""
        country          = ch_snip.get("country")
        account_created  = ch_snip.get("publishedAt")
        subscriber_count = int(ch_stats["subscriberCount"]) if "subscriberCount" in ch_stats else None
        video_count      = int(ch_stats["videoCount"])      if "videoCount"      in ch_stats else None
        view_count       = int(ch_stats["viewCount"])       if "viewCount"       in ch_stats else None

        # Public playlists
        playlists = _fetch_public_playlists(yt, channel_id)

        # Write to DB
        upsert_subscriber(
            channel_id       = channel_id,
            display_name     = name,
            description      = description or None,
            country          = country,
            account_created  = account_created,
            subscriber_count = subscriber_count,
            video_count      = video_count,
            view_count       = view_count,
            subscribed_to    = info["subscribed_to"],
            public_playlists = playlists,
        )

        _subs  = str(subscriber_count) if subscriber_count is not None else "hidden"
        _vids  = str(video_count)      if video_count      is not None else "0"
        _pls   = str(len(playlists))
        print(f"    ✓ saved — subs={_subs}  videos={_vids}  playlists={_pls}  follows={info['subscribed_to']}")

    print(f"\nDone — {len(seen)} subscriber(s) written to DB.")


if __name__ == "__main__":
    main()
