#!/usr/bin/env python3
"""
fetch_video_comments.py — Fetch viewer comments for published videos and write to DB.

Fetches top-level comments for all published videos that have comment_count > 0.
Uses YouTube Data API v3 commentThreads.list.

Run manually:
  cd /home/tnnd/data/code/pipe
  source /home/tnnd/.virtualenvs/pipe/bin/activate
  python code/deploy/youtube/fetch_video_comments.py

Or add to cron (daily):
  0 9 * * * cd /home/tnnd/data/code/pipe && \\
    source /home/tnnd/.virtualenvs/pipe/bin/activate && \\
    python code/deploy/youtube/fetch_video_comments.py \\
    >> /home/tnnd/data/code/http/logs/comments.log 2>&1

Use --refetch to re-fetch comments for videos already in the DB (picks up new comments).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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
    from db.models import init_db, get_connection, upsert_video_comment
except ImportError as _e:
    print(f"ERROR: cannot import story_engine db.models: {_e}", file=sys.stderr)
    sys.exit(1)

PROFILES_PATH = Path.home() / ".config" / "pipe" / "youtube_profiles.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
MAX_COMMENTS_PER_VIDEO = 20   # top-level comments to fetch per video


def _fetch_comments(yt, video_id: str) -> list[dict]:
    """
    Fetch up to MAX_COMMENTS_PER_VIDEO top-level comments for a video.

    Returns list of dicts: {comment_id, author_name, author_channel_id,
                             text, like_count, published_at}
    Returns [] if comments are disabled or on any API error.
    """
    try:
        resp = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=MAX_COMMENTS_PER_VIDEO,
            order="relevance",     # most relevant/liked first
            textFormat="plainText",
        ).execute()
    except Exception as exc:
        # commentThreads disabled (403) or video unavailable (404)
        print(f"  ⚠  Comments unavailable for {video_id}: {exc}")
        return []

    results = []
    for item in resp.get("items", []):
        top = item.get("snippet", {}).get("topLevelComment", {})
        snip = top.get("snippet", {})
        results.append({
            "comment_id":        top.get("id", ""),
            "author_name":       snip.get("authorDisplayName"),
            "author_channel_id": snip.get("authorChannelId", {}).get("value"),
            "text":              snip.get("textDisplay", ""),
            "like_count":        snip.get("likeCount", 0),
            "published_at":      snip.get("publishedAt"),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch viewer comments for published videos.")
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch comments for videos already in the DB (picks up new comments).",
    )
    args = parser.parse_args()

    start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  fetch_video_comments  [{start_ts}]")
    print(f"{'='*60}\n")

    init_db()

    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)
    profiles = json.load(open(PROFILES_PATH, encoding="utf-8"))

    # ── Find eligible videos ───────────────────────────────────────────────────
    conn = get_connection()
    if args.refetch:
        # Re-fetch all videos with comment_count > 0
        rows = conn.execute(
            """SELECT video_id, upload_profile, comment_count,
                      JSON_EXTRACT(hs.deep_story, '$.title') as story_title
               FROM youtube_publish_log ypl
               LEFT JOIN hierarchical_stories hs ON hs.story_set_id = ypl.story_set_id
               WHERE ypl.comment_count > 0
               ORDER BY ypl.published_at DESC"""
        ).fetchall()
    else:
        # Only fetch videos not yet in youtube_video_comments
        rows = conn.execute(
            """SELECT ypl.video_id, ypl.upload_profile, ypl.comment_count,
                      JSON_EXTRACT(hs.deep_story, '$.title') as story_title
               FROM youtube_publish_log ypl
               LEFT JOIN hierarchical_stories hs ON hs.story_set_id = ypl.story_set_id
               WHERE ypl.comment_count > 0
                 AND ypl.video_id NOT IN (
                     SELECT DISTINCT video_id FROM youtube_video_comments
                 )
               ORDER BY ypl.published_at DESC"""
        ).fetchall()
    conn.close()

    if not rows:
        msg = "No new videos with comments to fetch."
        if not args.refetch:
            msg += " (use --refetch to re-fetch already-fetched videos)"
        print(msg)
        return

    print(f"Found {len(rows)} video(s) to fetch comments for.\n")

    # Group by profile for auth reuse
    by_profile: dict[str, list] = defaultdict(list)
    for row in rows:
        by_profile[row["upload_profile"]].append(row)

    total_saved = 0
    for profile_key, profile_rows in by_profile.items():
        if profile_key not in profiles:
            print(f"  ⚠  Profile '{profile_key}' not in youtube_profiles.json — skip")
            continue

        token_path = Path(profiles[profile_key]["token_path"]).expanduser()
        if not token_path.is_file():
            print(f"  ⚠  Token not found: {token_path} — skip")
            continue

        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")

        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

        print(f"Profile '{profile_key}': {len(profile_rows)} video(s)")
        for row in profile_rows:
            title_preview = str(row["story_title"] or row["video_id"])[:55]
            comments = _fetch_comments(yt, row["video_id"])

            for c in comments:
                upsert_video_comment(
                    video_id          = row["video_id"],
                    comment_id        = c["comment_id"],
                    author_name       = c["author_name"],
                    author_channel_id = c["author_channel_id"],
                    text              = c["text"],
                    like_count        = c["like_count"],
                    published_at      = c["published_at"],
                )

            total_saved += len(comments)
            print(f"  ✓ {row['video_id']}  {len(comments)} comment(s)  — {title_preview}")

    print(f"\nDone — {total_saved} comment(s) written to DB.")


if __name__ == "__main__":
    main()
