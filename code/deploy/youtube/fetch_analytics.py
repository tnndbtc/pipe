#!/usr/bin/env python3
"""
fetch_analytics.py — Pull YouTube Analytics for published videos and write back to DB.

Standalone script, called by cron every 6 hours.  No arguments needed —
reads all eligible rows from youtube_publish_log automatically.

Eligibility:
  - analytics_pulled_at IS NULL   (not yet fetched or previously gave up)
  - published_at IS NOT NULL      (video has a known publish timestamp)
  - published_at <= now - 72h     (data latency: impressions take 24–72h to appear)

Writes back to youtube_publish_log:
  ctr_pct, avg_view_duration, avg_view_pct, views, analytics_pulled_at

On no-data after GIVE_UP_AFTER_DAYS: sets analytics_pulled_at = -1.

Cron entry:
  0 */6 * * * cd /home/tnnd/data/code/pipe && \
    python code/deploy/youtube/fetch_analytics.py \
    >> /home/tnnd/data/code/http/logs/analytics.log 2>&1

Usage:
  python code/deploy/youtube/fetch_analytics.py
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── story_engine path ──────────────────────────────────────────────────────────
# fetch_analytics.py is at pipe/code/deploy/youtube/ → parent×4 = pipe/
# pipe/.parent = /home/tnnd/data/code/ → + story_engine/src
_SE_SRC = Path(__file__).resolve().parent.parent.parent.parent.parent / "story_engine" / "src"
if str(_SE_SRC) not in sys.path:
    sys.path.insert(0, str(_SE_SRC))
try:
    from db.models import get_connection
except ImportError as _e:
    print(f"ERROR: cannot import story_engine db.models: {_e}", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

PROFILES_PATH     = Path.home() / ".config" / "pipe" / "youtube_profiles.json"
FETCH_DELAY_H     = 72      # hours after publish before pulling analytics
GIVE_UP_AFTER_DAYS = 14     # days: if still no data, mark as no-data (-1)
# impressionClickThroughRate requires YouTube Partner Program (monetization).
# Omitted until the channel is monetized — ctr_pct column stays NULL meanwhile.
# Query shape: filters=video==<id>, no dimensions (validated: works without dimensions).
METRICS           = "views,averageViewDuration,averageViewPercentage"
SCOPES            = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


# ── Analytics pull for a single video ─────────────────────────────────────────

def _pull_one(yt_analytics, channel_id: str, row) -> None:
    """
    Query YouTube Analytics for one video and update youtube_publish_log.

    Sets analytics_pulled_at = -1 if no data is available after GIVE_UP_AFTER_DAYS.
    Leave analytics_pulled_at = NULL on transient API errors (retry next cron run).

    NOTE: API exceptions (e.g. HTTP 403 for deleted/private video) leave
    analytics_pulled_at = NULL — the cron retries indefinitely for those rows.
    See OPEN 5 in phase3_analytics.txt for the Phase 3b fix.
    """
    pub_dt   = datetime.fromtimestamp(row["published_at"], tz=timezone.utc)
    end_dt   = datetime.now(timezone.utc)
    start_dt = pub_dt - timedelta(days=1)   # start slightly before publish date
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    try:
        resp = yt_analytics.reports().query(
            ids       = f"channel=={channel_id}",
            startDate = start_str,
            endDate   = end_str,
            metrics   = METRICS,
            filters   = f"video=={row['video_id']}",
            # No dimensions= : filtering by video without a dimension returns the
            # video's aggregate metrics directly (validated working query shape).
        ).execute()
    except Exception as e:
        print(f"  ✗ Analytics query failed for {row['video_id']}: {e}")
        # ⚠ NOTE: API exceptions leave analytics_pulled_at = NULL → retried next run.
        # For permanent errors (HTTP 403/404), a future fix should set
        # analytics_pulled_at = -1 to stop retries (see OPEN 5).
        return

    # Parse response — columnHeaders = [{"name": "video", ...}, {"name": metric, ...}, ...]
    col_headers = [h["name"] for h in resp.get("columnHeaders", [])]
    data_rows   = resp.get("rows", [])

    if not data_rows:
        # No impressions data yet — decide whether to retry or give up
        give_up_ts = row["published_at"] + GIVE_UP_AFTER_DAYS * 86400
        if int(time.time()) > give_up_ts:
            conn = get_connection()
            conn.execute(
                "UPDATE youtube_publish_log SET analytics_pulled_at = -1 WHERE id = ?",
                (row["id"],),
            )
            conn.commit()
            conn.close()
            print(f"  ⚠  {row['video_id']} — no data after {GIVE_UP_AFTER_DAYS}d, marking as no-data")
        else:
            print(f"  ⚠  {row['video_id']} — no data yet, will retry next run")
        return   # analytics_pulled_at stays NULL → retried next cron (unless gave up)

    # Map column names → values from the first (only) data row
    values  = dict(zip(col_headers, data_rows[0]))
    avg_dur = values.get("averageViewDuration")   # float or None (seconds)
    avg_pct = values.get("averageViewPercentage") # float or None (%)
    views   = values.get("views")                 # int or None
    # ctr_pct stays NULL — impressionClickThroughRate unavailable until channel is monetized

    conn = get_connection()
    conn.execute(
        """UPDATE youtube_publish_log
           SET avg_view_duration   = ?,
               avg_view_pct        = ?,
               views               = ?,
               analytics_pulled_at = ?
           WHERE id = ?""",
        (avg_dur, avg_pct, views, int(time.time()), row["id"]),
    )
    conn.commit()
    conn.close()

    # Guard format strings against None (API may return None for metrics with no data)
    _dur  = f"{avg_dur:.0f}s" if avg_dur is not None else "n/a"
    _pct  = f"{avg_pct:.1f}%" if avg_pct is not None else "n/a"
    _views = str(int(views)) if views is not None else "n/a"
    print(f"  ✓ {row['video_id']}  views={_views}  avg_dur={_dur}  avg_pct={_pct}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  fetch_analytics  [{start_ts}]")
    print(f"{'='*60}\n")

    # ── Step 1: find eligible rows ─────────────────────────────────────────────
    cutoff = int(time.time()) - FETCH_DELAY_H * 3600
    conn = get_connection()
    rows = conn.execute(
        """SELECT id, video_id, upload_profile, published_at, lang
           FROM youtube_publish_log
           WHERE analytics_pulled_at IS NULL
             AND published_at IS NOT NULL
             AND published_at <= ?
           ORDER BY published_at ASC""",
        (cutoff,),
    ).fetchall()
    conn.close()

    if not rows:
        print("No videos ready for analytics pull.")
        return

    print(f"Found {len(rows)} video(s) eligible for analytics pull.\n")

    # ── Step 2: group by upload_profile ───────────────────────────────────────
    # Authenticate once per channel, not once per video
    by_profile: dict[str, list] = defaultdict(list)
    for row in rows:
        by_profile[row["upload_profile"]].append(row)

    # ── Step 3: load profiles ─────────────────────────────────────────────────
    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)
    profiles = json.load(open(PROFILES_PATH, encoding="utf-8"))

    # ── Step 4: for each profile, authenticate + pull ─────────────────────────
    for profile_key, profile_rows in by_profile.items():
        if profile_key not in profiles:
            print(f"  ⚠  Profile '{profile_key}' not in youtube_profiles.json — skip")
            continue

        token_path = Path(profiles[profile_key]["token_path"]).expanduser()
        channel_id = profiles[profile_key].get("channel_id", "")

        if not token_path.is_file():
            print(f"  ⚠  Token file not found: {token_path} — skip profile '{profile_key}'")
            continue

        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")

        yt_analytics = build("youtubeAnalytics", "v2", credentials=creds,
                             cache_discovery=False)

        print(f"Profile '{profile_key}': {len(profile_rows)} video(s)")
        for row in profile_rows:
            try:
                _pull_one(yt_analytics, channel_id, row)
            except Exception as _e:
                print(f"  ✗ Unexpected error for {row['video_id']}: {_e} — skipping")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
