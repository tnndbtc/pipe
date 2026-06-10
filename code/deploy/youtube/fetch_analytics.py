#!/usr/bin/env python3
"""
fetch_analytics.py — Pull YouTube Analytics for published videos and write back to DB.

Standalone script, called by cron every 6 hours.  No arguments needed —
reads all eligible rows from youtube_publish_log automatically.

Eligibility:
  - analytics_pulled_at IS NULL   (not yet fetched or previously gave up)
  - published_at IS NOT NULL      (video has a known publish timestamp)
  - published_at <= now - 72h     (data latency: impressions take 24–72h to appear)

Data fetched per video:
  Analytics API v2 (aggregate metrics):
    views, averageViewDuration, averageViewPercentage
    insightTrafficSourceType breakdown → traffic_sources JSON
  Data API v3 (public statistics):
    likeCount    → like_count
    commentCount → comment_count

Writes back to youtube_publish_log:
  avg_view_duration, avg_view_pct, views,
  like_count, comment_count, traffic_sources,
  analytics_pulled_at

On no-data after GIVE_UP_AFTER_DAYS: sets analytics_pulled_at = -1.

Usage:
  python code/deploy/youtube/fetch_analytics.py                  # pull analytics
  python code/deploy/youtube/fetch_analytics.py --list           # print video table
  python code/deploy/youtube/fetch_analytics.py --backfill-stats # fill likes/comments for old rows

Cron entry:
  0 */6 * * * cd /home/tnnd/data/code/pipe && \\
    python code/deploy/youtube/fetch_analytics.py \\
    >> /home/tnnd/data/code/http/logs/analytics.log 2>&1
"""

import argparse
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

PROFILES_PATH      = Path.home() / ".config" / "pipe" / "youtube_profiles.json"
FETCH_DELAY_H      = 72     # hours after publish before pulling analytics
GIVE_UP_AFTER_DAYS = 14     # days: if still no data, mark as no-data (-1)

# Analytics API v2 — aggregate metrics (no dimensions = one row per video)
ANALYTICS_METRICS  = "views,averageViewDuration,averageViewPercentage"
# impressionClickThroughRate requires YouTube Partner Program (monetization).
# Omitted until channel is monetized — ctr_pct column stays NULL meanwhile.

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


# ── Data API v3: like_count + comment_count ────────────────────────────────────

def _pull_video_stats(yt_data, video_id: str) -> tuple[int | None, int | None]:
    """
    Fetch like_count and comment_count from YouTube Data API v3 videos.list.

    Returns (like_count, comment_count).  Either may be None if the API does
    not return the field (e.g. comments disabled, likes hidden).
    Non-fatal: returns (None, None) on any error.

    Note: dislikeCount was removed from the public API in Dec 2021 and is
    not available through any supported endpoint.
    """
    try:
        resp = yt_data.videos().list(
            part="statistics",
            id=video_id,
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None, None
        stats = items[0].get("statistics", {})
        like_count    = int(stats["likeCount"])    if "likeCount"    in stats else None
        comment_count = int(stats["commentCount"]) if "commentCount" in stats else None
        return like_count, comment_count
    except Exception as exc:
        print(f"  ⚠  Data API stats failed for {video_id}: {exc}")
        return None, None


def _pull_data_api_stats_batch(yt_data, video_ids: list[str]) -> dict[str, dict]:
    """
    Fetch viewCount, likeCount, commentCount for up to 50 video IDs in one
    Data API v3 call.  viewCount is available immediately (no 72h wait).

    Returns dict: video_id -> {view_count, like_count, comment_count}
    Any field may be None if not returned by the API.
    Non-fatal: returns {} on any error.
    """
    results: dict[str, dict] = {}
    try:
        resp = yt_data.videos().list(
            part="statistics",
            id=",".join(video_ids),
            maxResults=50,
        ).execute()
        for item in resp.get("items", []):
            vid_id = item["id"]
            stats  = item.get("statistics", {})
            results[vid_id] = {
                "view_count":    int(stats["viewCount"])    if "viewCount"    in stats else None,
                "like_count":    int(stats["likeCount"])    if "likeCount"    in stats else None,
                "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
            }
    except Exception as exc:
        print(f"  ⚠  Batch Data API stats failed: {exc}")
    return results


def _refresh_early_views(pending_rows: list, profiles: dict) -> None:
    """
    Pull viewCount / likeCount / commentCount from Data API v3 for ALL pending
    videos — including those < 72h old that aren't yet eligible for the full
    Analytics API pass.

    Writes views, like_count, comment_count to youtube_publish_log but does NOT
    touch analytics_pulled_at (those rows stay pending for the full analytics
    pass once they reach the 72h threshold).

    Uses batched videos.list calls (up to 50 IDs per request) — cheap quota.
    """
    if not pending_rows:
        return

    by_profile: dict[str, list] = defaultdict(list)
    for row in pending_rows:
        by_profile[row["upload_profile"]].append(row)

    total_ok = 0
    for profile_key, profile_rows in by_profile.items():
        if profile_key not in profiles:
            print(f"  ⚠  Early views: profile '{profile_key}' not found — skip")
            continue

        token_path = Path(profiles[profile_key]["token_path"]).expanduser()
        if not token_path.is_file():
            print(f"  ⚠  Early views: token not found for '{profile_key}' — skip")
            continue

        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")

        yt_data = build("youtube", "v3", credentials=creds, cache_discovery=False)
        print(f"  '{profile_key}': {len(profile_rows)} pending video(s)")

        for i in range(0, len(profile_rows), 50):
            batch     = profile_rows[i : i + 50]
            vid_ids   = [r["video_id"] for r in batch]
            stats_map = _pull_data_api_stats_batch(yt_data, vid_ids)

            conn = get_connection()
            for row in batch:
                s = stats_map.get(row["video_id"])
                if not s:
                    continue
                # Always update views (latest real-time count from Data API).
                # like_count/comment_count: only fill if not already set.
                conn.execute(
                    """UPDATE youtube_publish_log
                       SET views         = %s,
                           like_count    = COALESCE(like_count, %s),
                           comment_count = COALESCE(comment_count, %s)
                       WHERE id = %s""",
                    (s["view_count"], s["like_count"], s["comment_count"], row["id"]),
                )
                total_ok += 1
            conn.commit()
            conn.close()

    print(f"  Early views done — {total_ok} video(s) updated.\n")


# ── Analytics API: traffic source breakdown ────────────────────────────────────

def _pull_traffic_sources(
    yt_analytics,
    channel_id: str,
    video_id: str,
    start_str: str,
    end_str: str,
) -> dict | None:
    """
    Fetch traffic source breakdown from YouTube Analytics API.

    Returns a dict mapping source type → view count, e.g.:
      {"YT_SEARCH": 42, "SUGGESTED_VIDEOS": 31, "BROWSE_FEATURES": 18, ...}

    Common source types:
      YT_SEARCH          — YouTube search
      SUGGESTED_VIDEOS   — suggested / up next
      BROWSE_FEATURES    — YouTube home / browse
      EXT_URL            — external websites / embeds
      NOTIFICATION       — notifications
      YT_CHANNEL         — channel page
      NO_LINK_OTHER      — direct / unknown
      PLAYLIST           — playlist
      SHORTS             — YouTube Shorts feed

    Returns None if no data or on any error (non-fatal).
    """
    try:
        resp = yt_analytics.reports().query(
            ids       = f"channel=={channel_id}",
            startDate = start_str,
            endDate   = end_str,
            metrics   = "views",
            dimensions = "insightTrafficSourceType",
            filters   = f"video=={video_id}",
        ).execute()
        rows = resp.get("rows", [])
        if not rows:
            return None
        return {r[0]: int(r[1]) for r in rows}
    except Exception as exc:
        print(f"  ⚠  Traffic sources failed for {video_id}: {exc}")
        return None


# ── Analytics pull for a single video ─────────────────────────────────────────

def _pull_one(yt_analytics, yt_data, channel_id: str, row) -> None:
    """
    Query YouTube Analytics + Data API for one video and update youtube_publish_log.

    Fetches:
      - views, avg_view_duration, avg_view_pct   (Analytics API)
      - traffic_sources breakdown                (Analytics API, dimensions query)
      - like_count, comment_count               (Data API v3)

    Sets analytics_pulled_at = -1 if no analytics data after GIVE_UP_AFTER_DAYS.
    Leaves analytics_pulled_at = NULL on transient API errors (retry next cron run).

    NOTE: API exceptions (e.g. HTTP 403 for deleted/private video) leave
    analytics_pulled_at = NULL — the cron retries indefinitely for those rows.
    """
    pub_dt    = datetime.fromtimestamp(row["published_at"], tz=timezone.utc)
    end_dt    = datetime.now(timezone.utc)
    start_dt  = pub_dt - timedelta(days=1)   # start slightly before publish date
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    # ── Step 1: aggregate metrics from Analytics API ───────────────────────────
    try:
        resp = yt_analytics.reports().query(
            ids       = f"channel=={channel_id}",
            startDate = start_str,
            endDate   = end_str,
            metrics   = ANALYTICS_METRICS,
            filters   = f"video=={row['video_id']}",
            # No dimensions= : filtering by video without a dimension returns the
            # video's aggregate metrics directly (validated working query shape).
        ).execute()
    except Exception as e:
        print(f"  ✗ Analytics query failed for {row['video_id']}: {e}")
        # Permanent errors (403 forbidden, 404 not found) → mark as no-data so
        # we stop retrying on every run.  Transient errors (5xx, network) leave
        # analytics_pulled_at = NULL so the next cron run retries them.
        err_str = str(e)
        if "403" in err_str or "404" in err_str:
            conn = get_connection()
            conn.execute(
                "UPDATE youtube_publish_log SET analytics_pulled_at = -1 WHERE id = %s",
                (row["id"],),
            )
            conn.commit()
            conn.close()
            print(f"  ⚠  Permanent error — marking {row['video_id']} as no-data (-1)")
        return

    col_headers = [h["name"] for h in resp.get("columnHeaders", [])]
    data_rows   = resp.get("rows", [])

    if not data_rows:
        # No impressions data yet — decide whether to retry or give up
        give_up_ts = row["published_at"] + GIVE_UP_AFTER_DAYS * 86400
        if int(time.time()) > give_up_ts:
            conn = get_connection()
            conn.execute(
                "UPDATE youtube_publish_log SET analytics_pulled_at = -1 WHERE id = %s",
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

    # ── Step 2: traffic source breakdown ──────────────────────────────────────
    traffic = _pull_traffic_sources(
        yt_analytics, channel_id, row["video_id"], start_str, end_str
    )

    # ── Step 3: like_count + comment_count from Data API v3 ───────────────────
    like_count, comment_count = _pull_video_stats(yt_data, row["video_id"])

    # ── Step 4: write all fields to DB ────────────────────────────────────────
    conn = get_connection()
    conn.execute(
        """UPDATE youtube_publish_log
           SET avg_view_duration   = %s,
               avg_view_pct        = %s,
               views               = %s,
               like_count          = %s,
               comment_count       = %s,
               traffic_sources     = %s,
               analytics_pulled_at = %s
           WHERE id = %s""",
        (
            avg_dur,
            avg_pct,
            views,
            like_count,
            comment_count,
            json.dumps(traffic, ensure_ascii=False) if traffic else None,
            int(time.time()),
            row["id"],
        ),
    )
    conn.commit()
    conn.close()

    # Guard format strings against None (API may return None for metrics with no data)
    _dur   = f"{avg_dur:.0f}s"     if avg_dur       is not None else "n/a"
    _pct   = f"{avg_pct:.1f}%"     if avg_pct       is not None else "n/a"
    _views = str(int(views))       if views          is not None else "n/a"
    _likes = str(like_count)       if like_count     is not None else "n/a"
    _cmts  = str(comment_count)    if comment_count  is not None else "n/a"
    _top_src = (
        max(traffic, key=traffic.get) if traffic else "n/a"
    )
    print(
        f"  ✓ {row['video_id']}  views={_views}  avg_dur={_dur}  avg_pct={_pct}"
        f"  likes={_likes}  cmts={_cmts}  top_src={_top_src}"
    )


# ── --backfill-stats: fill like_count/comment_count for already-pulled videos ──

def _backfill_stats(profiles: dict) -> None:
    """
    Fetch like_count and comment_count (Data API v3) for videos that already
    have analytics_pulled_at set but are missing like_count.

    Makes NO Analytics API calls — only Data API v3 videos.list (cheap, public).
    Processes all eligible videos in one pass, grouped by profile for auth reuse.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT id, video_id, upload_profile
           FROM youtube_publish_log
           WHERE analytics_pulled_at IS NOT NULL
             AND analytics_pulled_at != -1
             AND like_count IS NULL
           ORDER BY published_at DESC"""
    ).fetchall()
    conn.close()

    if not rows:
        print("No videos need stats backfill.")
        return

    print(f"Backfilling stats for {len(rows)} video(s)...\n")

    by_profile: dict[str, list] = defaultdict(list)
    for row in rows:
        by_profile[row["upload_profile"]].append(row)

    total_ok = 0
    for profile_key, profile_rows in by_profile.items():
        if profile_key not in profiles:
            print(f"  ⚠  Profile '{profile_key}' not in youtube_profiles.json — skip")
            continue

        token_path = Path(profiles[profile_key]["token_path"]).expanduser()
        if not token_path.is_file():
            print(f"  ⚠  Token file not found: {token_path} — skip")
            continue

        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")

        yt_data = build("youtube", "v3", credentials=creds, cache_discovery=False)

        print(f"Profile '{profile_key}': {len(profile_rows)} video(s)")
        for row in profile_rows:
            like_count, comment_count = _pull_video_stats(yt_data, row["video_id"])
            conn = get_connection()
            conn.execute(
                """UPDATE youtube_publish_log
                   SET like_count = %s, comment_count = %s
                   WHERE id = %s""",
                (like_count, comment_count, row["id"]),
            )
            conn.commit()
            conn.close()
            _lk = str(like_count)    if like_count    is not None else "n/a"
            _cm = str(comment_count) if comment_count is not None else "n/a"
            print(f"  ✓ {row['video_id']}  likes={_lk}  comments={_cm}")
            total_ok += 1

    print(f"\nBackfill done — {total_ok} video(s) updated.")


# ── --list: print video performance table from DB ─────────────────────────────

def _list_videos() -> None:
    """Print a performance table of all published videos from the local DB."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT video_id, lang, upload_profile, views, like_count, comment_count,
                  avg_view_pct, avg_view_duration, traffic_sources,
                  published_at, analytics_pulled_at
           FROM youtube_publish_log
           ORDER BY published_at DESC"""
    ).fetchall()
    conn.close()

    total = len(rows)
    pulled = sum(1 for r in rows if r["analytics_pulled_at"] and r["analytics_pulled_at"] != -1)

    print(f"\n{'='*100}")
    print(f"  YouTube Video Performance  ({total} videos, {pulled} with analytics)")
    print(f"{'='*100}")
    print(
        f"  {'video_id':<16} {'lang':<4} {'profile':<22} "
        f"{'views':>6} {'likes':>6} {'cmts':>5} {'avp%':>6} {'top traffic source':<22}  published"
    )
    print(f"  {'-'*98}")

    for row in rows:
        pub = (
            datetime.fromtimestamp(row["published_at"], tz=timezone.utc).strftime("%Y-%m-%d")
            if row["published_at"] else "N/A"
        )
        views  = str(row["views"])          if row["views"]          is not None else "—"
        likes  = str(row["like_count"])     if row["like_count"]     is not None else "—"
        cmts   = str(row["comment_count"])  if row["comment_count"]  is not None else "—"
        avp    = f"{row['avg_view_pct']:.1f}%" if row["avg_view_pct"] is not None else "—"

        # Find top traffic source for display
        top_src = "—"
        if row["traffic_sources"]:
            try:
                ts = json.loads(row["traffic_sources"])
                if ts:
                    top_src = max(ts, key=ts.get)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        status = ""
        if row["analytics_pulled_at"] is None:
            status = " (pending)"
        elif row["analytics_pulled_at"] == -1:
            status = " (no data)"

        print(
            f"  {row['video_id']:<16} {(row['lang'] or ''):<4} {(row['upload_profile'] or ''):<22} "
            f"{views:>6} {likes:>6} {cmts:>5} {avp:>6} {top_src:<22}  {pub}{status}"
        )

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube analytics for published videos."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print video performance table from DB and exit (no API calls).",
    )
    parser.add_argument(
        "--backfill-stats",
        action="store_true",
        dest="backfill_stats",
        help="Fetch like_count/comment_count for already-analyzed videos (Data API only).",
    )
    args = parser.parse_args()

    if args.list:
        _list_videos()
        return

    if args.backfill_stats:
        start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'='*60}")
        print(f"  fetch_analytics --backfill-stats  [{start_ts}]")
        print(f"{'='*60}\n")
        if not PROFILES_PATH.is_file():
            print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
            sys.exit(1)
        profiles = json.load(open(PROFILES_PATH, encoding="utf-8"))
        _backfill_stats(profiles)
        return

    start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}")
    print(f"  fetch_analytics  [{start_ts}]")
    print(f"{'='*60}\n")

    # ── Step 1: find ALL pending videos ───────────────────────────────────────
    conn = get_connection()
    all_pending = conn.execute(
        """SELECT id, video_id, upload_profile, published_at, lang
           FROM youtube_publish_log
           WHERE analytics_pulled_at IS NULL
             AND published_at IS NOT NULL
           ORDER BY published_at ASC""",
    ).fetchall()
    conn.close()

    if not all_pending:
        print("No pending videos found.")
        return

    # Videos >= 72h old are ready for the full Analytics API pass
    cutoff = int(time.time()) - FETCH_DELAY_H * 3600
    rows   = [r for r in all_pending if r["published_at"] <= cutoff]

    print(
        f"Found {len(all_pending)} pending video(s) total, "
        f"{len(rows)} ready for full analytics pull (>= {FETCH_DELAY_H}h old).\n"
    )

    # ── Step 2: group >= 72h rows by profile (for Analytics API pass) ──────────
    by_profile: dict[str, list] = defaultdict(list)
    for row in rows:
        by_profile[row["upload_profile"]].append(row)

    # ── Step 3: load profiles ─────────────────────────────────────────────────
    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)
    profiles = json.load(open(PROFILES_PATH, encoding="utf-8"))

    # ── Step 3b: Early views pass — Data API v3 for ALL pending ───────────────
    # Populates views (and likes/comments) for new videos < 72h old so they
    # show a real view count in the UI before full analytics are available.
    print("Early views pass (Data API v3)...")
    _refresh_early_views(all_pending, profiles)

    if not rows:
        print("No videos ready for full analytics pull yet (all < 72h old).")
        return

    # ── Step 4: for each profile, authenticate + pull Analytics API ────────────
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

        # Build both API clients once per profile (one auth, two services)
        yt_analytics = build("youtubeAnalytics", "v2", credentials=creds,
                             cache_discovery=False)
        yt_data      = build("youtube",           "v3", credentials=creds,
                             cache_discovery=False)

        print(f"Profile '{profile_key}': {len(profile_rows)} video(s)")
        for row in profile_rows:
            try:
                _pull_one(yt_analytics, yt_data, channel_id, row)
            except Exception as _e:
                print(f"  ✗ Unexpected error for {row['video_id']}: {_e} — skipping")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
