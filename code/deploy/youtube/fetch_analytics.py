#!/usr/bin/env python3
"""
fetch_analytics.py — Pull YouTube Analytics for published videos and write back to DB.

Standalone script, called by cron every 6 hours.  No arguments needed —
reads all eligible rows from youtube_publish_log automatically.

Eligibility:
  - analytics_pulled_at IS NULL   (not yet fetched or previously gave up)
  - published_at IS NOT NULL      (video has a known publish timestamp)
  - published_at <= now - 72h     (data latency: impressions take 24–72h to appear)

Data fetched, per PROFILE (not per video — see _pull_batch_video_metrics):
  Analytics API v2, ONE call for the whole profile (dimensions=video, YouTube's
  "Top videos" report):
    views, averageViewDuration, averageViewPercentage, estimatedMinutesWatched,
    likes, dislikes, shares, subscribersGained, comments
  This replaced what used to be one Analytics API call per video. The only
  Analytics call still made per-video is traffic source breakdown
  (insightTrafficSourceType) — confirmed by live testing that YouTube does not
  support combining it with dimensions=video (400 "query not supported").

  Analytics API v2, ONCE per profile (channel-level Audience-tab snapshot):
    country, ageGroup+gender, deviceType, operatingSystem,
    insightPlaybackLocationType  →  youtube_channel_audience (full-replace)

  Analytics API v2, once per video (only the first time it's fetched):
    elapsedVideoTimeRatio audience retention curve (audienceWatchRatio,
    relativeRetentionPerformance) → youtube_video_retention_curve

  Data API v3 (public statistics, unchanged from before):
    likeCount    → like_count
    commentCount → comment_count

Writes back to youtube_publish_log:
  avg_view_duration, avg_view_pct, views, watch_time_hours, shares,
  subscribers_gained, dislikes, like_count, comment_count, traffic_sources,
  retention_curve_fetched_at, analytics_pulled_at

On no-data after GIVE_UP_AFTER_DAYS: sets analytics_pulled_at = -1.

NOTE on impressions/CTR: `impressionsClickThroughRate` and `impressions` are
NOT exposed by the YouTube Analytics API at all — confirmed by live testing
(`metrics=impressions` returns `400 Unknown identifier`). This is a permanent
Studio-only feature, unrelated to monetization status — do not wait for the
channel to be monetized expecting this to unlock; it won't. ctr_pct stays
NULL forever under the current API.

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

# Analytics API v2 — batch aggregate metrics, ONE call per profile via dimensions=video
# (YouTube's "Top videos" report). Column order in each result row matches this list.
BATCH_METRICS = (
    "views,averageViewDuration,averageViewPercentage,estimatedMinutesWatched,"
    "likes,dislikes,shares,subscribersGained,comments"
)
# impressions / impressionsClickThroughRate are NOT available via the Analytics API at
# all (confirmed: 400 "Unknown identifier"), regardless of monetization — omitted
# permanently, not "until monetized". ctr_pct column stays NULL under the current API.

# The dimensions=video ("Top videos") report 400s above this — undocumented, found by
# binary search (200 works, 250 doesn't). If a profile's pending backlog ever exceeds
# this, videos ranked outside the top 200 by views in the query window silently drop
# out of the batch result and read as "no data yet" until they age into the top 200 or
# GIVE_UP_AFTER_DAYS kicks in — see the warning at the call site in main().
TOP_VIDEOS_MAX_RESULTS = 200

# Channel-level Audience-tab queries — one call each, once per profile per run (not
# per video). age_gender uses two dimensions joined as "ageGroup|gender" in dim_key.
AUDIENCE_QUERIES = {
    "country":           dict(dimensions="country",              metrics="views",             sort="-views"),
    "age_gender":        dict(dimensions="ageGroup,gender",       metrics="viewerPercentage"),
    "device":            dict(dimensions="deviceType",            metrics="views",             sort="-views"),
    "os":                dict(dimensions="operatingSystem",       metrics="views",             sort="-views"),
    "playback_location": dict(dimensions="insightPlaybackLocationType", metrics="views",        sort="-views"),
}

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


# ── Analytics API: batch metrics for a whole profile in one call ───────────────

def _pull_batch_video_metrics(
    yt_analytics,
    channel_id: str,
    start_str: str,
    end_str: str,
    max_results: int,
) -> dict[str, dict]:
    """
    Fetch aggregate metrics for EVERY video on the channel within
    [start_str, end_str] in ONE Analytics API call (dimensions=video —
    YouTube's "Top videos" report; requires maxResults + sort or it 400s).

    This is what used to cost one API call per video. The only Analytics call
    still made per-video is traffic source breakdown — insightTrafficSourceType
    cannot be combined with dimensions=video (confirmed live: YouTube returns
    400 "query not supported" for that pairing), so it stays per-video.

    Returns {video_id: {views, avg_view_duration, avg_view_pct, watch_time_hours,
                         likes, dislikes, shares, subscribers_gained, comments}}.
    A video absent from the result has no data in this window at all (too new,
    or genuinely zero traffic ever) — a video WITH zero views still gets a row.

    Raises on error (e.g. 403 if the channel's GCP project has the Analytics
    API disabled) — the caller marks the whole profile's pending rows -1 in
    that case rather than retrying forever, since this failure mode showed up
    for real (katago3: 78 identical 403s in one run, one per video, before
    this batch rewrite made it a single error instead).

    maxResults is silently clamped to TOP_VIDEOS_MAX_RESULTS (confirmed live:
    the "Top videos" report 400s above 200 regardless of the value requested —
    it is NOT documented anywhere, discovered by binary search). If the caller
    asked for more than that, some pending videos ranked outside the top 200
    by views in this date window will be absent from the result and look like
    "no data yet" to _apply_batch_result — see the caller's warning.
    """
    if max_results > TOP_VIDEOS_MAX_RESULTS:
        max_results = TOP_VIDEOS_MAX_RESULTS
    resp = yt_analytics.reports().query(
        ids        = f"channel=={channel_id}",
        startDate  = start_str,
        endDate    = end_str,
        dimensions = "video",
        metrics    = BATCH_METRICS,
        sort       = "-views",
        maxResults = max_results,
    ).execute()
    out: dict[str, dict] = {}
    for r in resp.get("rows", []):
        vid, views, avg_dur, avg_pct, est_mins, likes, dislikes, shares, subs_gained, comments = r
        out[vid] = {
            "views":              int(views)   if views   is not None else None,
            "avg_view_duration":  float(avg_dur) if avg_dur is not None else None,
            "avg_view_pct":       float(avg_pct) if avg_pct is not None else None,
            "watch_time_hours":   round(float(est_mins) / 60.0, 4) if est_mins is not None else None,
            "likes":              int(likes)       if likes       is not None else None,
            "dislikes":           int(dislikes)    if dislikes    is not None else None,
            "shares":             int(shares)      if shares      is not None else None,
            "subscribers_gained": int(subs_gained) if subs_gained is not None else None,
            "comments":           int(comments)    if comments    is not None else None,
        }
    return out


# ── Analytics API: per-video audience retention curve (fetched once per video) ─

def _pull_retention_curve(
    yt_analytics, channel_id: str, video_id: str, start_str: str, end_str: str,
) -> list[dict] | None:
    """
    Fetch the per-video audience retention curve — Studio's Content-tab
    "Audience retention" graph: for ~100 points along the video's timeline,
    what fraction of viewers were still watching, and how that compares to
    similar-length videos on YouTube (relativeRetentionPerformance).

    Returns a list of {elapsed_video_time_pct, audience_watch_ratio,
    relative_performance} dicts (elapsed_video_time_pct is 0.00-1.00), or None
    on error / no data. Non-fatal — a failed curve pull never blocks the rest
    of the video's data from being written.
    """
    try:
        resp = yt_analytics.reports().query(
            ids        = f"channel=={channel_id}",
            startDate  = start_str,
            endDate    = end_str,
            dimensions = "elapsedVideoTimeRatio",
            metrics    = "audienceWatchRatio,relativeRetentionPerformance",
            filters    = f"video=={video_id}",
        ).execute()
        rows = resp.get("rows", [])
        if not rows:
            return None
        return [
            {
                "elapsed_video_time_pct": float(r[0]),
                "audience_watch_ratio":   float(r[1]) if r[1] is not None else None,
                "relative_performance":   float(r[2]) if r[2] is not None else None,
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"  ⚠  Retention curve failed for {video_id}: {exc}")
        return None


def _write_retention_curve(video_id: str, curve: list[dict]) -> None:
    """Replace the stored retention curve for one video and stamp the fetch gate."""
    if not curve:
        return
    now = int(time.time())
    conn = get_connection()
    conn.execute("DELETE FROM youtube_video_retention_curve WHERE video_id = %s", (video_id,))
    for point in curve:
        conn.execute(
            """INSERT INTO youtube_video_retention_curve
               (video_id, elapsed_video_time_pct, audience_watch_ratio, relative_performance, fetched_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (video_id, point["elapsed_video_time_pct"], point["audience_watch_ratio"],
             point["relative_performance"], now),
        )
    conn.execute(
        "UPDATE youtube_publish_log SET retention_curve_fetched_at = %s WHERE video_id = %s",
        (now, video_id),
    )
    conn.commit()
    conn.close()


# ── Analytics API: channel-level Audience-tab snapshot (once per profile) ──────

def _pull_channel_audience(
    yt_analytics, channel_id: str, start_str: str, end_str: str,
) -> dict[str, list[tuple[str, float]]]:
    """
    Fetch channel-level Audience-tab breakdowns: viewer country, age+gender,
    device type, OS, and playback location. Five cheap calls, once per profile
    per fetch run (not per video) — this is Studio's "Audience" tab data,
    which neither this pipeline nor the games pipeline fetched before (games
    only fetched country).

    Returns {dimension_name: [(dim_key, value), ...]}. A dimension missing
    from the result means its query failed — logged and skipped so one bad
    dimension (e.g. YouTube withholding a low-volume breakdown) doesn't block
    the others.
    """
    out: dict[str, list[tuple[str, float]]] = {}
    for name, q in AUDIENCE_QUERIES.items():
        try:
            resp = yt_analytics.reports().query(
                ids=f"channel=={channel_id}", startDate=start_str, endDate=end_str, **q,
            ).execute()
            rows = resp.get("rows", []) or []
            if name == "age_gender":
                out[name] = [(f"{r[0]}|{r[1]}", float(r[2])) for r in rows]
            else:
                out[name] = [(str(r[0]), float(r[1])) for r in rows]
        except Exception as exc:
            print(f"  ⚠  Audience dimension '{name}' failed: {exc}")
    return out


def _write_channel_audience(upload_profile: str, channel_id: str, audience: dict) -> None:
    """Full-replace youtube_channel_audience rows for this profile (channel-level snapshot,
    same replace-on-fetch pattern as the games pipeline's channel_country_views)."""
    if not audience:
        return
    now = int(time.time())
    conn = get_connection()
    conn.execute("DELETE FROM youtube_channel_audience WHERE upload_profile = %s", (upload_profile,))
    for dimension, pairs in audience.items():
        for dim_key, value in pairs:
            conn.execute(
                """INSERT INTO youtube_channel_audience
                   (channel_id, upload_profile, dimension, dim_key, metric_value, fetched_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (channel_id, upload_profile, dimension, dim_key, value, now),
            )
    conn.commit()
    conn.close()


# ── Apply one video's slice of the batch result + the per-video calls ──────────

def _apply_batch_result(
    yt_analytics, yt_data, channel_id: str, row, batch_metrics: dict,
    start_str: str, end_str: str,
) -> None:
    """
    Apply one video's slice of _pull_batch_video_metrics to the DB, then do the
    two things that still require a per-video call: traffic source breakdown
    (every run) and the retention curve (once, gated on retention_curve_fetched_at).

    Sets analytics_pulled_at = -1 if the video is absent from the batch AND
    past GIVE_UP_AFTER_DAYS. Leaves it NULL otherwise (retried next cron run).
    like_count/comment_count are NOT touched here — they come from the Data
    API v3 early-views pass (_refresh_early_views), which already ran for
    every pending video before this function is called.
    """
    metrics = batch_metrics.get(row["video_id"])
    if metrics is None:
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
        return

    # ── traffic source breakdown (still per-video — no supported batch combo) ──
    traffic = _pull_traffic_sources(yt_analytics, channel_id, row["video_id"], start_str, end_str)

    # ── retention curve — fetch once per video, not every run ─────────────────
    curve = None
    if not row.get("retention_curve_fetched_at"):
        curve = _pull_retention_curve(yt_analytics, channel_id, row["video_id"], start_str, end_str)

    conn = get_connection()
    conn.execute(
        """UPDATE youtube_publish_log
           SET avg_view_duration   = %s,
               avg_view_pct        = %s,
               views               = %s,
               watch_time_hours    = %s,
               shares              = %s,
               subscribers_gained  = %s,
               dislikes            = %s,
               traffic_sources     = %s,
               analytics_pulled_at = %s
           WHERE id = %s""",
        (
            metrics["avg_view_duration"],
            metrics["avg_view_pct"],
            metrics["views"],
            metrics["watch_time_hours"],
            metrics["shares"],
            metrics["subscribers_gained"],
            metrics["dislikes"],
            json.dumps(traffic, ensure_ascii=False) if traffic else None,
            int(time.time()),
            row["id"],
        ),
    )
    conn.commit()
    conn.close()

    if curve:
        _write_retention_curve(row["video_id"], curve)

    _dur   = f"{metrics['avg_view_duration']:.0f}s" if metrics['avg_view_duration'] is not None else "n/a"
    _pct   = f"{metrics['avg_view_pct']:.1f}%"       if metrics['avg_view_pct']      is not None else "n/a"
    _views = str(int(metrics['views']))              if metrics['views']            is not None else "n/a"
    _top_src = max(traffic, key=traffic.get) if traffic else "n/a"
    print(
        f"  ✓ {row['video_id']}  views={_views}  avg_dur={_dur}  avg_pct={_pct}"
        f"  shares={metrics['shares']}  subs+={metrics['subscribers_gained']}"
        f"  top_src={_top_src}" + ("  [+retention curve]" if curve else "")
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
        """SELECT id, video_id, upload_profile, published_at, lang, retention_curve_fetched_at
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

        # ── batch metrics pull: ONE Analytics API call replaces what used to be
        # one call per video (see _pull_batch_video_metrics docstring). ─────────
        if len(profile_rows) > TOP_VIDEOS_MAX_RESULTS:
            print(f"  ⚠  '{profile_key}' has {len(profile_rows)} pending video(s), above the "
                  f"{TOP_VIDEOS_MAX_RESULTS}-row API cap for this query — the lowest-view "
                  f"videos in this batch will not get data this run and will retry next time.")
        earliest_pub = min(r["published_at"] for r in profile_rows)
        start_str = (datetime.fromtimestamp(earliest_pub, tz=timezone.utc)
                     - timedelta(days=1)).strftime("%Y-%m-%d")
        end_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            batch_metrics = _pull_batch_video_metrics(
                yt_analytics, channel_id, start_str, end_str,
                max_results=TOP_VIDEOS_MAX_RESULTS,
            )
        except Exception as e:
            print(f"  ✗ Batch metrics query failed for profile '{profile_key}': {e}")
            err_str = str(e)
            if "accessNotConfigured" in err_str:
                # API not enabled for this GCP project — recoverable by the user
                # (enable it in Cloud Console), not a fact about the videos. Do
                # NOT mark -1: that's a per-video "we looked, there's nothing"
                # signal, and this is a per-profile "we couldn't even ask" signal.
                # Leaving analytics_pulled_at NULL costs one wasted API call per
                # run until fixed, but means it self-heals the moment the API is
                # enabled instead of requiring a manual DB reset. (Found live
                # 2026-08-02: katago3 hit this and was wrongly marked -1 before
                # this fix — see backfill note in the caller.)
                print(f"  ⚠  '{profile_key}': API not enabled for this GCP project — "
                      f"leaving {len(profile_rows)} video(s) pending, will retry next run")
            elif "403" in err_str or "404" in err_str:
                # Permanent, whole-profile error (e.g. channel deleted, real
                # permission denial) — mark every pending row -1 instead of
                # spending N identical failing calls every run forever.
                conn = get_connection()
                for row in profile_rows:
                    conn.execute(
                        "UPDATE youtube_publish_log SET analytics_pulled_at = -1 WHERE id = %s",
                        (row["id"],),
                    )
                conn.commit()
                conn.close()
                print(f"  ⚠  Permanent error — marking all {len(profile_rows)} video(s) in "
                      f"'{profile_key}' as no-data (-1)")
            continue  # nothing else to do for this profile without metrics

        for row in profile_rows:
            try:
                _apply_batch_result(yt_analytics, yt_data, channel_id, row,
                                    batch_metrics, start_str, end_str)
            except Exception as _e:
                print(f"  ✗ Unexpected error for {row['video_id']}: {_e} — skipping")

        # ── channel-level Audience-tab snapshot: 5 cheap calls, once per profile ──
        print(f"  Fetching Audience tab data for '{profile_key}' …")
        audience = _pull_channel_audience(yt_analytics, channel_id, start_str, end_str)
        _write_channel_audience(profile_key, channel_id, audience)
        if audience:
            print("  ✓ Audience snapshot: " +
                  ", ".join(f"{k}={len(v)} rows" for k, v in audience.items()))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
