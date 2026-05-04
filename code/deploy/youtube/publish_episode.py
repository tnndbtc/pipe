#!/usr/bin/env python3
"""
publish_episode.py — Publish an uploaded episode (set to public or schedule).

Usage:
    python code/deploy/youtube/publish_episode.py EPISODE_DIR [--locale en]

Behaviour:
    if publish_at is set in youtube.json AND video is already scheduled:
        → no action needed (YouTube will auto-publish at publish_at)
    if publish_at is set but NOT yet applied:
        → videos.update  privacyStatus="private" + publishAt=publish_at
    if publish_at is null:
        → videos.update  privacyStatus="public"  (publish immediately)

Exit codes:
    0 — success
    1 — failure
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path


class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, log_path: Path):
        self._stream  = stream
        self._logfile = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def fileno(self):           # needed by some stdlib internals
        return self._stream.fileno()

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as _yt_build
    from googleapiclient.errors import HttpError
except ImportError:
    print("ERROR: google-api-python-client not installed.", file=sys.stderr)
    sys.exit(1)

# ── story_engine import (optional — best-effort, never blocks publish) ─────────
_SE_SRC = Path(__file__).resolve().parent.parent.parent.parent.parent / "story_engine" / "src"
if str(_SE_SRC) not in sys.path:
    sys.path.insert(0, str(_SE_SRC))
try:
    from db.models import register_youtube_publish as _register_yt
    _SE_AVAILABLE = True
except ImportError:
    _SE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

SCOPES        = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for captions.insert
]
PROFILES_PATH = Path.home() / ".config" / "pipe" / "youtube_profiles.json"
RETRY_MAX     = 5
RETRY_ERRORS  = {"backendError", "rateLimitExceeded", "internalError"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_credentials(token_path: Path) -> Credentials:
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _build_youtube(creds: Credentials):
    return _yt_build("youtube", "v3", credentials=creds, cache_discovery=False)


def _is_transient(err: HttpError) -> bool:
    if err.resp.status in (500, 503):
        return True
    try:
        body   = json.loads(err.content)
        reason = body["error"]["errors"][0].get("reason", "")
        return reason in RETRY_ERRORS
    except Exception:
        return False


def _api_with_retry(request, label: str = ""):
    for attempt in range(RETRY_MAX):
        try:
            return request.execute()
        except HttpError as e:
            if _is_transient(e) and attempt < RETRY_MAX - 1:
                wait = 2 ** attempt
                print(f"  ⟳ Transient error on {label} — retry in {wait}s …")
                time.sleep(wait)
                continue
            raise


def _parse_rfc3339(value: str) -> datetime.datetime:
    """Parse RFC 3339 datetime string.  Raises ValueError on bad format."""
    ts = value.replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(ts)


def _validate_publish_at(value: str) -> None:
    """Abort if publish_at is missing timezone or in the past."""
    if not (value.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", value)):
        print(f"ERROR: publish_at '{value}' is missing timezone (Z or ±HH:MM required).",
              file=sys.stderr)
        print("  Fix: add timezone to publish_at in youtube.json, e.g. '2026-03-15T18:00:00Z'",
              file=sys.stderr)
        sys.exit(1)
    try:
        dt  = _parse_rfc3339(value)
        now = datetime.datetime.now(datetime.timezone.utc)
        if dt <= now:
            print(f"ERROR: publish_at '{value}' is in the past ({dt.isoformat()}).",
                  file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"ERROR: cannot parse publish_at '{value}': {e}", file=sys.stderr)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Publish episode on YouTube")
    parser.add_argument("episode_dir", help="Episode directory, e.g. projects/tennis/episodes/s01e01")
    parser.add_argument("--locale", default="en", help="Render locale (default: en)")
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    pipe_dir   = Path(__file__).resolve().parent.parent.parent.parent
    ep_dir     = Path(args.episode_dir)
    if not ep_dir.is_absolute():
        ep_dir = pipe_dir / ep_dir

    locale     = args.locale
    render_dir = ep_dir / "renders" / locale

    # Tee stdout+stderr to a persistent log so every publish run is traceable
    render_dir.mkdir(parents=True, exist_ok=True)
    _log_path = render_dir / "youtube_action.log"
    import datetime as _dt
    _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout = _Tee(sys.stdout, _log_path)
    sys.stderr = _Tee(sys.stderr, _log_path)
    print(f"\n{'='*60}\n[{_ts}] publish_episode  locale={locale}\n{'='*60}")

    yt_json_path = render_dir / "youtube.json"
    state_path   = render_dir / "upload_state.json"

    if not yt_json_path.is_file():
        print(f"ERROR: youtube.json not found at {yt_json_path}", file=sys.stderr)
        sys.exit(1)
    if not state_path.is_file():
        print(f"ERROR: upload_state.json not found at {state_path}\n"
              f"  Run upload_private.py first.", file=sys.stderr)
        sys.exit(1)

    with open(yt_json_path,  encoding="utf-8") as f:
        meta = json.load(f)
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    video_id = state.get("video_id")
    if not video_id:
        print("ERROR: video_id not found in upload_state.json — was upload_private.py run?",
              file=sys.stderr)
        sys.exit(1)

    ep_id = ep_dir.name
    slug  = ep_dir.parent.parent.name

    print(f"\n{'='*60}")
    print(f"  publish_episode  —  {slug} / {ep_id}  (locale: {locale})")
    print(f"{'='*60}\n")
    print(f"  Video ID: {video_id}")

    # ── Resolve upload profile and authenticate ────────────────────────────────
    upload_profile = meta.get("upload_profile", "")
    if not upload_profile:
        print("ERROR: upload_profile not set in youtube.json", file=sys.stderr)
        sys.exit(1)

    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)

    # Resolve profile key: exact match, then base-language fallback (e.g. "en-US" → "en")
    if upload_profile not in profiles:
        _base = upload_profile.split("-")[0]
        if _base in profiles:
            upload_profile = _base
        else:
            print(f"ERROR: profile '{upload_profile}' not in youtube_profiles.json", file=sys.stderr)
            sys.exit(1)

    token_path = Path(profiles[upload_profile]["token_path"]).expanduser()
    creds      = _load_credentials(token_path)
    youtube    = _build_youtube(creds)
    print(f"  Authenticated with profile: {upload_profile}\n")

    # ── Publish registration helper ────────────────────────────────────────────
    # Called before every _print_links(video_id) in all three exit paths.
    # Wrapped in try/except — must never abort a publish that already succeeded.
    # Uses INSERT OR IGNORE so multiple calls across paths A/B/C are idempotent.
    def _do_register(video_id, publish_at_str=None):
        if not _SE_AVAILABLE:
            print("  ⚠  story_engine not importable — publish log skipped")
            return
        try:
            meta_path = ep_dir / "meta.json"
            _set_id, _story_id = None, None
            if meta_path.is_file():
                _m = json.load(open(meta_path, encoding="utf-8"))
                _set_id   = _m.get("story_set_id")
                _story_id = _m.get("story_id")
            # Fallback: parse slug if story_set_id not in meta.json
            # r'_(\d+)_[^_]*$' fails when ≥2 tokens follow set_id (e.g. _100_no_norm)
            if _set_id is None:
                _sm = re.search(r'_(\d+)_', slug)
                if _sm:
                    _set_id = int(_sm.group(1))

            # published_at: use scheduled time if available, else now
            if publish_at_str:
                try:
                    _pub_ts = int(_parse_rfc3339(publish_at_str).timestamp())
                except Exception:
                    _pub_ts = int(time.time())
            else:
                _pub_ts = int(time.time())

            # lang from locale prefix (reads actual locale value from profile)
            _locale = profiles[upload_profile].get("locale", upload_profile)
            _lang   = "en" if _locale.startswith("en") else "zh"

            _register_yt(
                video_id       = video_id,
                story_set_id   = _set_id,
                story_id       = _story_id,
                channel_id     = profiles[upload_profile].get("channel_id", ""),
                upload_profile = upload_profile,
                lang           = _lang,
                locale         = _locale,
                published_at   = _pub_ts,
            )
            print(f"  ✓ YouTube publish logged (story_set_id={_set_id})")
        except Exception as _e:
            print(f"  ⚠  YouTube publish log failed (non-fatal): {_e}")

    # ── Determine current video status ────────────────────────────────────────
    vid_resp = _api_with_retry(
        youtube.videos().list(part="status", id=video_id),
        label="videos.list status"
    )
    items = vid_resp.get("items", [])
    if not items:
        print(f"ERROR: video {video_id} not found via API — check video_id in upload_state.json",
              file=sys.stderr)
        sys.exit(1)

    current_status = items[0].get("status", {})
    current_privacy = current_status.get("privacyStatus", "")
    current_pub_at  = current_status.get("publishAt", "")

    print(f"  Current status: privacyStatus={current_privacy!r}  publishAt={current_pub_at!r}")

    # ── Publish logic ─────────────────────────────────────────────────────────
    publish_at = meta.get("publish_at")

    if publish_at:
        # Validate publish_at before doing anything
        _validate_publish_at(publish_at)

        # Check if already scheduled correctly
        if current_privacy == "private" and current_pub_at:
            # Video is already scheduled
            try:
                dt_current = _parse_rfc3339(current_pub_at)
                dt_new     = _parse_rfc3339(publish_at)
                if abs((dt_current - dt_new).total_seconds()) < 60:
                    print(f"\n  ✓ Already scheduled for {publish_at} — no action needed.")
                    print(f"    YouTube will auto-publish at that time.")
                    _do_register(video_id, publish_at_str=publish_at)   # Path A
                    _print_links(video_id)
                    return
            except ValueError:
                pass  # can't compare, re-apply

        # Apply (or update) the schedule:
        # CRITICAL: privacyStatus must be "private" when using publishAt.
        # Setting "public" + publishAt is interpreted as publish-now by YouTube.
        print(f"\n  Scheduling video for {publish_at} …")
        body = {
            "id": video_id,
            "status": {
                "privacyStatus": "private",
                "publishAt":     publish_at,
            },
        }
        _api_with_retry(
            youtube.videos().update(part="status", body=body),
            label="videos.update schedule"
        )
        print(f"  ✓ Scheduled: video will go public at {publish_at}")

    else:
        # Publish immediately
        if current_privacy == "public":
            print(f"\n  ✓ Already public — no action needed.")
            # Sync local youtube.json in case it was never updated
            if meta.get("privacy") != "public":
                meta["privacy"] = "public"
                with open(yt_json_path, "w", encoding="utf-8") as _f:
                    json.dump(meta, _f, ensure_ascii=False, indent=2)
                print(f"  ✓ Synced youtube.json privacy → public")
            _do_register(video_id)                                       # Path B
            _print_links(video_id)
            return

        print(f"\n  Publishing now (privacyStatus → public) …")
        body = {
            "id": video_id,
            "status": {
                "privacyStatus": "public",
            },
        }
        _api_with_retry(
            youtube.videos().update(part="status", body=body),
            label="videos.update publish"
        )
        print(f"  ✓ Published: https://www.youtube.com/watch?v={video_id}")
        # Update local youtube.json so UI reflects the new privacy state
        meta["privacy"] = "public"
        with open(yt_json_path, "w", encoding="utf-8") as _f:
            json.dump(meta, _f, ensure_ascii=False, indent=2)
        print(f"  ✓ Updated youtube.json privacy → public")

    _do_register(video_id, publish_at_str=publish_at)                    # Path C
    _print_links(video_id)


def _print_links(video_id: str) -> None:
    print(f"\n  Links:")
    print(f"    Video:  https://www.youtube.com/watch?v={video_id}")
    print(f"    Studio: https://studio.youtube.com/video/{video_id}/edit")
    print(f"    Subs:   https://studio.youtube.com/video/{video_id}/translations")
    print()


if __name__ == "__main__":
    main()
