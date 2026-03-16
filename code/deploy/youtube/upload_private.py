#!/usr/bin/env python3
"""
upload_private.py — Upload episode video, captions, thumbnail, and add to playlist.

Video is uploaded as PRIVATE.  After upload, run publish_episode.py to go public.

Usage:
    python code/deploy/youtube/upload_private.py EPISODE_DIR [--locale en] [--profile en]

Example:
    python code/deploy/youtube/upload_private.py projects/tennis/episodes/s01e01

Reads:
    renders/{locale}/youtube.json
    renders/{locale}/upload_state.json  (if resuming)
    renders/{locale}/output.mp4
    renders/{locale}/output.*.srt
    renders/{locale}/thumbnail.jpg
    ~/.config/pipe/youtube_profiles.json

Writes:
    renders/{locale}/upload_state.json  (updated throughout)

Exit codes:
    0 — upload complete
    1 — failure
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as _yt_build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("ERROR: google-api-python-client not installed.", file=sys.stderr)
    print("  pip install google-api-python-client google-auth-oauthlib", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

SCOPES        = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for captions.insert
]
PROFILES_PATH = Path.home() / ".config" / "pipe" / "youtube_profiles.json"

CHUNK_SIZE      = 5 * 1024 * 1024   # 5 MB resumable chunks
POLL_INTERVAL   = 30                 # seconds between processing polls
POLL_MAX_WAIT   = 30 * 60           # 30-minute max wait for processing
RETRY_MAX       = 5                  # max retries for transient errors
RETRY_ERRORS    = {"backendError", "rateLimitExceeded", "internalError"}
CAPTION_VERIFY_RETRIES = 3
CAPTION_VERIFY_DELAY   = 60         # seconds between caption verification retries

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
        body = json.loads(err.content)
        reason = body["error"]["errors"][0].get("reason", "")
        return reason in RETRY_ERRORS
    except Exception:
        return False


def _api_with_retry(request, label: str = ""):
    """Execute an API request with exponential backoff on transient errors."""
    for attempt in range(RETRY_MAX):
        try:
            return request.execute()
        except HttpError as e:
            if _is_transient(e) and attempt < RETRY_MAX - 1:
                wait = 2 ** attempt
                print(f"  ⟳ Transient error ({e.resp.status}) on {label} — retry in {wait}s …")
                time.sleep(wait)
                continue
            raise


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state(state_path: Path) -> dict:
    if state_path.is_file():
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "video_id":            None,
        "resumable_upload_uri": None,
        "video_uploaded":      False,
        "processing_confirmed": False,
        "captions_uploaded":   {},
        "thumbnail_uploaded":  False,
        "playlist_added":      False,
    }


# ── Upload steps ──────────────────────────────────────────────────────────────

def upload_video(youtube, meta: dict, mp4_path: Path,
                 state: dict, state_path: Path) -> str:
    """
    Resumable video upload.  Returns video_id.
    If upload_state has a resumable_upload_uri, resumes from there.
    """
    if state.get("video_uploaded") and state.get("video_id"):
        print(f"  ✓ Video already uploaded: {state['video_id']} — skipping")
        return state["video_id"]

    body = {
        "snippet": {
            "title":        meta["title"],
            "description":  meta.get("description", ""),
            "tags":         meta.get("tags", []),
            "categoryId":   str(meta.get("category_id", "24")),
            "defaultLanguage": meta.get("video_language", "en"),
        },
        "status": {
            "privacyStatus":     "private",
            "selfDeclaredMadeForKids": bool(meta.get("made_for_kids", False)),
            "license":           meta.get("license", "youtube"),
            "embeddable":        bool(meta.get("embeddable", True)),
        },
    }

    # Set publish_at in initial insert if present (creates already-scheduled video)
    if meta.get("publish_at"):
        body["status"]["publishAt"] = meta["publish_at"]

    # notifySubscribers is insert-only
    notify = bool(meta.get("notify_subscribers", False))

    media = MediaFileUpload(str(mp4_path), mimetype="video/mp4",
                            chunksize=CHUNK_SIZE, resumable=True)

    print(f"\n  Uploading video: {mp4_path.name}  ({mp4_path.stat().st_size / 1024 / 1024:.1f} MB) …")

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        notifySubscribers=notify,
        media_body=media,
    )

    # If resuming, restore the URI (google-api-python-client supports this via _resumable_uri)
    if state.get("resumable_upload_uri"):
        request.resumable_uri = state["resumable_upload_uri"]
        print(f"  ↩ Resuming from URI: {state['resumable_upload_uri'][:60]}…")

    response = None
    while response is None:
        for attempt in range(RETRY_MAX):
            try:
                status, response = request.next_chunk()

                # Persist URI after first chunk so we can resume on crash
                if request.resumable_uri and not state.get("resumable_upload_uri"):
                    state["resumable_upload_uri"] = request.resumable_uri
                    _save_state(state_path, state)

                if status:
                    pct = int(status.progress() * 100)
                    print(f"  … {pct}%", end="\r", flush=True)
                break  # success — exit retry loop
            except HttpError as e:
                if _is_transient(e) and attempt < RETRY_MAX - 1:
                    wait = 2 ** attempt
                    print(f"\n  ⟳ Upload error ({e.resp.status}) — retry in {wait}s …")
                    time.sleep(wait)
                    continue
                raise

    print()  # newline after progress
    video_id = response["id"]
    print(f"  ✓ Uploaded: https://www.youtube.com/watch?v={video_id}")

    state["video_id"]             = video_id
    state["video_uploaded"]       = True
    state["resumable_upload_uri"] = None  # clear — upload complete
    _save_state(state_path, state)

    return video_id


def wait_for_processing(youtube, video_id: str,
                         state: dict, state_path: Path) -> None:
    """Poll processingStatus until != 'processing'.  Must complete before captions."""
    if state.get("processing_confirmed"):
        print("  ✓ Processing already confirmed — skipping poll")
        return

    print(f"\n  Waiting for YouTube to process video {video_id} …")
    print(f"  (polling every {POLL_INTERVAL}s, max {POLL_MAX_WAIT//60} min)\n")
    waited = 0

    while waited < POLL_MAX_WAIT:
        resp = _api_with_retry(
            youtube.videos().list(part="processingDetails", id=video_id),
            label="processingDetails poll"
        )
        items = resp.get("items", [])
        if not items:
            print(f"  ⚠  video not found in videos.list — waiting {POLL_INTERVAL}s …")
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            continue

        pd = items[0].get("processingDetails", {})
        status = pd.get("processingStatus", "unknown")
        print(f"  processingStatus: {status}", end="\r", flush=True)

        if status == "processing":
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            continue
        elif status == "succeeded":
            print(f"\n  ✓ Processing succeeded")
            state["processing_confirmed"] = True
            _save_state(state_path, state)
            return
        else:
            # "failed" or "terminated"
            print(f"\n  ✗ Processing ended with status: {status}", file=sys.stderr)
            sys.exit(1)

    print(f"\n  ✗ Timed out waiting for processing after {POLL_MAX_WAIT//60} min",
          file=sys.stderr)
    sys.exit(1)


def upload_captions(youtube, video_id: str, render_dir: Path, meta: dict,
                    state: dict, state_path: Path, pipe_dir: Path) -> None:
    """Upload all subtitle tracks listed in youtube.json."""
    subtitles = meta.get("subtitles", [])
    if not subtitles:
        print("  ⚠  No subtitles defined in youtube.json — skipping captions")
        return

    captions_state = state.setdefault("captions_uploaded", {})

    for sub in subtitles:
        lang     = sub.get("language", "")
        name     = sub.get("name", lang)
        sub_file = sub.get("file", "")

        if captions_state.get(lang):
            print(f"  ✓ Caption '{lang}' already uploaded — skipping")
            continue

        # Resolve SRT path
        sub_path = Path(sub_file) if os.path.isabs(sub_file) else pipe_dir / sub_file
        if not sub_path.is_file():
            # Try relative to render_dir
            sub_path = render_dir / Path(sub_file).name
        if not sub_path.is_file():
            print(f"  ✗ Subtitle file not found: {sub_file}", file=sys.stderr)
            sys.exit(1)

        print(f"\n  Uploading caption: {name} ({lang})  {sub_path.name} …")

        caption_body = {
            "snippet": {
                "videoId":      video_id,
                "language":     lang,
                "name":         name,
                "isDraft":      False,
            }
        }

        media = MediaFileUpload(str(sub_path), mimetype="application/octet-stream",
                                resumable=False)

        for attempt in range(RETRY_MAX):
            try:
                resp = youtube.captions().insert(
                    part="snippet",
                    body=caption_body,
                    media_body=media,
                ).execute()
                print(f"  ✓ Caption uploaded: {name} ({lang})  id={resp.get('id')}")
                captions_state[lang] = True
                _save_state(state_path, state)
                break
            except HttpError as e:
                if _is_transient(e) and attempt < RETRY_MAX - 1:
                    wait = 2 ** attempt
                    print(f"  ⟳ Caption upload error — retry in {wait}s …")
                    time.sleep(wait)
                    continue
                print(f"  ✗ Caption upload failed for '{lang}': {e}", file=sys.stderr)
                raise


def upload_thumbnail(youtube, video_id: str, thumb_path: Path,
                     state: dict, state_path: Path) -> None:
    if state.get("thumbnail_uploaded"):
        print("  ✓ Thumbnail already uploaded — skipping")
        return

    if not thumb_path.is_file():
        print(f"  ⚠  Thumbnail not found: {thumb_path} — skipping thumbnail upload")
        return

    print(f"\n  Uploading thumbnail: {thumb_path.name} …")

    mime = "image/jpeg" if thumb_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    media = MediaFileUpload(str(thumb_path), mimetype=mime, resumable=False)

    try:
        _api_with_retry(
            youtube.thumbnails().set(videoId=video_id, media_body=media),
            label="thumbnails.set"
        )
    except HttpError as e:
        # Always print the raw error detail so we can diagnose silently-swallowed failures.
        try:
            err_body   = json.loads(e.content)
            err_msg    = err_body.get("error", {}).get("message", "")
            err_errors = err_body.get("error", {}).get("errors", [{}])
            reason     = err_errors[0].get("reason", "")
            domain     = err_errors[0].get("domain", "")
        except Exception:
            err_msg = reason = domain = ""

        print(f"  ✗ thumbnails.set HTTP {e.resp.status}  reason={reason!r}  domain={domain!r}  message={err_msg!r}")
        print(f"      Raw response: {e.content[:400]}")

        # 403 "forbidden" means the channel isn't verified for custom thumbnails.
        # Treat as a warning — don't block playlist or future steps.
        if e.resp.status == 403 and reason == "forbidden":
            print(f"  ⚠  Thumbnail upload skipped: channel not verified for custom thumbnails.")
            print(f"      Verify your channel at https://www.youtube.com/verify then re-run upload.")
            return
        raise

    print(f"  ✓ Thumbnail uploaded")
    state["thumbnail_uploaded"] = True
    _save_state(state_path, state)


def add_to_playlist(youtube, video_id: str, playlist_id: str,
                    state: dict, state_path: Path) -> None:
    if state.get("playlist_added"):
        print("  ✓ Already in playlist — skipping")
        return

    if not playlist_id:
        print("  ⚠  No playlist_id in youtube.json — skipping playlist step")
        return

    print(f"\n  Adding to playlist: {playlist_id} …")

    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {
                "kind":    "youtube#video",
                "videoId": video_id,
            }
        }
    }

    _api_with_retry(
        youtube.playlistItems().insert(part="snippet", body=body),
        label="playlistItems.insert"
    )

    print(f"  ✓ Added to playlist {playlist_id}")
    state["playlist_added"] = True
    _save_state(state_path, state)


def verify_captions(youtube, video_id: str, meta: dict) -> None:
    """Confirm all expected caption tracks appear.  Retries 3× with 60s delay."""
    expected_langs = {sub["language"] for sub in meta.get("subtitles", [])}
    if not expected_langs:
        return

    print(f"\n  Verifying captions (may take 1-2 min) …")

    for attempt in range(CAPTION_VERIFY_RETRIES):
        resp = _api_with_retry(
            youtube.captions().list(part="snippet", videoId=video_id),
            label="captions.list"
        )
        found_langs = {item["snippet"]["language"] for item in resp.get("items", [])}
        missing = expected_langs - found_langs

        if not missing:
            print(f"  ✓ All captions verified: {', '.join(sorted(expected_langs))}")
            return

        if attempt < CAPTION_VERIFY_RETRIES - 1:
            print(f"  ⟳ Missing tracks: {missing} — waiting {CAPTION_VERIFY_DELAY}s …")
            time.sleep(CAPTION_VERIFY_DELAY)
        else:
            print(f"  ⚠  Caption verification: still missing {missing} after retries")
            print(f"      Check YouTube Studio — tracks may still be processing")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload episode to YouTube as private")
    parser.add_argument("episode_dir", help="Episode directory, e.g. projects/tennis/episodes/s01e01")
    parser.add_argument("--locale",  default="en",  help="Render locale (default: en)")
    parser.add_argument("--profile", default=None,  help="Override upload_profile from youtube.json")
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    pipe_dir   = Path(__file__).resolve().parent.parent.parent.parent
    ep_dir     = Path(args.episode_dir)
    if not ep_dir.is_absolute():
        ep_dir = pipe_dir / ep_dir

    locale     = args.locale
    render_dir = ep_dir / "renders" / locale

    if not render_dir.is_dir():
        print(f"ERROR: render directory not found: {render_dir}", file=sys.stderr)
        sys.exit(1)

    yt_json_path  = render_dir / "youtube.json"
    state_path    = render_dir / "upload_state.json"

    if not yt_json_path.is_file():
        print(f"ERROR: youtube.json not found at {yt_json_path}", file=sys.stderr)
        sys.exit(1)

    with open(yt_json_path, encoding="utf-8") as f:
        meta = json.load(f)

    # ── Load upload state for resumption ──────────────────────────────────────
    state = _load_state(state_path)

    # ── Resolve upload profile ────────────────────────────────────────────────
    upload_profile = args.profile or meta.get("upload_profile", "")
    if not upload_profile:
        print("ERROR: upload_profile not set in youtube.json and not provided via --profile",
              file=sys.stderr)
        sys.exit(1)

    if not PROFILES_PATH.is_file():
        print(f"ERROR: youtube_profiles.json not found at {PROFILES_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)

    if upload_profile not in profiles:
        print(f"ERROR: profile '{upload_profile}' not in youtube_profiles.json", file=sys.stderr)
        sys.exit(1)

    profile_info = profiles[upload_profile]
    token_path   = Path(profile_info["token_path"]).expanduser()
    expected_ch  = profile_info.get("channel_id")

    # ── Authenticate ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    ep_id = ep_dir.name
    slug  = ep_dir.parent.parent.name
    print(f"  upload_private  —  {slug} / {ep_id}  (locale: {locale},  profile: {upload_profile})")
    print(f"{'='*60}\n")

    print(f"  Authenticating with {token_path.name} …")
    creds   = _load_credentials(token_path)
    youtube = _build_youtube(creds)

    # ── Verify channel ownership ──────────────────────────────────────────────
    ch_resp  = _api_with_retry(
        youtube.channels().list(part="id,snippet", mine=True),
        label="channels.list"
    )
    ch_items = ch_resp.get("items", [])
    ch_ids   = [c["id"] for c in ch_items]

    if expected_ch and expected_ch not in ch_ids:
        all_ch = ", ".join(f"{c['snippet']['title']}({c['id']})" for c in ch_items)
        print(f"ERROR: channel_id '{expected_ch}' not found in this account.", file=sys.stderr)
        print(f"  This account owns: {all_ch}", file=sys.stderr)
        sys.exit(1)

    ch_name = next(
        (c["snippet"]["title"] for c in ch_items if c["id"] == expected_ch),
        ch_items[0]["snippet"]["title"] if ch_items else "Unknown",
    )
    print(f"  Uploading as: {ch_name}  ({expected_ch or ch_ids[0] if ch_ids else '?'})")

    # ── File paths ────────────────────────────────────────────────────────────
    mp4_path   = render_dir / "output.mp4"
    thumb_rel  = meta.get("thumbnail", f"renders/{locale}/thumbnail.jpg")
    thumb_path = pipe_dir / thumb_rel if not os.path.isabs(thumb_rel) else Path(thumb_rel)

    # ── Verify saved video_id still exists on YouTube ─────────────────────────
    # If the user deleted the video from YouTube Studio, reset state so it
    # re-uploads from scratch rather than crashing on caption/thumbnail steps.
    if state.get("video_id") and state.get("video_uploaded"):
        try:
            _check = _api_with_retry(
                youtube.videos().list(part="id", id=state["video_id"]),
                label="verify video exists"
            )
            if not _check.get("items"):
                print(f"  ⚠  Video {state['video_id']} no longer exists on YouTube — resetting upload state.")
                state.update({
                    "video_id":             None,
                    "resumable_upload_uri": None,
                    "video_uploaded":       False,
                    "processing_confirmed": False,
                    "captions_uploaded":    {},
                    "thumbnail_uploaded":   False,
                    "playlist_added":       False,
                })
                _save_state(state_path, state)
        except HttpError as _ve:
            print(f"  ⚠  Could not verify video {state['video_id']} exists "
                  f"({_ve.resp.status}) — skipping check, proceeding with saved state.")

    # ── Step 1: Upload video ──────────────────────────────────────────────────
    video_id = upload_video(youtube, meta, mp4_path, state, state_path)

    # ── Step 2: Wait for processing ───────────────────────────────────────────
    wait_for_processing(youtube, video_id, state, state_path)

    # ── Step 3: Upload captions ───────────────────────────────────────────────
    upload_captions(youtube, video_id, render_dir, meta, state, state_path, pipe_dir)

    # ── Step 4: Upload thumbnail ──────────────────────────────────────────────
    upload_thumbnail(youtube, video_id, thumb_path, state, state_path)

    # ── Step 5: Add to playlist ───────────────────────────────────────────────
    add_to_playlist(youtube, video_id, meta.get("playlist_id"), state, state_path)

    # ── Step 6: Verify captions ───────────────────────────────────────────────
    verify_captions(youtube, video_id, meta)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  UPLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Video ID:  {video_id}")
    print(f"  YouTube:   https://www.youtube.com/watch?v={video_id}")
    print(f"  Studio:    https://studio.youtube.com/video/{video_id}/edit")
    print(f"\n  Video is PRIVATE.  Review in YouTube Studio, then run:")
    print(f"  python code/deploy/youtube/publish_episode.py {args.episode_dir}"
          + (f" --locale {locale}" if locale != "en" else ""))
    print()


if __name__ == "__main__":
    main()
