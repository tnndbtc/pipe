#!/usr/bin/env python3
"""
prepare_upload.py — Validate episode files and write upload_review.json.

Usage:
    python code/deploy/youtube/prepare_upload.py EPISODE_DIR [--locale en] [--yes]

Example:
    python code/deploy/youtube/prepare_upload.py projects/tennis/episodes/s01e01

Reads:
    renders/{locale}/youtube.json
    renders/{locale}/output.mp4
    renders/{locale}/output.*.srt
    renders/{locale}/thumbnail.jpg
    ~/.config/pipe/youtube_profiles.json

Writes:
    renders/{locale}/upload_review.json

Exit codes:
    0 — all checks passed, ready to upload
    1 — one or more validation failures
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Optional Pillow — used for thumbnail validation
try:
    from PIL import Image as _PilImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Optional google-api-python-client — for channel/playlist validation
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as _yt_build
    _API_OK = True
except ImportError:
    _API_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────

SCOPES         = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # required for captions.insert
]
PROFILES_PATH  = Path.home() / ".config" / "pipe" / "youtube_profiles.json"

CATEGORY_NAMES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
    "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
    "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism",
}

PLACEHOLDER_TITLES = {"episode title", "todo", "untitled", "add title here"}
PLACEHOLDER_DESCS  = {"add description here", "todo", "…", "...", "add description"}

VALID_LOCALES = {"en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fail(msg: str) -> None:
    print(f"\n  ✗ {msg}", file=sys.stderr)
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _load_credentials(token_path: Path) -> "Credentials":
    """Load and auto-refresh OAuth credentials from disk."""
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _build_youtube(creds: "Credentials"):
    return _yt_build("youtube", "v3", credentials=creds, cache_discovery=False)


def _ffprobe_info(mp4: Path) -> dict:
    """Run ffprobe and return streams + format dict, or raise on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-print_format", "json",
        str(mp4),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.decode(errors='replace')}")
    return json.loads(r.stdout)


def _ffmpeg_extract_frame(mp4: Path, sec: float, dest: Path) -> None:
    """Extract a single frame at `sec` seconds and save as JPEG."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(sec),
        "-i", str(mp4),
        "-frames:v", "1",
        "-vf", "scale=1280:720",
        "-update", "1",          # required by newer ffmpeg for single-image JPEG output
        str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {r.stderr.decode(errors='replace')}")


def _srt_line_count(path: Path) -> int:
    """Count subtitle cue blocks (index lines) in an SRT file."""
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().isdigit():
            count += 1
    return count


def _srt_preview(path: Path, n: int = 2) -> list:
    """Return first n subtitle cues as list of dicts."""
    cues, current = [], {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines) and len(cues) < n:
        line = lines[i].strip()
        if line.isdigit():
            current = {"index": int(line)}
            i += 1
            if i < len(lines) and "-->" in lines[i]:
                parts = lines[i].split("-->")
                current["start"] = parts[0].strip()
                current["end"]   = parts[1].strip()
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                current["text"] = " ".join(text_lines)
                cues.append(current)
        else:
            i += 1
    return cues


def _validate_publish_at(value: str) -> bool:
    """Return True if value is RFC 3339 with timezone and in the future."""
    if not value:
        return True  # null is fine
    # Must end with Z or ±HH:MM
    if not (value.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", value)):
        return False
    try:
        # Parse with fromisoformat (Python 3.11+) or manual strip
        ts_str = value.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(ts_str)
        now = datetime.datetime.now(datetime.timezone.utc)
        return dt > now
    except ValueError:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate episode for YouTube upload")
    parser.add_argument("episode_dir", help="Episode directory, e.g. projects/tennis/episodes/s01e01")
    parser.add_argument("--locale", default="en", help="Render locale (default: en)")
    parser.add_argument("--yes", action="store_true", help="Skip interactive prompts (non-interactive mode)")
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    pipe_dir = Path(__file__).resolve().parent.parent.parent.parent
    ep_dir   = Path(args.episode_dir)
    if not ep_dir.is_absolute():
        ep_dir = pipe_dir / ep_dir

    locale     = args.locale
    render_dir = ep_dir / "renders" / locale

    if not ep_dir.is_dir():
        _fail(f"Episode directory not found: {ep_dir}")

    if not render_dir.is_dir():
        _fail(f"Render directory not found: {render_dir}\n"
              f"    Has the episode been rendered for locale '{locale}'?")

    # Derive slug and ep_id for the review packet
    ep_id = ep_dir.name
    slug  = ep_dir.parent.parent.name

    # Video probe results — initialized to safe defaults in case MP4 is missing/corrupt
    duration = 0.0
    size_mb  = 0.0
    width    = 0
    height   = 0
    fps      = 0

    print(f"\n{'='*60}")
    print(f"  prepare_upload  —  {slug} / {ep_id}  ({locale})")
    print(f"{'='*60}\n")

    # ── Load youtube.json ──────────────────────────────────────────────────────
    yt_json_path = render_dir / "youtube.json"
    if not yt_json_path.is_file():
        _fail(f"youtube.json not found at {yt_json_path}\n"
              f"    Create it with the required fields before running prepare_upload.")

    try:
        with open(yt_json_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"youtube.json is not valid JSON: {e}")

    # ── Load youtube_profiles.json ────────────────────────────────────────────
    if not PROFILES_PATH.is_file():
        _fail(f"youtube_profiles.json not found at {PROFILES_PATH}\n"
              f"    Run gen_tokens.py first.")

    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)

    upload_profile = meta.get("upload_profile", "").strip()
    if not upload_profile:
        _fail("youtube.json missing 'upload_profile' field.")
    if upload_profile not in profiles:
        _fail(f"upload_profile '{upload_profile}' not found in youtube_profiles.json.\n"
              f"    Available profiles: {list(profiles.keys())}")

    profile_info = profiles[upload_profile]
    token_path   = Path(profile_info["token_path"]).expanduser()
    expected_ch  = profile_info.get("channel_id")

    # ── Tracking: checks list and failure flag ────────────────────────────────
    checks  = []
    failed  = False
    warnings = []

    def record(name: str, ok: bool, detail: str) -> bool:
        nonlocal failed
        checks.append({"name": name, "ok": ok, "detail": detail})
        if ok:
            _ok(f"{name}: {detail}")
        else:
            _fail_soft(f"{name}: {detail}")
            failed = True
        return ok

    def _fail_soft(msg: str) -> None:
        print(f"  ✗ {msg}", file=sys.stderr)

    # ── Step 1: Validate MP4 with ffprobe ────────────────────────────────────
    mp4_path = render_dir / "output.mp4"
    if not mp4_path.is_file():
        record("mp4_exists", False, f"output.mp4 not found in {render_dir}")
    else:
        try:
            probe = _ffprobe_info(mp4_path)
            streams   = probe.get("streams", [])
            fmt       = probe.get("format", {})
            has_video = any(s.get("codec_type") == "video" for s in streams)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
            duration  = float(fmt.get("duration", 0))
            size_mb   = mp4_path.stat().st_size / 1024 / 1024

            # Resolution from video stream
            vstream = next((s for s in streams if s.get("codec_type") == "video"), {})
            width   = vstream.get("width", 0)
            height  = vstream.get("height", 0)
            fps_str = vstream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = round(int(num) / int(den), 2)
            except Exception:
                fps = 0

            mp4_ok = has_video and has_audio and duration > 0
            record("mp4_valid", mp4_ok,
                   f"{'video+audio streams' if mp4_ok else 'MISSING video or audio stream'}, "
                   f"{duration:.1f}s, {width}x{height}, {fps}fps, {size_mb:.1f} MB")
        except Exception as e:
            record("mp4_valid", False, str(e))
            # duration, size_mb, width, height, fps stay at 0 (initialized above)

    # ── Step 2: Title validation ──────────────────────────────────────────────
    title = meta.get("title", "").strip()
    title_len = len(title)
    if not title:
        record("title_length", False, "title is empty")
    elif title_len > 70:
        record("title_length", False, f"{title_len}/70 chars — exceeds limit")
    else:
        record("title_length", True, f"{title_len}/70 chars")

    if title.lower() in PLACEHOLDER_TITLES or any(p in title.lower() for p in ["todo", "untitled", "episode title"]):
        _warn("title looks like a placeholder — review before uploading")
        warnings.append("title_placeholder")
    if title and title == title.lower():
        _warn("title is all lowercase — unusual for YouTube titles")
        warnings.append("title_all_lowercase")

    # ── Step 3: Description validation ───────────────────────────────────────
    desc = meta.get("description", "").strip()
    desc_len = len(desc)

    if not desc or desc.lower() in PLACEHOLDER_DESCS:
        record("description_quality", False, "description is empty or placeholder")
    elif desc_len > 5000:
        record("description_quality", False, f"{desc_len}/5000 chars — exceeds limit")
    else:
        record("description_length", True, f"{desc_len}/5000 chars")

    # Check first two lines not blank
    desc_lines = desc.split("\n")
    if len(desc_lines) >= 2 and not desc_lines[0].strip():
        _warn("description first line is blank — search results show first 2 lines")
        warnings.append("desc_blank_first_line")

    # Check hashtags only at end
    hashtag_re = re.compile(r"#\w+")
    non_last_para = "\n".join(desc.split("\n\n")[:-1]) if "\n\n" in desc else ""
    if hashtag_re.search(non_last_para):
        _warn("hashtags found in description body (not last paragraph) — YouTube ignores them there")
        warnings.append("desc_hashtags_mid")

    # ── Step 4: Subtitle validation ───────────────────────────────────────────
    subtitles_meta = meta.get("subtitles", [])
    subtitle_infos = []

    for sub in subtitles_meta:
        sub_file = sub.get("file", "")
        sub_lang = sub.get("language", "")
        sub_name = sub.get("name", sub_lang)

        # Resolve path relative to pipe root
        sub_path = pipe_dir / sub_file if sub_file and not os.path.isabs(sub_file) else Path(sub_file)

        exists = sub_path.is_file()
        if not exists:
            # Also try relative to ep_dir
            alt = ep_dir / sub_file
            if alt.is_file():
                sub_path = alt
                exists = True

        lang_ok = sub_lang not in ("zh-Hans",)  # zh-Hans is invalid for API
        if not lang_ok:
            _warn(f"subtitle language '{sub_lang}' should be 'zh-CN' for YouTube API (file: {sub_file})")
            warnings.append(f"subtitle_lang_{sub_lang}")

        line_count = _srt_line_count(sub_path) if exists else 0
        preview    = _srt_preview(sub_path, 2)  if exists else []

        check_name = f"subtitle_{re.sub(r'[^a-z0-9]', '_', sub_lang)}_exists"
        record(check_name, exists, sub_file if exists else f"NOT FOUND: {sub_path}")

        subtitle_infos.append({
            "language":          sub_lang,
            "api_language_code": sub_lang,
            "file":              sub_file,
            "exists":            exists,
            "line_count":        line_count,
            "preview":           preview,
        })

    # ── Step 5: Thumbnail ────────────────────────────────────────────────────
    thumb_src_sec = meta.get("thumbnail_source_sec")
    thumb_rel     = meta.get("thumbnail", f"renders/{locale}/thumbnail.jpg")
    thumb_path    = pipe_dir / thumb_rel if thumb_rel and not os.path.isabs(thumb_rel) else Path(thumb_rel)

    # Auto-extract if thumbnail_source_sec is set and thumbnail.jpg is missing
    if not thumb_path.is_file() and thumb_src_sec is not None and mp4_path.is_file():
        print(f"\n  Extracting thumbnail at {thumb_src_sec}s from output.mp4 …")
        try:
            _ffmpeg_extract_frame(mp4_path, float(thumb_src_sec), thumb_path)
            print(f"  → saved {thumb_path}")
        except Exception as e:
            _warn(f"thumbnail auto-extract failed: {e}")

    thumb_info = {"file": str(thumb_rel), "exists": thumb_path.is_file()}

    if not thumb_path.is_file():
        record("thumbnail_exists", False,
               f"{thumb_rel} not found — provide thumbnail.jpg or set thumbnail_source_sec")
        thumb_info.update({"format": None, "width": 0, "height": 0,
                           "size_mb": 0, "aspect_ratio": None,
                           "aspect_ratio_ok": False, "size_ok": False})
    else:
        thumb_size_mb = thumb_path.stat().st_size / 1024 / 1024
        if _PIL_OK:
            try:
                with _PilImage.open(thumb_path) as img:
                    t_fmt  = img.format  # "JPEG" or "PNG"
                    t_w, t_h = img.size
            except Exception as e:
                record("thumbnail_format", False, f"cannot open with Pillow: {e}")
                t_fmt, t_w, t_h = None, 0, 0
        else:
            # Fallback: check magic bytes
            with open(thumb_path, "rb") as _tf:
                magic = _tf.read(4)
            if magic[:2] == b"\xff\xd8":
                t_fmt = "JPEG"
            elif magic[:4] == b"\x89PNG":
                t_fmt = "PNG"
            else:
                t_fmt = "UNKNOWN"
            t_w, t_h = 0, 0
            _warn("Pillow not installed — skipping thumbnail dimension check")

        fmt_ok    = t_fmt in ("JPEG", "PNG")
        dim_ok    = t_w >= 1280 and t_h >= 720
        size_ok   = thumb_size_mb <= 2.0
        aspect_ok = (t_h > 0) and abs((t_w / t_h) - (16 / 9)) < 0.02

        record("thumbnail_format", fmt_ok,
               f"{t_fmt}" if fmt_ok else f"{t_fmt} — only JPEG/PNG accepted")
        if t_w and t_h:
            record("thumbnail_size", dim_ok,
                   f"{t_w}x{t_h}" if dim_ok else f"{t_w}x{t_h} — minimum 1280×720")
        record("thumbnail_filesize", size_ok,
               f"{thumb_size_mb:.1f} MB" if size_ok else f"{thumb_size_mb:.1f} MB — exceeds 2 MB limit")
        if t_w and t_h and not aspect_ok:
            _warn(f"thumbnail aspect ratio is not 16:9 ({t_w}x{t_h})")
            warnings.append("thumbnail_aspect")

        thumb_info.update({
            "format":          t_fmt,
            "width":           t_w,
            "height":          t_h,
            "aspect_ratio":    f"{t_w}:{t_h}" if t_w else None,
            "size_mb":         round(thumb_size_mb, 2),
            "aspect_ratio_ok": aspect_ok,
            "size_ok":         size_ok,
        })

    # ── Step 6: Required fields ───────────────────────────────────────────────
    cat_id = str(meta.get("category_id", "")).strip()
    cat_name = CATEGORY_NAMES.get(cat_id, f"unknown ({cat_id})")
    record("category_present", bool(cat_id),
           f"{cat_id} ({cat_name})" if cat_id else "category_id missing")

    mfk = meta.get("made_for_kids")
    record("made_for_kids_set", mfk is not None,
           str(mfk) if mfk is not None else "made_for_kids missing")

    # ── Step 7: publish_at timezone ───────────────────────────────────────────
    pub_at = meta.get("publish_at")
    if pub_at:
        tz_ok     = bool(pub_at.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", pub_at))
        future_ok = _validate_publish_at(pub_at)
        if not tz_ok:
            record("publish_at_timezone", False,
                   f"'{pub_at}' — missing timezone (Z or ±HH:MM required)")
        elif not future_ok:
            record("publish_at_future", False,
                   f"'{pub_at}' — time is in the past")
        else:
            record("publish_at_valid", True, pub_at)

    # ── Steps 7b + 8: API-based validation (channel, playlist) ───────────────
    channel_info = {"channel_id": expected_ch, "channel_name": None,
                    "ownership_verified": False, "playlist_id": meta.get("playlist_id")}

    if not _API_OK:
        _warn("google-api-python-client not installed — skipping channel/playlist API checks")
    elif not token_path.is_file():
        _warn(f"Token file not found: {token_path} — skipping API checks")
    else:
        try:
            creds   = _load_credentials(token_path)
            youtube = _build_youtube(creds)

            # Channel ownership check
            print("\n  Verifying channel ownership …")
            ch_resp = youtube.channels().list(
                part="id,snippet", mine=True
            ).execute()
            ch_items = ch_resp.get("items", [])
            ch_ids   = [c["id"] for c in ch_items]

            if expected_ch and expected_ch in ch_ids:
                ch_name = next(
                    (c["snippet"]["title"] for c in ch_items if c["id"] == expected_ch),
                    "Unknown",
                )
                channel_info["channel_name"] = ch_name
                channel_info["ownership_verified"] = True
                print(f"  Uploading as: {ch_name}  ({expected_ch})")
                record("channel_ownership", True, f"{expected_ch} verified")
            else:
                all_ch = ", ".join(f"{c['snippet']['title']}({c['id']})" for c in ch_items)
                record("channel_ownership", False,
                       f"channel_id '{expected_ch}' not found in this account.\n"
                       f"    This account owns: {all_ch}")

            # Playlist validation
            playlist_id = meta.get("playlist_id")
            if playlist_id:
                print(f"\n  Checking playlist {playlist_id} …")
                pl_resp = youtube.playlists().list(
                    part="id", id=playlist_id
                ).execute()
                pl_found = bool(pl_resp.get("items"))
                record("playlist_exists", pl_found,
                       playlist_id if pl_found
                       else f"playlist '{playlist_id}' not found or not accessible")

        except Exception as e:
            _warn(f"API check error: {e}")

    # ── Compile final result ──────────────────────────────────────────────────
    validation_passed = not failed
    mp4_ok_val  = next((c["ok"] for c in checks if c["name"] == "mp4_valid"), False)
    thumb_exists = thumb_path.is_file()

    ready_to_upload = validation_passed and thumb_exists

    # ── Write upload_review.json ──────────────────────────────────────────────
    review = {
        "episode": {
            "project_id":  slug,
            "episode_id":  ep_id,
            "render_dir":  str(render_dir.relative_to(pipe_dir)),
        },
        "video": {
            "path":         str(mp4_path.relative_to(pipe_dir)) if mp4_path.is_file() else None,
            "duration_sec": round(duration, 2),
            "resolution":   f"{width}x{height}" if width else "unknown",
            "fps":          fps,
            "size_mb":      round(size_mb, 2),
            "ffprobe_ok":   mp4_ok_val,
        },
        "metadata": {
            "title":               meta.get("title", ""),
            "title_length":        title_len,
            "title_limit":         70,
            "description":         meta.get("description", ""),
            "description_length":  desc_len,
            "description_limit":   5000,
            "tags":                meta.get("tags", []),
            "category_id":         cat_id,
            "category_name":       cat_name,
            "made_for_kids":       meta.get("made_for_kids"),
            "privacy":             meta.get("privacy", "private"),
            "publish_at":          meta.get("publish_at"),
            "video_language":      meta.get("video_language", "en"),
            "upload_profile":      upload_profile,
        },
        "channel": channel_info,
        "subtitles": subtitle_infos,
        "thumbnail": thumb_info,
        "validation": {
            "passed": validation_passed,
            "warnings": warnings,
            "checks":  checks,
        },
        "generated_at":    datetime.datetime.utcnow().isoformat(),
        "ready_to_upload": ready_to_upload,
    }

    review_path = render_dir / "upload_review.json"
    review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY  —  {slug} / {ep_id} ({locale})")
    print(f"{'='*60}")
    print(f"  Title:   {title[:60]}{'…' if len(title) > 60 else ''}")
    print(f"  Channel: {channel_info.get('channel_name') or expected_ch or '(unknown)'}")
    print(f"  Upload profile: {upload_profile}")
    print()

    pass_count = sum(1 for c in checks if c["ok"])
    fail_count = sum(1 for c in checks if not c["ok"])
    print(f"  Checks:  {pass_count} passed,  {fail_count} failed,  {len(warnings)} warnings")
    print(f"  Review:  {review_path}")
    print()

    if ready_to_upload:
        print("  ✅ Ready to upload.")
        print("     Run: python code/deploy/youtube/upload_private.py "
              + args.episode_dir + (f" --locale {locale}" if locale != "en" else ""))
    else:
        print("  ❌ Not ready — fix the errors above, then re-run prepare_upload.")

    print()
    sys.exit(0 if ready_to_upload else 1)


if __name__ == "__main__":
    main()
