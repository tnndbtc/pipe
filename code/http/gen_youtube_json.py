#!/usr/bin/env python3
"""
gen_youtube_json.py — Generate youtube.json directly (no server required).

This script replicates the logic of the server's /api/generate_youtube_json
and /api/youtube_save_all endpoints so that simple_run.sh can call it during
unattended (crontab) runs when the HTTP server is not running.

Usage:
    python3 gen_youtube_json.py \\
        --slug   world_story_2026-04-29_... \\
        --ep_id  s01e01 \\
        --locale zh-Hans \\
        [--playlist_id PLxxx] \\
        [--story_basename world_story_...]

Exits 0 on success, 1 on failure.  Prints exactly one final line:
    ok        — youtube.json written successfully
    fail      — generation failed (details on stderr)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# ── Genre → YouTube category_id mapping (mirror of server.py) ─────────────────
_GENRE_TO_CATEGORY = {
    "history":       "27",
    "documentary":   "27",
    "education":     "27",
    "sports":        "17",
    "news":          "25",
    "entertainment": "24",
    "comedy":        "23",
    "narration":     "25",
}
_DEFAULT_CATEGORY = "25"  # News & Politics

# ── Repo root (pipe/) ──────────────────────────────────────────────────────────
PIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _jload(p):
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _fetch_playlists_direct(locale: str) -> list:
    """Fetch YouTube playlists directly via the API (no server needed)."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request as _GRequest
        from googleapiclient.discovery import build as _yt_build
    except ImportError:
        return []

    profiles_path = os.path.expanduser("~/.config/pipe/youtube_profiles.json")
    if not os.path.isfile(profiles_path):
        return []
    try:
        profiles = json.load(open(profiles_path, encoding="utf-8"))
        # Match profile by locale
        locale_to_profile = {v.get("locale", k): k for k, v in profiles.items()}
        profile_key = locale_to_profile.get(locale, locale)
        if profile_key not in profiles:
            profile_key = next(iter(profiles))
        prof = profiles[profile_key]
        creds = Credentials.from_authorized_user_file(prof["token_path"])
        if creds.expired and creds.refresh_token:
            creds.refresh(_GRequest())
            open(prof["token_path"], "w", encoding="utf-8").write(creds.to_json())
        yt = _yt_build("youtube", "v3", credentials=creds)
        playlists = []
        req = yt.playlists().list(part="snippet", mine=True, maxResults=50)
        while req:
            resp = req.execute()
            for item in resp.get("items", []):
                playlists.append({"id": item["id"], "title": item["snippet"]["title"]})
            req = yt.playlists().list_next(req, resp)
        return playlists
    except Exception as exc:
        print(f"  [youtube] WARNING: could not fetch playlists: {exc}", file=sys.stderr)
        return []


def _match_playlist(story_basename: str, playlists: list) -> str | None:
    """Match a playlist by story filename prefix (e.g. 'world_story_...' → 'World')."""
    m = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)[\W_]', story_basename)
    if not m:
        return None
    prefix = m.group(1).lower()
    for exact in (True, False):
        for pl in playlists:
            t = pl["title"].strip().lower()
            if (exact and t == prefix) or (not exact and t.startswith(prefix)):
                return pl["id"]
    return None


def _call_claude(prompt_text: str, pipe_dir: str) -> str:
    """Call claude CLI and return stripped stdout."""
    exec_directive = (
        "You are an automated batch pipeline stage running with no human operator present. "
        "Execute the given task IMMEDIATELY and COMPLETELY. "
        "NEVER ask for confirmation, permission, or clarification. "
        "NEVER describe what you are about to do. "
        "NEVER offer choices or options. "
        "Complete every instruction from start to finish and then stop."
    )
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # prevent nested-session guard
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(prompt_text)
        tmp_path = tf.name
    try:
        result = subprocess.run(
            ["claude", "-p",
             "--model", "sonnet",
             "--dangerously-skip-permissions",
             "--no-session-persistence",
             "--append-system-prompt", exec_directive,
             tmp_path],
            capture_output=True, text=True, cwd=pipe_dir, timeout=120, env=env,
        )
    finally:
        os.unlink(tmp_path)
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(
            f"claude CLI failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:300]}"
        )
    raw = result.stdout.strip()
    # Strip markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    # Extract JSON object even if claude wrapped it in explanation text
    start = raw.find('{')
    end   = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return raw


def main():
    parser = argparse.ArgumentParser(description="Generate youtube.json without the HTTP server")
    parser.add_argument("--slug",            required=True)
    parser.add_argument("--ep_id",           required=True)
    parser.add_argument("--locale",          default="zh-Hans")
    parser.add_argument("--playlist_id",     default="")
    parser.add_argument("--story_basename",  default="")  # for playlist auto-match
    args = parser.parse_args()

    slug        = args.slug.strip()
    ep_id       = args.ep_id.strip()
    locale      = args.locale.strip()
    playlist_id = args.playlist_id.strip() or None

    ep_dir     = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    render_dir = os.path.join(ep_dir, "renders", locale)

    if not os.path.isdir(ep_dir):
        print(f"err: episode dir not found: {ep_dir}", file=sys.stderr)
        print("fail"); sys.exit(1)

    # ── Load episode files ─────────────────────────────────────────────────────
    script       = _jload(os.path.join(ep_dir, "Script.json"))      or {}
    shotlist     = _jload(os.path.join(ep_dir, "ShotList.json"))     or {}
    story_prompt = _jload(os.path.join(ep_dir, "StoryPrompt.json"))

    # ── Parse story.txt ───────────────────────────────────────────────────────
    sources          = []
    story_txt_title  = ""
    story_txt_lines  = []
    story_txt_path   = os.path.join(ep_dir, "story.txt")
    if os.path.isfile(story_txt_path):
        for sl in open(story_txt_path, encoding="utf-8"):
            sl_s = sl.rstrip()
            hm = re.match(r'^##\s+(.+)', sl_s)
            if hm:
                story_txt_title = hm.group(1).strip(); continue
            sm = re.match(r'^###\s+(.+)', sl_s)
            if sm:
                sources = re.findall(r'#([^\s#]+)', sm.group(1)); continue
            if sl_s.strip() in ("-", ""):
                continue
            story_txt_lines.append(sl_s.strip())

    # ── Collect narrator text (capped at 4000 chars) ──────────────────────────
    lines = []
    for scene in script.get("scenes", []):
        for action in scene.get("actions", []):
            if action.get("type") == "dialogue" and action.get("line"):
                lines.append(action["line"])
    if not lines and story_txt_lines:
        lines = story_txt_lines
    total, truncated = 0, []
    for line in lines:
        if total + len(line) > 4000:
            break
        truncated.append(line); total += len(line)

    # ── Shot summaries ─────────────────────────────────────────────────────────
    shots_data    = shotlist.get("shots", [])
    shot_cursor   = 0.0
    shot_summaries = []
    for sh in shots_data:
        dur = sh.get("duration_sec", 0)
        shot_summaries.append({
            "emotional_tag": sh.get("emotional_tag", ""),
            "duration_sec":  dur,
            "start_sec":     round(shot_cursor, 2),
            "background":    sh.get("background_id", ""),
        })
        shot_cursor += dur

    # ── category_id from genre ─────────────────────────────────────────────────
    genre       = script.get("genre", "").lower()
    category_id = _GENRE_TO_CATEGORY.get(genre, _DEFAULT_CATEGORY)

    # ── subtitle scan — ALL locale render dirs ─────────────────────────────────
    subtitles = []
    renders_root = os.path.join(ep_dir, "renders")
    if os.path.isdir(renders_root):
        for loc_dir in sorted(os.listdir(renders_root)):
            loc_render = os.path.join(renders_root, loc_dir)
            if not os.path.isdir(loc_render):
                continue
            for fname in sorted(os.listdir(loc_render)):
                if not fname.endswith(".srt"):
                    continue
                if ".en." in fname:
                    subtitles.append({"file": f"renders/{loc_dir}/{fname}",
                                      "language": "en", "name": "English"})
                elif ".zh-Hans." in fname:
                    subtitles.append({"file": f"renders/{loc_dir}/{fname}",
                                      "language": "zh-CN", "name": "Chinese Simplified"})

    # ── Load profiles ──────────────────────────────────────────────────────────
    profiles_path = os.path.expanduser("~/.config/pipe/youtube_profiles.json")
    profiles = {}
    if os.path.isfile(profiles_path):
        with open(profiles_path, encoding="utf-8") as f:
            profiles = json.load(f)
    locale_to_profile = {v.get("locale", k): k for k, v in profiles.items()}
    upload_profile = locale_to_profile.get(locale, locale)
    profile_info   = profiles.get(upload_profile, {})

    # ── Auto-match playlist if not provided ───────────────────────────────────
    if not playlist_id and args.story_basename:
        print("  Fetching playlists from YouTube API…", file=sys.stderr)
        playlists = _fetch_playlists_direct(locale)
        if playlists:
            playlist_id = _match_playlist(args.story_basename, playlists)
            if playlist_id:
                print(f"  Playlist  : auto-matched → {playlist_id}", file=sys.stderr)

    # ── Build Claude prompt ────────────────────────────────────────────────────
    ep_goal = None
    if story_prompt:
        ep_goal = (story_prompt.get("episode_goal")
                   or story_prompt.get("prompt_text", "")[:500])

    _eff_locale = profile_info.get("locale") or locale
    output_lang = "Chinese (Simplified)" if _eff_locale.startswith("zh") else "English"
    total_dur   = shotlist.get("total_duration_sec", 0)
    _script_title = script.get("title", "") or story_txt_title

    user_msg = json.dumps({
        "locale":             locale,
        "output_language":    output_lang,
        "episode_id":         ep_id,
        "genre":              genre,
        "total_duration_sec": total_dur,
        "script_title":       _script_title,
        "episode_goal":       ep_goal,
        "narrator_text":      truncated,
        "shots":              shot_summaries,
        "sources":            sources,
    }, ensure_ascii=False)

    system_prompt = (
        "You are a YouTube metadata expert. Generate upload metadata "
        "for a short narrative video episode. "
        "Output ONLY valid JSON with exactly these fields: "
        "title, description, tags, thumbnail_source_sec. "
        "No markdown, no explanation — raw JSON only.\n\n"
        "CRITICAL JSON RULES:\n"
        "- ALL double-quote characters inside string values MUST be escaped as \\\"\n"
        "- Example: write \\\"超级DSC\\\" not \"超级DSC\" inside a string value\n"
        "- Use Chinese quotation marks「」or『』instead of \\\" when quoting terms\n"
        "- The JSON must be complete and valid — do not truncate\n\n"
        "Constraints:\n"
        f"- title: ≤ 70 characters, in {output_lang}\n"
        "- description: first 2 lines are compelling hooks (shown in search results); "
        "hashtags ONLY in last paragraph; ≤ 5000 chars total; in {output_lang}\n"
        "- tags: 10-15 items, mix specific and broad terms\n"
        f"- thumbnail_source_sec: pick midpoint of shot with emotional_tag "
        f"'triumph', 'climax', or 'reveal'; must be within [0, {total_dur}]\n"
        "- Do NOT include category_id in the response"
    ).format(output_lang=output_lang)

    prompt_text = system_prompt + "\n\n---\n\nEpisode data (JSON):\n\n" + user_msg
    REQUIRED    = {"title", "description", "tags", "thumbnail_source_sec"}

    def _try_parse(raw: str):
        """Try json.loads; on failure attempt to repair unescaped inner quotes."""
        try:
            return json.loads(raw), None
        except json.JSONDecodeError as e:
            pass
        # Repair: replace bare " inside JSON string values with \" using a
        # state-machine walk.  This handles the common case where Claude
        # emits "quoted term" inside a string value without escaping.
        repaired = _repair_json_quotes(raw)
        try:
            return json.loads(repaired), None
        except json.JSONDecodeError as e2:
            return None, str(e2)

    def _repair_json_quotes(s: str) -> str:
        """Escape unescaped double-quotes that appear inside JSON string values."""
        out = []
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if in_string:
                if c == '\\':
                    # escaped character — pass both chars through unchanged
                    out.append(c)
                    i += 1
                    if i < len(s):
                        out.append(s[i])
                elif c == '"':
                    # This quote closes the string (we assume the JSON key/value
                    # structure is correct at the top level).  To decide whether
                    # this is a legitimate string-end or an errant inner quote,
                    # peek ahead: a valid string-end is followed by whitespace
                    # then one of  : , } ]
                    j = i + 1
                    while j < len(s) and s[j] in ' \t\r\n':
                        j += 1
                    next_ch = s[j] if j < len(s) else ''
                    if next_ch in (':', ',', '}', ']'):
                        # Legitimate string end
                        out.append(c)
                        in_string = False
                    else:
                        # Errant inner quote — escape it
                        out.append('\\"')
                else:
                    out.append(c)
            else:
                if c == '"':
                    in_string = True
                out.append(c)
            i += 1
        return ''.join(out)

    # ── Call Claude (with one retry on parse failure) ──────────────────────────
    print("  Calling claude CLI to generate YouTube metadata…", file=sys.stderr)
    raw = _call_claude(prompt_text, PIPE_DIR)
    suggested, parse_err = _try_parse(raw)
    if suggested is None:
        print("  Retrying after JSON parse failure…", file=sys.stderr)
        raw = _call_claude(prompt_text, PIPE_DIR)
        suggested, parse_err = _try_parse(raw)
    if suggested is None:
        print(f"err: Claude returned invalid JSON: {parse_err}\nraw: {raw[:300]}", file=sys.stderr)
        print("fail"); sys.exit(1)

    missing = REQUIRED - set(suggested.keys())
    if missing:
        print(f"err: Missing fields: {sorted(missing)}", file=sys.stderr)
        print("fail"); sys.exit(1)

    # ── Append CC BY credits ───────────────────────────────────────────────────
    licenses_path  = os.path.join(render_dir, "licenses.json")
    credits_block  = ""
    if os.path.isfile(licenses_path):
        try:
            lic_data = json.load(open(licenses_path, encoding="utf-8"))
            seen, credit_lines = set(), []
            for seg in lic_data.get("segments", []):
                if not seg.get("attribution_required"):
                    continue
                text = (seg.get("attribution_text") or "").strip()
                if text and text not in seen:
                    seen.add(text); credit_lines.append(text)
            if credit_lines:
                credits_block = "\n\n---\nCredits\n" + "\n".join(credit_lines)
        except Exception as exc:
            print(f"  [youtube] WARNING: could not read licenses.json: {exc}", file=sys.stderr)

    # ── Append sources deterministically ──────────────────────────────────────
    sources_block = ""
    if sources:
        _src_hashtags = " ".join(f"#{s}" for s in sources)
        sources_block = (
            f"\n\n来源：{_src_hashtags}" if output_lang.startswith("Chinese")
            else f"\n\nSources: {_src_hashtags}"
        )

    final_description = (
        suggested["description"].rstrip()
        + sources_block
        + credits_block
    )

    # ── Assemble draft ─────────────────────────────────────────────────────────
    _thumb_jpg = os.path.join(render_dir, "thumbnail.jpg")
    _thumb_png = os.path.join(render_dir, "thumbnail.png")
    _thumb_rel = (
        f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg"
        if os.path.isfile(_thumb_jpg) else
        f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.png"
        if os.path.isfile(_thumb_png) else None
    )

    draft = {
        "upload_profile":       upload_profile,
        "title":                suggested["title"],
        "description":          final_description,
        "tags":                 suggested.get("tags", []),
        "category_id":          category_id,
        "playlist_id":          playlist_id or profile_info.get("playlist_id"),
        "channel_id":           profile_info.get("channel_id"),
        "video_language":       locale if locale != "zh-Hans" else "zh-Hans",
        "privacy":              "private",
        "made_for_kids":        False,
        "thumbnail":            _thumb_rel,
        "thumbnail_source_sec": suggested.get("thumbnail_source_sec"),
        "subtitles":            subtitles,
        "publish_at":           None,
        "notify_subscribers":   False,
        "license":              "youtube",
        "embeddable":           True,
    }

    # ── Write youtube.json ─────────────────────────────────────────────────────
    os.makedirs(render_dir, exist_ok=True)
    yt_path = os.path.join(render_dir, "youtube.json")
    with open(yt_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)

    print("ok")
    sys.exit(0)


if __name__ == "__main__":
    main()
