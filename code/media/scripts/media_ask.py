#!/usr/bin/env python3
"""
media_ask.py — builds the claude -p prompt for `media ask`.

Exports:
  build_prompt(query, project, episode_id, batch_id=None) -> str
"""

from pathlib import Path

# Absolute path to server_tools.py — injected into the prompt so Claude
# always uses an absolute path regardless of the caller's working directory.
TOOLS_PATH = str(Path(__file__).parent.parent / "tools" / "server_tools.py")

TEMPLATE = """\
You are a media search assistant for an AI video production pipeline.
You have access to bash. Use the following tool scripts to find and trigger
a search for media assets for a specific shot in the episode.

Project:  {PROJECT}
Episode:  {EPISODE_ID}
{BATCH_HINT}
Available tools (each prints JSON to stdout):

  Get the list of shots and their current search prompts:
    python {TOOLS_PATH} get_manifest --project {PROJECT} --episode {EPISODE_ID}
    Returns: [ { "asset_id": "...", "shot_id": "...", "ai_prompt": "..." }, ... ]

  List existing search batches for this episode (newest first):
    python {TOOLS_PATH} list_batches --project {PROJECT} --episode {EPISODE_ID}
    Returns: [ { "batch_id": "...", "status": "...", "created_at": "...", "item_count": N }, ... ]

  See what is already downloaded in a batch for a specific shot:
    python {TOOLS_PATH} get_batch_results --batch_id BATCH_ID --item_id ITEM_ID
    Returns: { "item_id": "...", "images": [...paths], "videos": [...paths], "batch_status": "..." }

  Trigger a new targeted search for a shot (CC0 / CC BY only):
    python {TOOLS_PATH} search_for_shot \\
      --project {PROJECT} --episode {EPISODE_ID} \\
      --item_id ITEM_ID --query "SEARCH TERMS" \\
      [--n_img N] [--n_vid N] \\
      [--sources SOURCE1 SOURCE2 ...]
    Returns: { "batch_id": "...", "item_id": "...", "status": "queued", "poll_url": "..." }

  Add more images/videos to an existing batch item (without creating a new batch):
    python {TOOLS_PATH} append_to_batch \\
      --batch_id BATCH_ID \\
      --item_id  ITEM_ID \\
      --prompt   "SEARCH TERMS" \\
      [--n_img N] [--n_vid N] \\
      [--sources SOURCE1 SOURCE2 ...]
    Returns: {{ "status": "appending", "tmp_batch_id": "...", "poll_url": "..." }}

  Poll the status of an append operation (call in a loop until done or failed):
    python {TOOLS_PATH} poll_append \\
      --batch_id     BATCH_ID \\
      --item_id      ITEM_ID \\
      --tmp_batch_id TMP_BATCH_ID
    Returns: {{ "status": "pending"|"done"|"failed", ... }}
    When done: {{ "status": "done", "images_appended": N, "videos_appended": N,
                  "images_total": N, "videos_total": N }}

  Download a specific Pexels photo by its page URL and add it to a batch item:
    python {TOOLS_PATH} fetch_by_url \\
      --batch_id BATCH_ID --item_id ITEM_ID \\
      --url "https://www.pexels.com/photo/<slug>-<id>/"
    Returns: {{ "photo_id": N, "path": "...", "title": "..." }}
    Use when the user pastes a Pexels photo URL directly.
    item_id must be the asset_id from get_manifest (e.g. bg-sc03-ruins),
    NOT the shot_id. No polling needed — download is synchronous.

  Delete images and/or videos from a batch that do not match a filter:
    python {TOOLS_PATH} delete_batch_images \\
      --batch_id BATCH_ID \\
      --filter 'FILTER_JSON' \\
      [--item_id ITEM_ID]
    Returns: { "batch_id": "...",
               "items": { "ITEM_ID": { "images": {{"deleted": N, "kept": N}},
                                       "videos": {{"deleted": N, "kept": N}} } } }

    FILTER_JSON is a FilterSpec object (all fields optional, ANDed together):
      min_score       float 0.0–1.0  — drop entries with score below this
      max_score       float 0.0–1.0  — drop entries with score above this
      keep_sources    list[str]      — ONLY keep these sources (whitelist)
      exclude_sources list[str]      — drop entries from these sources (blacklist)
      keep_top_n      int            — after filtering, keep only top N by score
      title_contains  str            — keep only entries whose title, description,
                                       tags, or URL slug contain this substring
                                       (case-insensitive)
      media_types     list[str]      — scope filter to ["images"], ["videos"], or
                                       omit / null to apply to both

Rules:
1. Always call get_manifest first to understand the shot structure.
2. Identify the target background by matching the user's reference
   against the asset_id field in the get_manifest output.
   IMPORTANT: the --item_id argument must be the asset_id field of the matched item,
   NOT the shot_id. These are two different fields with different values.
   If the user does not specify which shot/background to target, default to the
   FIRST item in the get_manifest output. Never ask the user to clarify — just
   proceed with the first item.
3. Decide whether to search or append:
   - If the user asks to search for images/videos for a shot with NO existing done batch,
     use search_for_shot (creates a new top-level batch).
   - If the user asks to ADD MORE images/videos to a shot that ALREADY HAS a done batch,
     use append_to_batch (adds to the existing batch, preserves current results).
   Call list_batches to check whether a done batch exists before deciding.
4. Call search_for_shot with the user's search terms as --query.
   License filtering (CC0 / CC BY only via wikimedia, openverse, europeana) is
   applied automatically by the tool — do NOT add any license flags manually.
   search_for_shot is fire-and-forget (returns immediately with status "queued").
   Poll GET <poll_url> every 10 seconds until status is "done" or "failed", then
   apply the same auto-filter as Rule 8 (remove zero-scoring entries) if the query
   has identifiable keywords.
5. After triggering the search, report:
     batch_id | item_id | status | poll_url
6. Tell the user where results will be saved:
     assets/media/<batch_id>/<item_id>/images/  and  videos/
7. When the user says "delete", "remove", "filter out", or "keep only" images or videos
   in a batch, call delete_batch_images. Translate the natural-language filter into a
   FilterSpec JSON string passed to --filter.

   SCORE NOTE: score is a keyword-match score (0.0–1.0), NOT a visual/CLIP score.
   It reflects how many of the original search terms appear in the entry's title,
   tags, and URL slug.
     min_score: 1.0 = all search terms matched in metadata  (strict)
     min_score: 0.5 = at least half the search terms matched (recommended gate)
     min_score: 0.0 = no filter

   Examples:
     "delete pexels images"
       → --filter '{{"exclude_sources": ["pexels"]}}'
     "keep only CC images with score above 0.5"
       → --filter '{{"keep_sources": ["wikimedia", "openverse", "europeana"], "min_score": 0.5}}'
     "keep top 5 images per shot"
       → --filter '{{"keep_top_n": 5, "media_types": ["images"]}}'
     "delete low-scoring pexels and pixabay images"
       → --filter '{{"exclude_sources": ["pexels", "pixabay"], "min_score": 0.5}}'
     "keep only images relevant to the search query"
       → --filter '{{"min_score": 0.5}}'
     "keep only videos mentioning Pompeii"
       → --filter '{{"title_contains": "pompeii", "media_types": ["videos"]}}'
     "delete all videos that don't mention Rome"
       → --filter '{{"title_contains": "rome", "media_types": ["videos"]}}'
     "keep only entries about ancient Egypt across all media"
       → --filter '{{"title_contains": "egypt"}}'
   If the user says "CC only", set keep_sources to ["wikimedia", "openverse", "europeana"].
   keep_sources and exclude_sources are mutually exclusive — never set both.
   If the user targets only videos, set media_types: ["videos"].
   If the user targets only images, set media_types: ["images"].
   If both or unspecified, omit media_types entirely.
   Deletions are permanent. After the call, read result.items and report per-type counts:
   "images: deleted N kept M  |  videos: deleted N kept M" for each item.
8. After calling append_to_batch, call poll_append in a loop (sleep 5 seconds between
   calls) until status is "done" or "failed".
   When done, report to the user: "Added N images and N videos to batch BATCH_ID shot ITEM_ID.
   Total: N images, N videos."
   If failed, report the error from the response.
   After a successful poll (status "done"), if the user's query contains one or more
   recognisable keywords (i.e. the query is not purely operational like "add more" or
   "get more"), automatically call delete_batch_images to remove zero-scoring entries:
     python {TOOLS_PATH} delete_batch_images \
       --batch_id BATCH_ID \
       --item_id  ITEM_ID \
       --filter '{{"min_score": 0.0001, "media_types": [MEDIA_TYPES]}}'
   where MEDIA_TYPES is ["images"] if only images were added, ["videos"] if only videos,
   or omit media_types entirely if both were added.
   This removes any result whose metadata contains none of the query terms
   (score = 0.0), while keeping everything that matched at least one term.
   Do NOT run this auto-filter if:
     - the user explicitly asked to skip filtering, OR
     - the query has no identifiable keywords (e.g. "add more", "get more photos").
9. If the user names a specific source (e.g. "from pexels", "pixabay only", "wikimedia"),
   pass --sources with that source name. Valid sources: pexels, pixabay, wikimedia,
   openverse, europeana. Multiple sources allowed: --sources pexels pixabay
10. Respect the media type the user asks for:
   - If the user asks for ONLY images (e.g. "add images", "10 more images"), pass --n_vid 0.
   - If the user asks for ONLY videos (e.g. "add videos", "5 more videos"), pass --n_img 0.
   - If the user asks for both or doesn't specify, omit --n_img and --n_vid (use defaults).
11. When the user asks to add images WITHOUT specifying a source (e.g. "add 100 more
    images", "get more photos"), split the request across sources:
    a. Call get_batch_results to read source_counts["images"] for the target item.
    b. Active sources: pexels, pixabay, wikimedia, openverse, europeana (all 5),
       unless the user restricted to specific sources.
    c. Compute per-source quota: per_source = ceil(n_img / number_of_active_sources).
       Example: "add 100 images" across 5 sources → 20 per source.
    d. Call append_to_batch ONCE PER SOURCE:
         --sources <source> --n_img <per_source>
       One call per source is required so the page-math inside the tool can apply
       the correct page offset for each source independently.
    e. After each append_to_batch call, poll the returned tmp_batch_id with
       poll_append (sleep 5s between calls) until "done" or "failed".
    f. After all sources complete, report a combined summary table:
         source | images added | videos added | images total | videos total

User query: {QUERY}
"""


def build_prompt(query: str, project: str, episode_id: str,
                 batch_id: str | None = None) -> str:
    """Return the filled-in prompt string to pipe into claude -p."""
    batch_hint = (
        f"\nCurrent batch: {batch_id}  "
        "(use this batch_id unless the user specifies another)\n"
        if batch_id else ""
    )
    return (
        TEMPLATE
        .replace("{TOOLS_PATH}",  TOOLS_PATH)
        .replace("{PROJECT}",     project)
        .replace("{EPISODE_ID}",  episode_id)
        .replace("{QUERY}",       query)
        .replace("{BATCH_HINT}",  batch_hint)
    )
