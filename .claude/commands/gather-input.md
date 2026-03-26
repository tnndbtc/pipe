# gather-input

Scans an existing project and prints a complete briefing for an external agent
to write a new Script.json for the next episode.

Usage: /gather-input <project-name>

---

## Step 1 — Locate project

Find `projects/$ARGUMENTS/` in the current directory.
If it does not exist, exit immediately with:
  `✗ Project not found: projects/$ARGUMENTS/`

---

## Step 2 — Read VoiceCast.json (if exists)

Path: `projects/$ARGUMENTS/VoiceCast.json`

If present, extract for each character:
  - `character_id`
  - `gender`

If absent, character gender will be inferred from Script.json files in Step 3.

---

## Step 3 — Read all existing Script.json files

Glob: `projects/$ARGUMENTS/episodes/*/Script.json`

For each file found:
  - Extract `cast[].character_id`, `cast[].gender`, `cast[].role`
  - Collect all unique characters across all episodes

Merge with Step 2 data (VoiceCast.json is authoritative for gender; Script.json
`cast[].role` supplements it).

---

## Step 4 — Read canon.json (if exists)

Path: `projects/$ARGUMENTS/canon.json`

If present, extract:
  - `episode_summaries[]`       — one entry per past episode
  - `unresolved_threads[]`      — open story threads
  - `characters{}.status`       — current status of each character
  - `characters{}.location`     — current location of each character

---

## Step 5 — Read latest episode meta.json

List `projects/$ARGUMENTS/episodes/` and find the episode with the highest
episode number (e.g. s01e03 > s01e02).

Read `projects/$ARGUMENTS/episodes/{latest}/meta.json` and extract:
  - `project_slug`
  - `genre` (field: `series_genre`)
  - `locales`
  - `primary_locale` (if absent, use the first locale from `locales`)

---

## Step 6 — Determine next episode ID

List `projects/$ARGUMENTS/episodes/` directory.
Find the highest episode number (e.g. if s01e03 is the latest, next is s01e04).
Increment by 1 to get the next episode ID.

---

## Step 7 — Print output in 5 sections

```
── SECTION 1: Project Metadata ──────────────────────────────────────────
project_id    : {project_slug}
next_episode  : {next_episode_id}
genre         : {genre}
locales       : {locales}
primary_locale: {primary_locale}

── SECTION 2: Characters ─────────────────────────────────────────────────
| character_id | gender | role | status | location |
|--------------|--------|------|--------|----------|
| ...          | ...    | ...  | ...    | ...      |

(status and location from canon.json if available; leave blank if not known)

── SECTION 3: Story So Far ───────────────────────────────────────────────
Episode summaries (one per past episode, from canon.json episode_summaries[]).
If canon.json is absent, write: "(No canon history available yet.)"

Unresolved threads:
  • {thread 1}
  • {thread 2}
  ...
If none, write: "(No unresolved threads.)"

── SECTION 4: Script.json Template ──────────────────────────────────────
{
  "schema_id": "Script",
  "schema_version": "1.0.0",
  "script_id": "{project_slug}-{next_episode_id}-script",
  "project_id": "{project_slug}",
  "title": "Episode {next_episode_number}",
  "genre": "{genre}",
  "cast": [
    // pre-filled from Steps 2+3 — one entry per unique character
    { "character_id": "{id}", "gender": "{gender}", "role": "{role}" },
    ...
  ],
  "scenes": [
    {
      "scene_id": "s01",
      "location": "",
      "time_of_day": "day",
      "actions": [
        { "type": "action",   "text": "" },
        { "type": "dialogue", "speaker_id": "{character_id}", "line": "" }
      ]
    }
  ]
}

── SECTION 5: Rules for the External Agent ──────────────────────────────
1. character_id — use ONLY IDs from cast[] above; do not invent new ones
   unless adding a genuinely new character (add them to cast[] if so).
2. gender — must be "male", "female", or "neutral" (neutral for
   supernatural / robot / disembodied voices).
3. speaker_id in dialogue actions must exactly match a character_id in cast[].
4. actions[] — unified array mixing action beats and dialogue:
     action item:   { "type": "action",   "text": "Wind is blowing." }
     dialogue item: { "type": "dialogue",  "speaker_id": "boy", "line": "I thought you'd be here." }
   Use "text" for action items (not "description").
   Every scene MUST include at least "actions": [] (even if empty) — it is a
   required field in Script.v1.json; omitting it blocks the pipeline.
5. scene_id — simple slugs: s01, s02, s03 ...
6. time_of_day — one of: dawn, morning, afternoon, dusk, evening, night, midnight, day
7. Dialogue lines are copied VERBATIM into the pipeline. Do not add stage
   directions inside "line" values.
8. Place the completed Script.json at:
     projects/{project_slug}/episodes/{next_episode_id}/Script.json
   Then run the pipeline with:
     ./run.sh projects/{project_slug}/episodes/{next_episode_id}
```
