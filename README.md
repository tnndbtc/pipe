# Pipe — AI Narrative Generation Pipeline

An AI-driven system that transforms a story brief into fully rendered episodic video
with dubbed audio, character portraits, background images, SFX, and music.

---

## Architecture Overview

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│   Pipeline Server (CPU)         │     │   AI Asset Server (GPU node)     │
│   code/http/test_server.py      │────▶│   code/ai/http/server.py         │
│   port 8000  (stdlib only)      │     │   port 8000  (FastAPI)           │
│                                 │     │                                  │
│  • Web UI for pipeline          │     │  • Accepts job submissions       │
│  • Runs stages 0-10 via         │     │  • Runs gen_*.py scripts         │
│    run.sh / claude -p           │     │  • Polls: queued→running→done    │
│  • Azure TTS synthesis          │     │  • Jobs expire after 24 h        │
│  • Serves media files           │     │  • Config: code/ai/http/         │
└─────────────────────────────────┘     │    config.json                   │
                                        └──────────────────────────────────┘
```

Two separate servers, two separate machines. The pipeline server delegates GPU-heavy
generation (images, video, SFX) to the AI asset server over HTTP.

---

## Repository Layout

```
pipe/
├── run.sh                          # Pipeline executor (bash)
├── setup.sh
├── story_*.txt                     # Raw story input files
├── pipeline_vars.story_N.sh        # Per-story vars written by Stage 0
├── stage_logs/                     # stage_N.log written by run.sh
│
├── prompts/
│   ├── p_0.txt … p_9.txt          # Stage prompts (placeholders substituted by run.sh)
│   └── azure_tts_styles.txt       # Azure voice/style catalog for Stage 0 casting
│
├── contracts/schemas/              # JSON Schema v1 files for every output type
│   ├── AssetManifest.v1.json
│   ├── Script.v1.json
│   ├── ShotList.v1.json
│   └── …
│
├── code/
│   ├── http/                       # ── Pipeline server code (CPU) ──────────
│   │   ├── test_server.py          # Web UI + pipeline orchestration server
│   │   ├── gen_tts_cloud.py        # Azure TTS synthesis (--manifest only)
│   │   ├── gen_music_clip.py       # Music clip generation
│   │   ├── gen_render_plan.py      # Render plan assembly
│   │   ├── manifest_merge.py       # Merges shared + locale manifests
│   │   ├── post_tts_analysis.py    # Analyses TTS output timing
│   │   ├── resolve_assets.py       # Resolves media asset paths
│   │   ├── render_video.py         # Final video render
│   │   ├── pre_cache_voices.py     # Pre-warms Azure TTS voice cache
│   │   ├── fetch_ai_assets.py      # ★ NEW: client for AI asset server
│   │   └── tag_music.py
│   │
│   └── ai/                         # ── AI generation scripts (GPU) ─────────
│       ├── gen_character_images.py # Flux/SDXL portrait generation
│       ├── gen_background_images.py
│       ├── gen_background_video.py
│       ├── gen_sfx.py              # Sound effects (AudioGen / stable-audio)
│       ├── gen_character_animation.py
│       ├── gen_lipsync.py
│       ├── gen_upscale.py
│       └── http/
│           ├── server.py           # FastAPI AI asset server
│           ├── job_store.py        # Job state persistence
│           └── config.json         # api_key, job_dir, model selections
│
└── projects/
    └── {project_slug}/
        ├── canon.json              # Story world canon (project-level)
        ├── canon.{locale}.json
        ├── VoiceCast.json          # ★ Voice assignments (project-level, all locales)
        ├── characters/             # ★ AI-generated portraits (project-level)
        │   ├── {asset_id}.png      # e.g. char-amunhotep-v1.png
        │   └── index.json          # {generated_at, model, assets[]}
        └── episodes/{episode_id}/
            ├── StoryPrompt.json
            ├── Script.json
            ├── ShotList.json
            ├── AssetManifest.shared.json   # ★ locale-neutral assets
            ├── AssetManifest.{locale}.json        # locale vo_items (Stage 5) + merged (Stage 9)
            ├── RenderPlan.{locale}.json
            ├── canon_diff.json
            ├── assets/
            │   ├── {locale}/audio/vo/{item_id}.mp3
            │   ├── backgrounds/
            │   ├── sfx/
            │   └── bg_video/
            └── renders/{locale}/
                ├── output.mp4
                └── youtube_dubbed.aac
```

---

## Pipeline Stages

Run via `./run.sh [story_file] [from_stage] [to_stage]`

| Stage | Model  | Output file(s)                                 | Purpose                            |
|-------|--------|------------------------------------------------|------------------------------------|
| 0     | haiku  | `pipeline_vars.story_N.sh`, `VoiceCast.json`   | Extract vars, cast voices          |
| 1     | haiku  | _(no output file)_                             | Canon consistency check            |
| 2     | sonnet | `StoryPrompt.json`                             | Episode direction                  |
| 3     | sonnet | `Script.json`                                  | Script + character dialogue        |
| 4     | sonnet | `ShotList.json`                                | Visual shot breakdown              |
| 5     | sonnet | `AssetManifest.shared.json` + locale files | Asset manifest (images/sfx/TTS)  |
| 6     | haiku  | `canon_diff.json`                              | New story facts                    |
| 7     | haiku  | `canon.json` (updated)                         | World canon update                 |
| 8     | sonnet | `Script.{locale}.json` + locale manifests      | Translation & adaptation           |
| 9     | —      | `AssetManifest.{locale}.json`, `RenderPlan.{locale}.json` | Resolve assets & build render plan |
| 9     | —      | `renders/{locale}/output.mp4`                  | 7-step render (see below)          |

### Stage 9 substeps (per locale)
```
1. gen_music_clip.py    --manifest AssetManifest.shared.json
2. manifest_merge.py    --shared … --locale …  → AssetManifest.{locale}.json
3. gen_tts_cloud.py     --manifest AssetManifest.{locale}.json
4. post_tts_analysis.py --manifest AssetManifest.{locale}.json
5. resolve_assets.py    --manifest AssetManifest.{locale}.json        (in-place)
6. gen_render_plan.py   --manifest AssetManifest.{locale}.json        (in-place + RenderPlan)
7. render_video.py      --plan RenderPlan.{locale}.json
```

### run.sh mechanics
- Substitutes `{{PLACEHOLDER}}` tokens in `prompts/p_N.txt`
- Inlines all referenced input files into the prompt (eliminates Read tool latency)
- Passes `--dangerously-skip-permissions` and a no-confirmation system prompt
- Tees output to `stage_logs/stage_N.log`
- Model defaults: sonnet for stages 2/3/4/5/8; haiku for all others
- Override: `MODEL=opus ./run.sh` or `STAGE_MODEL_3=opus ./run.sh`

---

## Manifest Structure

Stage 5 produces two output files:

| File | Contents | Used by |
|------|----------|---------|
| `AssetManifest.shared.json` | `character_packs`, `backgrounds`, `sfx_items` | GPU gen scripts, gen_music_clip |
| `AssetManifest.{locale}.json` | `vo_items` (TTS dialogue), locale overrides (`locale_scope: "locale"`) | gen_tts_cloud.py |

### Section → asset type mapping
```
character_packs  →  asset_type "characters"  →  gen_character_images.py
backgrounds      →  asset_type "backgrounds" →  gen_background_images.py
backgrounds      →  asset_type "bg_video"    →  gen_background_video.py
sfx_items        →  asset_type "sfx"         →  gen_sfx.py
```

### character_packs entry (minimum required fields)
```json
{
  "asset_id":  "char-amunhotep-v1",
  "ai_prompt": "Ancient Egyptian High Priest, aged male 60s …"
}
```
Output filename from gen_character_images.py: `{asset_id}.png`

---

## VoiceCast.json  (project-level)

Written by Stage 0. Shared across all episodes of the same story.
Append-only — existing entries are never overwritten.

```json
{
  "schema_id": "VoiceCast", "schema_version": "1.0.0",
  "project_id": "the-pharaoh-who-defied-death",
  "characters": [
    {
      "character_id": "narrator",
      "role": "narrator", "gender": "neutral",
      "en": {
        "azure_voice":        "en-US-GuyNeural",
        "available_styles":   ["newscast", "angry", "cheerful", …],
        "azure_pitch":        "-5%",
        "azure_break_ms":     600,
        "azure_style_degree": 1.3
      },
      "zh-Hans": {
        "azure_voice":        "zh-CN-YunxiNeural",
        "available_styles":   ["angry", "cheerful", "sad", …],
        "azure_pitch":        "-5%",
        "azure_break_ms":     600,
        "azure_style_degree": 1.3
      }
    }
  ]
}
```

Every character including `narrator` must have an entry.
Stage 3 must use `character_id` values from VoiceCast.json in Script.json cast[].

---

## Azure TTS Fields on vo_items

All six fields must be set explicitly by Stage 5 (read from VoiceCast.json):

| Field | Source | Notes |
|-------|--------|-------|
| `azure_voice` | VoiceCast[char][locale].azure_voice | Explicit voice name |
| `azure_style` | Derived per line emotion + fallback chain | Must be in available_styles |
| `azure_style_degree` | 1.2–1.8 (1.5 default) | Cap at 1.8 |
| `azure_rate` | Derived from pace: slow→"-25%", fast→"+25%" | Overrides `pace` field |
| `azure_pitch` | VoiceCast[char][locale].azure_pitch | e.g. "-5%" |
| `azure_break_ms` | VoiceCast[char][locale].azure_break_ms | Set to 0 when style=whispering |

Style fallback chain (use nearest available when first choice missing):
```
angry   → shouting → unfriendly → serious → (omit)
fearful → terrified → whispering → sad    → (omit)
cheerful→ excited  → friendly   → hopeful → (omit)
excited → cheerful → friendly              → (omit)
```

---

## AI Asset Server API  (code/ai/http/server.py)

Base URL: `AI_SERVER_URL` env var (default `http://192.168.86.27:8000`)
Auth: `X-Api-Key` header (matches `code/ai/http/config.json` → `api_key`)

```
POST /jobs                         Submit generation job
GET  /jobs/{job_id}                Poll status + progress
GET  /jobs/{job_id}/files/{name}   Download output file
GET  /health                       GPU info, queue depth
```

### Job request
```json
{
  "manifest":    { … stripped to relevant section only … },
  "asset_types": ["characters"],
  "asset_ids":   ["char-amunhotep-v1"]   ← optional; omit = all
}
```

### Job state
```json
{
  "job_id": "uuid", "status": "queued|running|done|failed",
  "total": 5, "done": 3,
  "files": ["char-amunhotep-v1.png", …],
  "errors": [], "log_tail": ["…last 20 lines…"]
}
```

**Always send a stripped manifest** — only the section the GPU needs.
Do not send `vo_items`, `music_items`, or other sections.

---

## fetch_ai_assets.py  (★ new script — code/http/)

Client wrapper around the AI asset server. Equivalent role to `gen_tts_cloud.py`
but for GPU-side visual/audio generation.

```bash
python3 fetch_ai_assets.py --manifest AssetManifest.shared.json \
                            --asset_type characters
python3 fetch_ai_assets.py --manifest AssetManifest.shared.json \
                            --asset_type characters --asset-id char-amunhotep-v1
```

Behaviour:
- Scans output dir for already-downloaded files (`{asset_id}.png`)
- Submits only missing IDs via `asset_ids` — GPU wastes no time on cached work
- Strips manifest to the relevant section before sending
- Derives output path from manifest fields (no `--output_dir`):
  - `characters` → `projects/{project_id}/characters/`  (project-level)
  - `backgrounds` → `projects/{project_id}/episodes/{ep_id}/assets/backgrounds/`
  - `sfx`         → `projects/{project_id}/episodes/{ep_id}/assets/sfx/`
  - `bg_video`    → `projects/{project_id}/episodes/{ep_id}/assets/bg_video/`
- Writes `index.json` alongside downloaded files (AI server jobs expire in 24 h)
- Config: `AI_SERVER_URL` and `AI_SERVER_KEY` env vars
- stdlib only (urllib.request) — no external dependencies

---

## pre_cache_voices.py  (code/http/)

Pre-warms the Azure TTS cache for all voices in `VOICES` dict across all
sentence categories and styles. Runs offline; idempotent.

Cache key: `sha256(json.dumps({"v":voice,"s":style,"d":degree,"r":rate,"p":pitch,"b":break_ms,"t":text}, sort_keys=True))[:16]`

Cache location: `projects/{slug}/assets/{locale}/audio/tts_cache/{voice_dir}/{key}.mp3`
where `voice_dir` replaces `:` with `_` for filesystem safety (Dragon HD voice names).

Only styled clips are pre-cached — baseline (no-style) clips are generated
on first use by the Sample button in the Voice Cast editor.

---

## test_server.py  (code/http/)

Standard-library HTTP server (no FastAPI/Flask). Key characteristics:
- `BaseHTTPRequestHandler` + `ThreadingHTTPServer`
- `PIPE_DIR` = repo root, derived at module level from `__file__`
- Media served via `/serve_media?path=<relative-to-PIPE_DIR>` (range-request capable)
- SSE via `text/event-stream` + `sse(event, data)` helper function
- No JSON config file — constants and env vars only
- Azure TTS config from `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` env vars
- `_pipeline_status(slug, ep_id)` returns: `llm_stages`, `locales`, `voice_cast`,
  `ready_videos`, `ready_dubbed`, `story_file`

### Locales note
`_pipeline_status().locales` is derived from `AssetManifest.*.json` files
→ it is empty after Stage 0 (before Stage 5 runs).
The Voice Cast editor should derive locales from `voice_cast.characters[0]` keys
instead (filtered to exclude metadata keys).

---

## Key Conventions

| Convention | Detail |
|------------|--------|
| Script takes `--manifest` only | Output path derived from `manifest["project_id"]` / `["episode_id"]` — never `--output_dir` |
| asset_id format | `char-{name}-v{N}`, `bg-{name}-v{N}`, `sfx-{name}-v{N}` |
| Output filename | `{asset_id}.{ext}` (e.g. `char-amunhotep-v1.png`) |
| Characters are project-level | Same portrait across all episodes; saved to `projects/{slug}/characters/` |
| Manifests split at Stage 5 | `.shared.json` for GPU assets, `.{locale}.json` for TTS |
| Voice Cast is project-level | `projects/{slug}/VoiceCast.json` — append-only |
| prompts use file inlining | `run.sh` pre-embeds referenced files — no Read tool calls inside prompts |
| Concurrent stories supported | `pipeline_vars.story_N.sh` keyed by story filename |

---

## Active Work-in-Progress

| Item | Design doc | Status |
|------|-----------|--------|
| Voice Cast Editor | `/tmp/o1` (Rev 5) | Being implemented in `test_server.py` |
| `fetch_ai_assets.py` | `/tmp/p1` | Designed; not yet written |
| Azure TTS schema + Stage 0 voice casting | Previous plan | Partially implemented |

### `/tmp/o1` — Voice Cast Editor (Rev 5)
Full design for the Voice Cast editor tab in the pipeline web UI.
All implementation concerns resolved. One open issue: locales source
(see "Locales note" above in test_server.py section).

### `/tmp/p1` — fetch_ai_assets.py design
Full design doc for the AI asset client script including:
- Which manifest to send (`shared.json`), stripped payload rationale
- Already-downloaded check logic
- Complete script source code (~160 lines)
- How test_server.py should invoke it in Stage 9

---

## Environment Variables

| Variable | Used by | Default |
|----------|---------|---------|
| `AZURE_SPEECH_KEY` | gen_tts_cloud.py, test_server.py | _(required)_ |
| `AZURE_SPEECH_REGION` | gen_tts_cloud.py, test_server.py | `""` |
| `AI_SERVER_URL` | fetch_ai_assets.py | `http://192.168.86.27:8000` |
| `AI_SERVER_KEY` | fetch_ai_assets.py | `change-me` |
| `MODEL` | run.sh | _(stage defaults)_ |
| `STAGE_MODEL_N` | run.sh | _(stage defaults)_ |

---

## Test Project

- **Story**: The Pharaoh Who Defied Death (`the-pharaoh-who-defied-death`)
- **Episodes**: s01e01, s01e02
- **Locales**: en, zh-Hans
- **Characters**: char-ramesses_ka-v1, char-amunhotep-v1, char-neferet-v1,
  char-khamun-v1, char-prisoner-v1
