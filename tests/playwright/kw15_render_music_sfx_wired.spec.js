// TEST COVERAGE: KW-15
// Source: prompts/regression.txt § "KW-15: render_video.py — Music and SFX wired into ffmpeg"
// Regression for: final output.mp4 has no music or SFX because render_video.py
//   never passes MusicPlan / SfxPlan audio files as -i inputs to ffmpeg.
//
// Root causes captured by this test:
//   MUSIC — _music_plan_overrides dict build fails: `{o["shot_id"]: o ...}` throws
//            KeyError because shot_overrides in MusicPlan.json store "item_id" only
//            (no "shot_id"). Exception is caught silently → overrides = {} →
//            music_asset_id = None for all shots → music WAV never reaches ffmpeg.
//   SFX   — SfxPlan.json absent at render time → _sfx_plan_by_shot = {} →
//            sfx_plan_override = None → fallback to shot["sfx_plan_entries"] = [] →
//            SFX WAV never reaches ffmpeg.
//
// Strategy: run render_video.py directly (no HTTP server) with a fake ffmpeg that
//   logs every -i input path. Assert that a music WAV and a dedicated SFX WAV appear
//   in the captured inputs.  Both assertions FAIL with current code.

const { test, expect } = require('@playwright/test');
const { spawnSync }    = require('child_process');
const fs   = require('fs');
const path = require('path');
const os   = require('os');

// ── Paths ──────────────────────────────────────────────────────────────────────
const FIXTURE_EP = path.join(
  __dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01'
);
const STEP_OUT  = path.join(__dirname, '..', 'step_outputs');
const RENDER_PY = path.join(__dirname, '..', '..', 'code', 'http', 'render_video.py');

// ── Fake ffmpeg source ─────────────────────────────────────────────────────────
// A Python shebang script (portable).  It:
//   1. Appends all -i inputs to $FFMPEG_LOG as a JSON line.
//   2. Creates the output file (always sys.argv[-1] in render_video.py calls).
//   3. Exits 0 so render_video.py continues through all shots and the concat pass.
const FAKE_FFMPEG_SRC = `#!/usr/bin/env python3
import sys, json, os, pathlib
log  = os.environ.get('FFMPEG_LOG', '')
args = sys.argv[1:]
inputs = []
i = 0
while i < len(args):
    if args[i] == '-i' and i + 1 < len(args):
        inputs.append(args[i + 1])
        i += 2
    else:
        i += 1
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'inputs': inputs}) + '\\n')
if args and not args[-1].startswith('-'):
    out = pathlib.Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'\\x00' * 100)
sys.exit(0)
`;

// ── Fixture setup ──────────────────────────────────────────────────────────────
function setupEpDir(epDir, sfxWavSrc) {
  fs.mkdirSync(epDir, { recursive: true });

  // Core plan files
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(epDir, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'ShotList.json'), path.join(epDir, 'ShotList.json')
  );

  // MusicPlan.json — intentionally uses the step_outputs version which has NO
  // "shot_id" in shot_overrides.  This replicates the production bug: the dict
  // comprehension {o["shot_id"]: o ...} throws KeyError, is swallowed by
  // try/except, and _music_plan_overrides stays {}.
  fs.copyFileSync(
    path.join(STEP_OUT, 'music_review.MusicPlan.json'),
    path.join(epDir, 'MusicPlan.json')
  );

  // Music WAVs — render_video.py resolves these via
  //   assets/music/{music_asset_id}.wav (or .loop.wav)
  const musicDir = path.join(epDir, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'assets', 'music', 'music-sc01-sh01.wav'),
    path.join(musicDir, 'music-sc01-sh01.wav')
  );
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'assets', 'music', 'music-sc02-sh02.wav'),
    path.join(musicDir, 'music-sc02-sh02.wav')
  );

  // SfxPlan.json — one confirmed SFX entry pointing at sfxWavSrc (a real file).
  // render_video.py checks sp_path.exists() before adding to ffmpeg -i, so the
  // source_file must be an absolute path to an existing WAV.
  const sfxPlan = {
    timing_format: 'episode_absolute',
    sfx_entries: [
      {
        item_id:     'sfx-sc01-001',
        shot_id:     'sc01-sh01',
        source_file: sfxWavSrc,
        start_sec:   0.5,
        end_sec:     2.5,
      },
    ],
  };
  fs.writeFileSync(path.join(epDir, 'SfxPlan.json'), JSON.stringify(sfxPlan, null, 2));
}

// ── Test ───────────────────────────────────────────────────────────────────────
test('KW-15: music and SFX WAVs must reach ffmpeg — both are missing in current build', () => {
  // ── 1. Temp dirs ────────────────────────────────────────────────────────────
  const tmpRoot  = fs.mkdtempSync(path.join(os.tmpdir(), 'kw15-'));
  const epDir    = path.join(tmpRoot, 'ep');
  const binDir   = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_inputs.jsonl');

  // SFX WAV: fixture WAV used as a stand-in SFX clip in SfxPlan.json.
  // Must be an absolute path to an existing file (render_video.py calls exists() on it).
  const sfxWavSrc = path.join(FIXTURE_EP, 'assets', 'music', 'music-sc01-sh01.wav');

  setupEpDir(epDir, sfxWavSrc);

  // ── 2. Fake ffmpeg ───────────────────────────────────────────────────────────
  fs.mkdirSync(binDir);
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  // ── 3. Run render_video.py ───────────────────────────────────────────────────
  const voplanPath = path.join(epDir, 'VOPlan.en.json');
  const result = spawnSync(
    'python3',
    [RENDER_PY, '--plan', voplanPath, '--locale', 'en'],
    {
      env: {
        ...process.env,
        PATH:       `${binDir}:${process.env.PATH}`,
        FFMPEG_LOG: ffmpegLog,
      },
      encoding: 'utf8',
      timeout:  60_000,
    }
  );

  // Fail fast if render_video.py itself crashed
  expect(
    result.status,
    `render_video.py exited ${result.status}.\nstderr: ${result.stderr}`
  ).toBe(0);

  // ── 4. Parse captured ffmpeg -i inputs ──────────────────────────────────────
  const logLines  = fs.existsSync(ffmpegLog)
    ? fs.readFileSync(ffmpegLog, 'utf8').trim().split('\n').filter(Boolean)
    : [];
  const allInputs = logLines.flatMap(line => {
    try { return JSON.parse(line).inputs; } catch { return []; }
  });

  const hasMusicSh01 = allInputs.some(p => p.includes('music-sc01-sh01.wav'));
  const hasMusicSh02 = allInputs.some(p => p.includes('music-sc02-sh02.wav'));
  const hasSfx       = allInputs.some(p => p === sfxWavSrc);

  // ── 5. Assertions — BOTH FAIL with current code ─────────────────────────────

  expect(hasMusicSh01,
    'MUSIC NOT WIRED: music-sc01-sh01.wav was not passed to ffmpeg.\n' +
    'Root cause: MusicPlan.json shot_overrides lack "shot_id" so the dict\n' +
    'comprehension {o["shot_id"]: o} throws KeyError, caught silently, leaving\n' +
    '_music_plan_overrides = {} → music_asset_id = None on every shot → music WAV\n' +
    'path is never resolved → ffmpeg never receives a music -i input.'
  ).toBe(true);

  expect(hasMusicSh02,
    'MUSIC NOT WIRED: music-sc02-sh02.wav was not passed to ffmpeg.\n' +
    'Same root cause as sc01-sh01 — all shot overrides are dropped.'
  ).toBe(true);

  expect(hasSfx,
    'SFX NOT WIRED: SFX WAV was not passed to ffmpeg.\n' +
    'Root cause: SfxPlan.json is absent at render time (written only after\n' +
    'SFX tab is confirmed, but render runs before that gate) →\n' +
    '_sfx_plan_by_shot = {} → {} or None = None → sfx_plan_override = None →\n' +
    'fallback to shot["sfx_plan_entries"] = [] → no SFX reaches ffmpeg.'
  ).toBe(true);
});
