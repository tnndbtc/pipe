// TEST COVERAGE: KW-16
// Source: prompts/regression.txt § "KW-16: SFX timing coordinate-space bug"
// Regression for: SFX entries in SfxPlan.json store episode-absolute start_sec,
//   but render_video.py uses the value directly as a shot-relative adelay.
//   For shots that don't start at t=0 in the episode, this places SFX at the
//   wrong position — or past the shot end entirely, silencing it.
//
// Concrete failure captured here:
//   Shot sc02-sh02 starts at cumulative episode offset 28.860 s (57348 ms long).
//   SFX entry: start_sec = 60.0  (episode-absolute → shot-relative should be 31.14 s)
//   Bug:   adelay = 60 000 ms  > shot duration 57 348 ms → SFX is SILENT.
//   Fixed: adelay = 31 140 ms  < shot duration             → SFX plays correctly.
//
// Strategy: run render_video.py directly with a fake ffmpeg that logs the full
//   argument list (including -filter_complex).  Find the invocation for shot
//   sc02-sh02 and assert the SFX adelay is ≤ the shot duration.

const { test, expect } = require('@playwright/test');
const { spawnSync }    = require('child_process');
const fs   = require('fs');
const path = require('path');
const os   = require('os');

// ── Constants (derived from fixture ShotList + VOPlan, must match render logic) ─
const SHOT2_CUMULATIVE_MS = 28860;   // sc01-sh01 duration_ms → cumulative start of sc02-sh02
const SHOT2_DURATION_MS   = 57348;   // sc02-sh02 duration_ms (VO-ceiling applied)
const SFX_EPISODE_ABS_SEC = 60.0;    // episode-absolute start_sec written to SfxPlan.json
const SFX_CORRECT_DELAY_MS = Math.round((SFX_EPISODE_ABS_SEC - SHOT2_CUMULATIVE_MS / 1000) * 1000);
//  = round((60.0 - 28.860) * 1000) = 31 140 ms  ← what render_video.py SHOULD emit
const SFX_BUG_DELAY_MS    = Math.round(SFX_EPISODE_ABS_SEC * 1000);
//  = 60 000 ms  ← what render_video.py currently emits (treats episode-abs as shot-rel)

// ── Paths ──────────────────────────────────────────────────────────────────────
const FIXTURE_EP = path.join(
  __dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01'
);
const RENDER_PY = path.join(__dirname, '..', '..', 'code', 'http', 'render_video.py');

// ── Fake ffmpeg — logs FULL arg list so we can inspect -filter_complex ─────────
const FAKE_FFMPEG_SRC = `#!/usr/bin/env python3
import sys, json, os, pathlib
log  = os.environ.get('FFMPEG_LOG', '')
args = sys.argv[1:]
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'args': args}) + '\\n')
if args and not args[-1].startswith('-'):
    out = pathlib.Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'\\x00' * 100)
sys.exit(0)
`;

// ── Fixture setup ──────────────────────────────────────────────────────────────
function setupEpDir(epDir) {
  fs.mkdirSync(epDir, { recursive: true });

  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(epDir, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'ShotList.json'), path.join(epDir, 'ShotList.json')
  );

  // Music WAVs — needed so render_video.py doesn't warn and bail early.
  const musicDir = path.join(epDir, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(f =>
    fs.copyFileSync(path.join(FIXTURE_EP, 'assets', 'music', f), path.join(musicDir, f))
  );

  // SFX WAV — an existing file used as the SFX clip source.
  const sfxWav = path.join(FIXTURE_EP, 'assets', 'music', 'music-sc02-sh02.wav');

  // SfxPlan.json — one entry for sc02-sh02 with EPISODE-ABSOLUTE start_sec.
  // Shot sc02-sh02 starts at 28.860 s in the episode.
  // Correct shot-relative delay = (60.0 - 28.860) * 1000 = 31 140 ms.
  // The bug: render_video.py uses 60.0 * 1000 = 60 000 ms (> shot duration 57 348 ms → silent).
  const sfxPlan = {
    timing_format: 'episode_absolute',
    sfx_entries: [
      {
        item_id:     'sfx-sc02-001',
        shot_id:     'sc02-sh02',
        source_file: sfxWav,
        start_sec:   SFX_EPISODE_ABS_SEC,   // 60.0 — episode-absolute
      },
    ],
  };
  fs.writeFileSync(path.join(epDir, 'SfxPlan.json'), JSON.stringify(sfxPlan, null, 2));
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Parse the first adelay=X value from a filter_complex string. Returns ms (int) or null. */
function parseAdelay(filterComplex) {
  const m = filterComplex.match(/adelay=(\d+)\|/);
  return m ? parseInt(m[1], 10) : null;
}

// ── Test ───────────────────────────────────────────────────────────────────────
test('KW-16: SFX adelay must use shot-relative timing, not episode-absolute', () => {
  // ── 1. Temp dirs ────────────────────────────────────────────────────────────
  const tmpRoot   = fs.mkdtempSync(path.join(os.tmpdir(), 'kw16-'));
  const epDir     = path.join(tmpRoot, 'ep');
  const binDir    = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_calls.jsonl');

  setupEpDir(epDir);

  // ── 2. Fake ffmpeg ───────────────────────────────────────────────────────────
  fs.mkdirSync(binDir);
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  // ── 3. Run render_video.py ───────────────────────────────────────────────────
  const result = spawnSync(
    'python3',
    [RENDER_PY, '--plan', path.join(epDir, 'VOPlan.en.json'), '--locale', 'en'],
    {
      env: { ...process.env, PATH: `${binDir}:${process.env.PATH}`, FFMPEG_LOG: ffmpegLog },
      encoding: 'utf8',
      timeout: 60_000,
    }
  );

  expect(
    result.status,
    `render_video.py crashed.\nstderr: ${result.stderr}`
  ).toBe(0);

  // ── 4. Find the ffmpeg invocation for shot sc02-sh02 ────────────────────────
  const logLines = fs.existsSync(ffmpegLog)
    ? fs.readFileSync(ffmpegLog, 'utf8').trim().split('\n').filter(Boolean)
    : [];

  const shot2Call = logLines
    .map(l => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean)
    .find(entry => (entry.args || []).some(a => a.includes('sc02-sh02')));

  expect(
    shot2Call,
    'Could not find any ffmpeg invocation whose output path contains "sc02-sh02" — ' +
    'render_video.py may have skipped the shot or the log was not written.'
  ).not.toBeNull();

  // ── 5. Extract -filter_complex and parse the SFX adelay ─────────────────────
  const args          = shot2Call.args;
  const fcIdx         = args.indexOf('-filter_complex');
  const filterComplex = fcIdx !== -1 ? args[fcIdx + 1] : '';

  expect(
    filterComplex,
    'No -filter_complex found in the sc02-sh02 ffmpeg invocation.'
  ).toBeTruthy();

  const adelayMs = parseAdelay(filterComplex);

  expect(
    adelayMs,
    'No adelay= found in filter_complex — SFX entry was not wired into ffmpeg at all ' +
    '(check that source_file exists and SfxPlan.json is being read).\n' +
    `filter_complex: ${filterComplex.slice(0, 400)}`
  ).not.toBeNull();

  // ── 6. Assert timing is shot-relative, not episode-absolute ─────────────────
  //
  // Bug value:   adelay = 60 000 ms  (episode-absolute used directly)
  //              60 000 > 57 348 ms (shot duration) → SFX is placed past shot end → SILENT
  //
  // Fixed value: adelay = 31 140 ms  (episode-absolute − cumulative_shot_offset)
  //              31 140 < 57 348 ms → SFX plays at the correct position

  expect(
    adelayMs,
    `SFX TIMING BUG: adelay=${adelayMs}ms but shot sc02-sh02 is only ${SHOT2_DURATION_MS}ms long — ` +
    `SFX is placed past the shot end and will be SILENT in the final render.\n` +
    `Root cause: render_video.py uses SfxPlan start_sec (${SFX_EPISODE_ABS_SEC}s) directly ` +
    `as a shot-relative delay but SfxPlan.json stores episode-absolute timestamps.\n` +
    `Fix: subtract _cumulative_shot_sec (${SHOT2_CUMULATIVE_MS / 1000}s) before converting to ms.\n` +
    `Expected adelay ≈ ${SFX_CORRECT_DELAY_MS}ms, got ${adelayMs}ms.`
  ).toBeLessThanOrEqual(SHOT2_DURATION_MS);
});
