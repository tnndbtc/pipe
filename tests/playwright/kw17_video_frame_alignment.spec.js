// TEST COVERAGE: KW-17
// Source: prompts/regression.txt § "KW-17: Video frame-alignment accumulation bug"
// Regression for: render_video.py passes raw `duration_ms / 1000.0` as the ffmpeg
//   `-t` duration flag for each shot.  ffmpeg can only encode whole video frames,
//   so the actual encoded video duration = floor(dur_sec × fps) / fps ≤ dur_sec.
//   The shortfall (up to 1/fps ≈ 41.67 ms per shot) accumulates across all shots.
//
// Concrete failure:
//   2 fixture shots: sc01-sh01 (28860 ms) + sc02-sh02 (57348 ms)
//     sc01-sh01: 28.860 s × 24 = 692.64 frames → 692 encoded → 28.833 s  (−27 ms)
//     sc02-sh02: 57.348 s × 24 = 1376.352 frames → 1376 encoded → 57.333 s (−15 ms)
//   For a production episode with ~72 shots the shortfall reaches ~3 s:
//     video track ends at 52 s while the container (audio) reports 55 s total.
//
// Fix: snap dur_sec to the nearest video frame before passing to `-t`:
//   dur_sec_snapped = round(dur_ms * fps / 1000) / fps
//   Then floor(dur_sec_snapped × fps) / fps == dur_sec_snapped exactly.
//
// Strategy: run render_video.py directly with a fake ffmpeg that logs the full
//   argument list.  For every per-shot MKV invocation find the `-t` value and
//   assert it equals round(t × fps) / fps (i.e. it is frame-aligned).

const { test, expect } = require('@playwright/test');
const { spawnSync }    = require('child_process');
const fs   = require('fs');
const path = require('path');
const os   = require('os');

// ── Constants ─────────────────────────────────────────────────────────────────
const FPS = 24;   // must match FPS constant in render_video.py

// ── Paths ──────────────────────────────────────────────────────────────────────
const FIXTURE_EP = path.join(
  __dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01'
);
const RENDER_PY = path.join(__dirname, '..', '..', 'code', 'http', 'render_video.py');

// ── Fake ffmpeg — logs FULL arg list for every invocation ─────────────────────
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

  // Music WAVs — needed so render_video.py doesn't bail out at music-path check.
  const musicDir = path.join(epDir, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(f =>
    fs.copyFileSync(path.join(FIXTURE_EP, 'assets', 'music', f), path.join(musicDir, f))
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Given a `-t` value (seconds, float), return true if it is frame-aligned:
 *   round(t × fps) / fps == t  (within a 0.1 ms tolerance)
 */
function isFrameAligned(t, fps) {
  const snapped = Math.round(t * fps) / fps;
  return Math.abs(snapped - t) < 1e-4;   // 0.1 ms tolerance
}

/**
 * From a logged ffmpeg args array, return the value after `-t` if present, else null.
 */
function extractT(args) {
  // Use lastIndexOf: the output -t is always the LAST -t in the arg list.
  // Earlier -t flags are input loop budgets (e.g. -loop 1 -t X for image segments)
  // and do not represent the encoded video duration.
  const idx = args.lastIndexOf('-t');
  if (idx === -1 || idx + 1 >= args.length) return null;
  return parseFloat(args[idx + 1]);
}

/**
 * Return true when the ffmpeg call is a per-shot MKV render (not black_frame, not concat).
 * Heuristics: has a `-t` flag AND the output path ends with `.mkv`
 *             AND does NOT end with `black_frame.mkv` (that uses lavfi -f, no -t).
 */
function isPerShotCall(args) {
  if (!args.includes('-t')) return false;
  const outPath = args[args.length - 1] || '';
  return outPath.endsWith('.mkv') && !outPath.endsWith('black_frame.mkv');
}

// ── Tests ──────────────────────────────────────────────────────────────────────

// KW-17b fixture — minimal two-shot episode where sc01's VO overruns its
// ShotList boundary, triggering the stale-start_sec duration inflation bug.
//
// ShotList:
//   sc01-sh01  duration_sec=10.0   start_sec=0  (implicit)
//   sc02-sh02  duration_sec=15.0   start_sec=10.0
//
// VOPlan VO items (episode-absolute):
//   vo-sc01-001  end_sec=18.0  →  overruns sc01 (10.0 s boundary) by 8 s
//   vo-sc02-001  end_sec=28.0
//
// Expected durations (correct):
//   sc01: last_ms=(18.0−0.0)×1000=18000, tail=2000  → max(20000,10000) = 20000 ms
//         _cumulative_shot_sec after sc01 = round(20000×24/1000)/24 = 480/24 = 20.0 s
//   sc02: last_ms=(28.0−20.0)×1000= 8000, derived_tail=15000−8000=7000
//                                          → max(8000+7000,15000) = 15000 ms  (base wins)
//   boundary (sc01→sc02, different scene_id): round(1000/24) = 42 ms
//   CORRECT total: 20000 + 42 + 15000 = 35042 ms  (≈ 0:35)
//
// Bug path (what render_video.py actually does):
//   sc02: last_ms=(28.0−10.0)×1000=18000  ← stale ShotList.start_sec used as origin
//         tail=2000  → max(18000+2000,15000) = 20000 ms  (floor wrongly fires)
//   BUGGY  total: 20000 + 42 + 20000 = 40042 ms  (≈ 0:40)
//   Excess: 5000 ms
const KW17B_SHOTLIST = {
  schema_id: 'ShotList', schema_version: '1.0.0',
  shotlist_id: 'kw17b-test',
  shots: [
    {
      shot_id: 'sc01-sh01', scene_id: 'sc01',
      duration_sec: 10.0, characters: [],
      background_id: 'bg-placeholder',
      audio_intent: { vo_item_ids: ['vo-sc01-001'] },
    },
    {
      shot_id: 'sc02-sh02', scene_id: 'sc02',
      duration_sec: 15.0, start_sec: 10.0, characters: [],
      background_id: 'bg-placeholder',
      audio_intent: { vo_item_ids: ['vo-sc02-001'] },
    },
  ],
};

const KW17B_VOPLAN = {
  schema_id: 'VOPlan', schema_version: '1.0.0',
  manifest_id: 'kw17b-test-en', locale: 'en',
  shotlist_ref: 'ShotList.json',
  vo_items: [
    // sc01 VO ends at 18.0 s — 8 s past sc01's 10.0 s ShotList boundary.
    // This forces sc01 to be floor-extended to 20 000 ms, shifting sc02's
    // actual render start from 10.0 s to 20.0 s.
    { item_id: 'vo-sc01-001', speaker_id: 'narrator', text: 'Line one.',
      start_sec: 0.0, end_sec: 18.0 },
    // sc02 VO ends at 28.0 s (episode-absolute).
    // Relative to ShotList sc02 start (10.0):  28.0−10.0 = 18.0 s → last_ms=18 000 (BUGGY)
    // Relative to actual render sc02 start (20.0): 28.0−20.0 =  8.0 s → last_ms= 8 000 (CORRECT)
    { item_id: 'vo-sc02-001', speaker_id: 'narrator', text: 'Line two.',
      start_sec: 20.0, end_sec: 28.0 },
  ],
  resolved_assets: [],
};

test('KW-17b: sc02 duration must not be inflated when sc01 is VO-floor-extended', () => {
  // ── 1. Temp dirs ─────────────────────────────────────────────────────────────
  const tmpRoot   = fs.mkdtempSync(path.join(os.tmpdir(), 'kw17b-'));
  const epDir     = path.join(tmpRoot, 'ep');
  const binDir    = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_calls.jsonl');

  fs.mkdirSync(epDir,  { recursive: true });
  fs.mkdirSync(binDir, { recursive: true });

  fs.writeFileSync(path.join(epDir, 'ShotList.json'),    JSON.stringify(KW17B_SHOTLIST));
  fs.writeFileSync(path.join(epDir, 'VOPlan.en.json'),   JSON.stringify(KW17B_VOPLAN));

  // ── 2. Fake ffmpeg ─────────────────────────────────────────────────────────
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  // ── 3. Run render_video.py ─────────────────────────────────────────────────
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
    `render_video.py crashed.\nstderr: ${result.stderr}\nstdout: ${result.stdout}`
  ).toBe(0);

  // ── 4. Read render_output.json ─────────────────────────────────────────────
  const renderOutputPath = path.join(epDir, 'renders', 'en', 'render_output.json');
  expect(
    fs.existsSync(renderOutputPath),
    `render_output.json not written to ${renderOutputPath}`
  ).toBe(true);

  const renderOutput = JSON.parse(fs.readFileSync(renderOutputPath, 'utf8'));
  const actualMs     = renderOutput.total_duration_ms;

  // ── 5. Assert correct total duration ──────────────────────────────────────
  //
  // Correct calculation (both shots, 1 scene-boundary frame @ 24 fps):
  //   sc01: last_ms=18000, tail=2000  → duration_ms = 20000
  //   sc02: last_ms= 8000, tail=7000  → duration_ms = 15000  (base wins)
  //   boundary: round(1000/24) = 42 ms
  //   CORRECT total = 20000 + 42 + 15000 = 35042 ms
  //
  // Bug produces:
  //   sc02: last_ms=18000 (uses stale ShotList.start_sec=10.0 instead of
  //         _cumulative_shot_sec=20.0) → tail=2000 → duration_ms = 20000
  //   BUGGY total = 20000 + 42 + 20000 = 40042 ms  (+5000 ms excess)
  //
  const CORRECT_TOTAL_MS = 35042;
  const BUGGY_TOTAL_MS   = 40042;

  expect(
    actualMs,
    `KW-17b FAIL: total_duration_ms = ${actualMs} ms.\n` +
    `Expected ${CORRECT_TOTAL_MS} ms (≈ 0:35).\n` +
    `Got      ${actualMs} ms (≈ 0:${Math.floor(actualMs/1000)}).\n\n` +
    `Root cause: build_shot_plan() computes last_ms for sc02 using\n` +
    `ShotList.start_sec (10.0 s) as the shot origin, but sc01 was\n` +
    `floor-extended from 10 000 ms to 20 000 ms, shifting sc02's actual\n` +
    `render start to 20.0 s.  The stale origin inflates last_ms from\n` +
    `8 000 ms to 18 000 ms, wrongly triggering the VO-floor formula on\n` +
    `sc02 and adding ${BUGGY_TOTAL_MS - CORRECT_TOTAL_MS} ms of extra duration.\n` +
    `(If actualMs === ${BUGGY_TOTAL_MS} the original bug is still present.)`
  ).toBe(CORRECT_TOTAL_MS);
});

test('KW-17: per-shot -t duration must be frame-aligned (multiple of 1/fps)', () => {
  // ── 1. Temp dirs ─────────────────────────────────────────────────────────────
  const tmpRoot   = fs.mkdtempSync(path.join(os.tmpdir(), 'kw17-'));
  const epDir     = path.join(tmpRoot, 'ep');
  const binDir    = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_calls.jsonl');

  setupEpDir(epDir);

  // ── 2. Fake ffmpeg ─────────────────────────────────────────────────────────
  fs.mkdirSync(binDir);
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  // ── 3. Run render_video.py ─────────────────────────────────────────────────
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

  // ── 4. Parse per-shot ffmpeg calls ─────────────────────────────────────────
  const logLines = fs.existsSync(ffmpegLog)
    ? fs.readFileSync(ffmpegLog, 'utf8').trim().split('\n').filter(Boolean)
    : [];

  const allCalls = logLines
    .map(l => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean);

  const perShotCalls = allCalls.filter(entry => isPerShotCall(entry.args || []));

  expect(
    perShotCalls.length,
    'No per-shot MKV ffmpeg calls were logged — render_video.py may have skipped ' +
    'all shots or the log was not written. Check that FFMPEG_LOG is set and the ' +
    'fake ffmpeg is on PATH.'
  ).toBeGreaterThan(0);

  // ── 5. Assert each -t value is frame-aligned ───────────────────────────────
  //
  // Bug:   dur_sec = dur_ms / 1000.0  →  e.g. 28860 ms → 28.860 s
  //        28.860 × 24 = 692.64 frames → ffmpeg encodes 692 → 28.8333 s actual
  //        shortfall = 27 ms per shot; across 72 shots ≈ 3 s gap in final render
  //
  // Fixed: dur_sec = round(dur_ms × fps / 1000) / fps
  //        e.g. round(28860 × 24 / 1000) / 24 = round(692.64) / 24 = 693/24 = 28.875 s
  //        28.875 × 24 = 693.000 frames — exact, no shortfall
  //
  const misaligned = [];
  for (const entry of perShotCalls) {
    const tVal    = extractT(entry.args);
    const outPath = entry.args[entry.args.length - 1] || '(unknown)';
    if (tVal === null) continue;
    if (!isFrameAligned(tVal, FPS)) {
      const framesRaw    = tVal * FPS;
      const framesActual = Math.floor(framesRaw);
      const shortfallMs  = Math.round((framesRaw - framesActual) / FPS * 1000);
      misaligned.push(
        `  shot ${path.basename(outPath)}: -t ${tVal} s → ${framesRaw.toFixed(4)} frames ` +
        `→ only ${framesActual} encoded → shortfall ${shortfallMs} ms`
      );
    }
  }

  expect(
    misaligned.length,
    'FRAME ALIGNMENT BUG: The following per-shot -t values are not multiples of ' +
    `1/fps (1/${FPS} = ${(1/FPS).toFixed(6)} s).\n` +
    'Root cause: render_video.py uses dur_ms / 1000.0 directly as -t, but ffmpeg\n' +
    'can only encode whole frames. The fractional frame shortfall accumulates across\n' +
    'all shots and causes the final video track to end seconds before the audio track.\n' +
    'Fix: snap to nearest frame — dur_sec = round(dur_ms * fps / 1000) / fps\n\n' +
    misaligned.join('\n')
  ).toBe(0);
});
