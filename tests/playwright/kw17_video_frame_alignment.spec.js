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
  const idx = args.indexOf('-t');
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

// ── Test ───────────────────────────────────────────────────────────────────────
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
