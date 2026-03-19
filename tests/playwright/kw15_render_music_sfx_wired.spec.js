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
const RENDER_PY = path.join(__dirname, '..', '..', 'code', 'http', 'render_video.py');

// ── Fake ffmpeg source ─────────────────────────────────────────────────────────
// A Python shebang script (portable).  It:
//   1. Appends all -i inputs and the -filter_complex value to $FFMPEG_LOG as a JSON line.
//   2. Creates the output file (always sys.argv[-1] in render_video.py calls).
//   3. Exits 0 so render_video.py continues through all shots and the concat pass.
const FAKE_FFMPEG_SRC = `#!/usr/bin/env python3
import sys, json, os, pathlib
log  = os.environ.get('FFMPEG_LOG', '')
args = sys.argv[1:]
inputs = []
filter_complex = ''
i = 0
while i < len(args):
    if args[i] == '-i' and i + 1 < len(args):
        inputs.append(args[i + 1])
        i += 2
    elif args[i] == '-filter_complex' and i + 1 < len(args):
        filter_complex = args[i + 1]
        i += 2
    else:
        i += 1
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'inputs': inputs, 'filter_complex': filter_complex}) + '\\n')
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

  fs.copyFileSync(
    path.join(FIXTURE_EP, 'MusicPlan.json'),
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

// ── KW-15b: cut-clip volume_db is applied in the ffmpeg filter ─────────────────
// Regression for: SfxPlan v1.2 moves clip volume from sfx_entries[].clip_volume_db
// to cut_clips[].volume_db.  render_video.py must inject it into clip_volume_db
// before building the ffmpeg filter.  If the injection is missing, render_video.py
// reads clip_volume_db=0 and the filter will contain volume=0.707946 (SFX_DB=-3 only)
// instead of volume=2.818383 (SFX_DB=-3 + clip_volume_db=+12).
test('KW-15b: cut-clip volume_db flows into ffmpeg volume= filter', () => {
  const SFX_DB       = -3.0;
  const CLIP_VOL_DB  = 12.0;
  // Expected amplitude: 10^((SFX_DB + CLIP_VOL_DB) / 20) = 10^(9/20)
  const expectedAmp  = Math.pow(10, (SFX_DB + CLIP_VOL_DB) / 20);
  // render_video.py formats with :.6f  →  "2.818383"
  const expectedStr  = `volume=${expectedAmp.toFixed(6)}`;

  // ── 1. Temp dirs ──────────────────────────────────────────────────────────
  const tmpRoot   = fs.mkdtempSync(path.join(os.tmpdir(), 'kw15b-'));
  const epDir     = path.join(tmpRoot, 'ep');
  const binDir    = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_inputs.jsonl');

  // ── 2. Episode dir ────────────────────────────────────────────────────────
  fs.mkdirSync(epDir, { recursive: true });
  fs.copyFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'),  path.join(epDir, 'VOPlan.en.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'ShotList.json'),   path.join(epDir, 'ShotList.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'MusicPlan.json'),  path.join(epDir, 'MusicPlan.json'));

  const musicDir = path.join(epDir, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  fs.copyFileSync(path.join(FIXTURE_EP, 'assets', 'music', 'music-sc01-sh01.wav'), path.join(musicDir, 'music-sc01-sh01.wav'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'assets', 'music', 'music-sc02-sh02.wav'), path.join(musicDir, 'music-sc02-sh02.wav'));

  // Cut-clip WAV: place the fixture SFX source as the pre-trimmed clip.
  // render_video.py calls sp_path.exists() on clip_path — it must exist.
  const sfxItemDir = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001');
  fs.mkdirSync(sfxItemDir, { recursive: true });
  const cutClipRel = 'assets/sfx/sfx-sc01-sh01-001/cut-clip.wav';
  const cutClipAbs = path.join(epDir, cutClipRel);
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'assets', 'sfx', 'sfx_source_fixture.wav'),
    cutClipAbs
  );

  // SfxPlan v1.2: volume lives on cut_clips[].volume_db, NOT on sfx_entries.
  //
  // Two entries deliberately chosen:
  //   Entry 1 (sfx-sc01-sh01-001): end_sec - start_sec == clip_duration (2.0s)
  //            → atrim not strictly needed, volume/adelay regression is the focus.
  //   Entry 2 (sfx-sc01-sh01-002): end_sec - start_sec == 1.0s, but the WAV is
  //            5.0s long. render_video.py MUST emit atrim=duration=1.000 to honour
  //            end_sec. Bug: sp_end=None for clip_path entries → atrim never emitted
  //            → full 5.0s clip plays past the designated stop point.
  const sfxPlan = {
    schema_id:      'SfxPlan',
    schema_version: '1.2',
    timing_format:  'episode_absolute',
    sfx_entries: [
      {
        item_id:     'sfx-sc01-sh01-001',
        shot_id:     'sc01-sh01',
        source_file: cutClipAbs,
        start_sec:   1.0,
        end_sec:     3.0,
        volume_db:   0.0,
        duck_db:     0.0,
        fade_sec:    0.0,
        clip_id:     'cut-clip',
        clip_path:   cutClipRel,
      },
      {
        // Entry 2: wants only 1.0s of a 5.0s clip — end_sec MUST be honoured.
        item_id:     'sfx-sc01-sh01-002',
        shot_id:     'sc01-sh01',
        source_file: cutClipAbs,
        start_sec:   4.0,     // episode-absolute placement
        end_sec:     5.0,     // stop after 1.0s — clip is 5.0s, so atrim is essential
        volume_db:   0.0,
        duck_db:     0.0,
        fade_sec:    0.0,
        clip_id:     'cut-clip-2',
        clip_path:   cutClipRel,   // same 5.0s WAV file
      },
    ],
    cut_clips: [
      {
        clip_id:      'cut-clip',
        item_id:      'sfx-sc01-sh01-001',
        start_sec:    1.0,
        end_sec:      3.0,
        duration_sec: 2.0,
        source_file:  cutClipAbs,
        path:         cutClipRel,
        volume_db:    CLIP_VOL_DB,   // 12 dB — must reach ffmpeg filter
      },
      {
        clip_id:      'cut-clip-2',
        item_id:      'sfx-sc01-sh01-002',
        start_sec:    4.0,
        end_sec:      5.0,
        duration_sec: 5.0,   // full source WAV — longer than the 1.0s we want
        source_file:  cutClipAbs,
        path:         cutClipRel,
        volume_db:    0.0,
      },
    ],
    cut_assign: {
      'sfx-sc01-sh01-001': 'cut-clip',
      'sfx-sc01-sh01-002': 'cut-clip-2',
    },
  };
  fs.writeFileSync(path.join(epDir, 'SfxPlan.json'), JSON.stringify(sfxPlan, null, 2));

  // ── 3. Fake ffmpeg ────────────────────────────────────────────────────────
  fs.mkdirSync(binDir);
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  // ── 4. Run render_video.py ────────────────────────────────────────────────
  const result = spawnSync(
    'python3',
    [RENDER_PY, '--plan', path.join(epDir, 'VOPlan.en.json'), '--locale', 'en'],
    {
      env: { ...process.env, PATH: `${binDir}:${process.env.PATH}`, FFMPEG_LOG: ffmpegLog },
      encoding: 'utf8',
      timeout:  60_000,
    }
  );
  expect(result.status, `render_video.py crashed.\nstderr: ${result.stderr}`).toBe(0);

  // ── 5. Collect all filter_complex strings from every shot-render ffmpeg call ─
  const logLines = fs.existsSync(ffmpegLog)
    ? fs.readFileSync(ffmpegLog, 'utf8').trim().split('\n').filter(Boolean)
    : [];
  const allFilters = logLines
    .flatMap(l => { try { return [JSON.parse(l).filter_complex]; } catch { return []; } })
    .filter(Boolean)
    .join('\n');

  // ── 6. Assert volume= matches 10^((SFX_DB + CLIP_VOL_DB) / 20) ───────────
  expect(
    allFilters,
    `SFX clip volume NOT applied in ffmpeg filter.\n` +
    `Expected filter to contain "${expectedStr}" (SFX_DB=${SFX_DB} + clip_volume_db=${CLIP_VOL_DB}).\n` +
    `Without the fix, render_video.py reads clip_volume_db=0 and emits ` +
    `volume=${Math.pow(10, SFX_DB / 20).toFixed(6)} instead.\n` +
    `Root cause: cut_clips[].volume_db not injected into sfx_entries before render.`
  ).toContain(expectedStr);

  // ── 7. Assert adelay= honours start_sec from sfx_entries ─────────────────
  // sfx_entry.start_sec = 1.0 (episode-absolute); shot sc01-sh01 starts at
  // episode t=0.0 → correct shot-relative delay = (1.0 - 0.0) * 1000 = 1000 ms.
  // Bug: when clip_path is present render_video.py hard-codes sp_start = 0.0,
  // discarding start_sec → delay_ms = 0 → adelay=0|0 instead of adelay=1000|1000.
  const SFX_START_SEC    = 1.0;   // sfx_entries[0].start_sec
  const SHOT_OFFSET_SEC  = 0.0;   // sc01-sh01 is the first shot, starts at t=0
  const expectedDelayMs  = Math.round((SFX_START_SEC - SHOT_OFFSET_SEC) * 1000);
  const expectedDelay    = `adelay=${expectedDelayMs}|${expectedDelayMs}`;
  expect(
    allFilters,
    `SFX cut-clip start_sec NOT honoured — fires at shot start instead.\n` +
    `Expected filter to contain "${expectedDelay}" ` +
    `(start_sec=${SFX_START_SEC} - shot_offset=${SHOT_OFFSET_SEC}) * 1000 = ${expectedDelayMs} ms).\n` +
    `Bug: render_video.py sets sp_start = 0.0 when clip_path is present, ` +
    `discarding sfx_entries[].start_sec entirely.\n` +
    `Produces adelay=0|0 instead of adelay=${expectedDelayMs}|${expectedDelayMs}.`
  ).toContain(expectedDelay);

  // ── 8. Assert atrim= honours end_sec for cut-clip entries ─────────────────
  // sfx_entries[1]: start_sec=4.0, end_sec=5.0 → must trim to 1.0s.
  // The clip WAV is 5.0s long; without atrim the full 5.0s plays past end_sec.
  //
  // Bug: render_video.py always sets sp_end=None when clip_path is present
  // (same branch that previously set sp_start=0.0), so the
  //   if sp_end_ep is not None:
  //       filt += ",atrim=duration=..."
  // block is never reached for cut clips, and the designated end point is silently
  // discarded — the SFX overshoots into the next shot or beyond the episode.
  const SFX2_START_SEC   = 4.0;
  const SFX2_END_SEC     = 5.0;
  const expectedTrimDur  = (SFX2_END_SEC - SFX2_START_SEC).toFixed(3);   // "1.000"
  const expectedTrim     = `atrim=duration=${expectedTrimDur}`;
  expect(
    allFilters,
    `SFX cut-clip end_sec NOT honoured — clip plays ${5.0 - (SFX2_END_SEC - SFX2_START_SEC)}s beyond the designated stop point.\n` +
    `Expected filter to contain "${expectedTrim}" ` +
    `(end_sec=${SFX2_END_SEC} - start_sec=${SFX2_START_SEC} = ${expectedTrimDur}s trim).\n` +
    `Clip is 5.0s long; without atrim the full clip plays past end_sec=${SFX2_END_SEC}.\n` +
    `Bug: render_video.py sets sp_end=None when clip_path is present,\n` +
    `discarding sfx_entries[].end_sec entirely.`
  ).toContain(expectedTrim);
});
