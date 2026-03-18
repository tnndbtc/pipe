// TEST COVERAGE: KW-14
// Regression: music does NOT stop at end_sec even though MusicPlan specifies it.
//
// Root cause:
//   media_preview_pack.py builds a delay_ms from start_sec and plays the WAV
//   for its FULL duration.  end_sec and clip_duration_sec from MusicPlan are
//   never used to trim the clip.
//
//   Fixture MusicPlan (real project settings):
//     music-sc01-sh01  start_sec=5   end_sec=20   WAV=28.089s → plays until 33.1s (13s overshoot)
//     music-sc02-sh02  start_sec=30  end_sec=35   WAV=29.559s → plays until 59.6s (24s overshoot)
//
// Detection strategy:
//   Run /api/media_preview twice — once with include_music=true, once without.
//   Extract the RMS amplitude difference between the two outputs at two windows:
//     • [31, 32]s  — INSIDE  the music window for sc02 (start=30, end=35)
//     • [36, 37]s  — OUTSIDE the music window for sc02 (after end_sec=35)
//
//   After fix  : diff_inside >> 0 (music present), diff_after ≈ 0 (music stopped at 35)
//   Before fix : diff_inside >> 0,                  diff_after >> 0 (music still playing at 36s)
//
//   Assertion: diff_after < RMS_THRESHOLD  → FAILS before fix.
//
const { test, expect } = require('@playwright/test');
const { execSync }     = require('child_process');
const fs               = require('fs');
const os               = require('os');
const path             = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW13, getEpDir }            = require('../helpers/fixture_state');

// Threshold in 16-bit PCM amplitude units.
// Music at BASE_MUSIC_DB_PREVIEW = -6 dB → amplitude ≈ 16 000.
// Silence / rounding noise stays well under 300.
const RMS_THRESHOLD = 500;

// ── Python helper: RMS amplitude of a 1-second window inside an MP4/WAV ──────
// Writes a temp script file to avoid shell quoting issues with multi-line code.
const RMS_SCRIPT = path.join(os.tmpdir(), '_kw14_rms_helper.py');
fs.writeFileSync(RMS_SCRIPT, `
import subprocess, wave, struct, math, sys

def rms_seg(p, t, dur=1.0):
    tmp = '/tmp/_kw14_rms.wav'
    r = subprocess.run(
        ['ffmpeg', '-y', '-ss', str(t), '-t', str(dur),
         '-i', p, '-ac', '1', '-ar', '44100', '-f', 'wav', tmp],
        capture_output=True)
    if r.returncode != 0:
        return 0.0
    with wave.open(tmp) as w:
        frames = w.readframes(w.getnframes())
    if not frames:
        return 0.0
    s = [struct.unpack_from('<h', frames, i * 2)[0] for i in range(len(frames) // 2)]
    return math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0

print(f'{rms_seg(sys.argv[1], float(sys.argv[2])):.2f}')
`);

function rmsAt(filePath, startSec) {
  return parseFloat(
    execSync(`python3 ${RMS_SCRIPT} ${JSON.stringify(filePath)} ${startSec}`)
      .toString().trim()
  );
}

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW13(); });

test('KW-14: music stops at end_sec=35 (not at wav_duration 59.6s)', async ({ request }) => {
  // ── Run preview WITH music ─────────────────────────────────────────────────
  const withResp = await request.post('/api/media_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selections:    {},
      include_music: true,
      include_sfx:   false,
      out_name:      'kw14_with_music.mp4',
    },
  });
  expect(withResp.ok()).toBe(true);
  const withBody = await withResp.json();
  expect(withBody.ok).toBe(true);

  // ── Run preview WITHOUT music (VO-only baseline) ───────────────────────────
  const noResp = await request.post('/api/media_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selections:    {},
      include_music: false,
      include_sfx:   false,
      out_name:      'kw14_no_music.mp4',
    },
  });
  expect(noResp.ok()).toBe(true);
  const noBody = await noResp.json();
  expect(noBody.ok).toBe(true);

  const packDir    = path.join(getEpDir(), 'assets', 'media', 'MediaPreviewPack');
  const withMp4   = path.join(packDir, 'kw14_with_music.mp4');
  const noMp4     = path.join(packDir, 'kw14_no_music.mp4');

  // ── Measure RMS amplitude at two time windows ──────────────────────────────
  // Window A: t=31s — INSIDE sc02 music window (start_sec=30, end_sec=35).
  // Sanity check: with_music should be louder than no_music here.
  const rmsWithInside = rmsAt(withMp4,  31);
  const rmsNoInside   = rmsAt(noMp4,    31);
  const diffInside    = Math.abs(rmsWithInside - rmsNoInside);

  // Window B: t=36s — OUTSIDE sc02 music window (1s after end_sec=35).
  // After fix  : music has stopped → diff ≈ 0.
  // Before fix : music still playing (WAV runs to ~59.6s) → diff ≈ 11 000+.
  const rmsWithAfter  = rmsAt(withMp4,  36);
  const rmsNoAfter    = rmsAt(noMp4,    36);
  const diffAfter     = Math.abs(rmsWithAfter - rmsNoAfter);

  // Sanity: music IS present before end_sec (ensures the fixture is correct)
  expect(diffInside).toBeGreaterThan(RMS_THRESHOLD);

  // KEY ASSERTION — FAILS before fix:
  // Music must NOT be audible after end_sec=35.
  // (Before fix: diffAfter ≈ 11 000+, so this fails.)
  expect(diffAfter).toBeLessThan(RMS_THRESHOLD);   // FAILS before fix
});
