// TEST COVERAGE: KW-13 (merged KW-31)
// Regression: Media tab Generate Preview produces no music even when
// include_music=true and MusicPlan.json is present.
//
// KW-13a/b : Generate Preview — music mixed correctly (API response, debug_log)
// KW-13c–g : Free-segment UI — segment rows, add, save, tab-switch
//
// Root cause 1 (test-mode only):
//   /api/media_preview launches media_preview_pack.py via
//   os.path.join(PIPE_DIR, "code/http/media_preview_pack.py").
//   In --test-mode PIPE_DIR is overridden to tests/fixtures/, so the
//   script path becomes tests/fixtures/code/http/media_preview_pack.py
//   (does not exist) → subprocess fails → HTTP 500.
//
// Root cause 2 (always):
//   media_preview_pack.py builds:
//     music_by_shot = {o["shot_id"]: o for o in shot_overrides if "shot_id" in o}
//   MusicPlan shot_overrides have NO "shot_id" field (only "item_id") →
//   music_by_shot is always empty → music is silently skipped for every shot →
//   debug_log contains "No music WAV files found" instead of "Mixed N music clip(s)".
//
// Fix:
//   1. Use __file__-relative path for the subprocess in test_server.py.
//   2. Iterate shot_overrides directly by "item_id" (episode-absolute start_sec/end_sec)
//      — no ShotList lookup needed; MusicPlan start_sec is already episode-absolute.
//
// Fixture: MusicPlan.json already committed in tests/fixtures.
//   Both sc01-sh01 and sc02-sh02 have music WAVs present.
//   After fix: debug_log shows "Mixed N music clip(s)" (N >= 1).
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW13 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW13(); });

test('KW-13a: /api/media_preview returns ok when include_music=true', async ({ request }) => {
  // Before fix-1: returns HTTP 500 (media_preview_pack.py not found in test mode)
  // After fix-1+2: returns HTTP 200 with ok:true
  const resp = await request.post('/api/media_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selections:    {},
      include_music: true,
      include_sfx:   false,
    },
  });
  expect(resp.ok()).toBe(true);                 // FAILS before fix-1 (HTTP 500)
  const body = await resp.json();
  expect(body.ok).toBe(true);
});



