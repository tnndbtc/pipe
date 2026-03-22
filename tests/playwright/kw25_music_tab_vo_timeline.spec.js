// KW-25: VO timeline bars must render in #music-tl-vo on Music Tab load.
//
// Root cause: _musicLoadExisting() fetches only /api/music_timeline, which
// intentionally passes empty dicts for VOPlan to build_timeline() — so shots
// have no vo_lines field.  _musicRenderTimeline() iterates sh.vo_lines and
// gets nothing, leaving voItems = [] → _tlRender() renders 0 bars into
// #music-tl-vo.
//
// The fix: also fetch /api/vo_timeline in _musicLoadExisting() and store the
// result as _musicTimeline.vo_items, then seed voItems from tl.vo_items in
// _musicRenderTimeline().
//
// Fixture: resetKW13 — VOPlan.en.json + MusicPlan.json + music WAVs present.
// The VOPlan fixture has 9 vo_items with valid start_sec / end_sec values.
// No Generate Preview click needed — _musicLoadExisting() runs automatically
// on episode select and calls _musicRenderBody() at completion.

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW13 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW13(); });

test('KW-25: VO timeline bars render in #music-tl-vo on Music Tab load', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');

  // Sentinel: _musicRenderBody() always creates #music-tl-vo in the same innerHTML
  // string as #music-tl-music.  Waiting for it to exist means _musicLoadExisting()
  // completed ALL its awaited fetches and called _musicRenderBody().
  // _musicRenderTimeline() is synchronous inside _musicRenderBody(), so by the
  // time this sentinel fires, VO bar insertion has already run (or been skipped).
  await page.waitForFunction(
    () => document.getElementById('music-tl-vo') !== null,
    { timeout: 15000 }
  );

  const voBarCount = await page.evaluate(
    () => document.getElementById('music-tl-vo')?.children?.length ?? 0
  );

  expect(
    voBarCount,
    'VO TIMELINE MISSING: #music-tl-vo has 0 children after Music Tab load.\n' +
    'VOPlan.en.json has 9 vo_items with start_sec/end_sec — VO bars must be\n' +
    'visible in the Music Tab preview.\n' +
    'Root cause: _musicLoadExisting() never fetches /api/vo_timeline, so\n' +
    '_musicRenderTimeline() has no vo_items to render.'
  ).toBeGreaterThanOrEqual(1);
});
