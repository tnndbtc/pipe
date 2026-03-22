// KW-24: Music bar must render in #media-tl-music after Generate Preview.
//
// Root cause: _mediaLoadTimeline() does not populate music_items, so
// _tlRender() renders 0 bars into #media-tl-music.
//
// This test checks the end-user visible result (DOM bars) and the API
// response, regardless of which endpoint provides music_items.

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW13 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW13(); });

test('KW-24: music bar renders in #media-tl-music after Generate Preview', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');
  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );

  // Click Generate Preview and wait for it to complete
  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);
  const previewBody = await previewResp.json();
  expect(previewBody.ok, `media_preview failed: ${JSON.stringify(previewBody)}`).toBe(true);

  // Poll for #media-tl-music to have ≥1 child.
  // _mediaLoadTimeline() is called without await, so status "Preview ready" fires
  // before _tlRender() completes — we must poll, not rely on status text.
  // Wait up to 10s, then assert — gives a clear failure message either way.
  await page.waitForTimeout(10000);

  const barCount = await page.evaluate(
    () => document.getElementById('media-tl-music')?.children?.length ?? 0
  );
  expect(
    barCount,
    'MUSIC BAR MISSING: #media-tl-music has 0 children after Generate Preview.\n' +
    'MusicPlan.json has 2 shot_overrides — music bar must be visible after preview.'
  ).toBeGreaterThanOrEqual(1);
});
