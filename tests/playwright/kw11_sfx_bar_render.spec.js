// TEST COVERAGE: KW-11
// Regression: SFX tab PREVIEW AUDIO section shows no VO/SFX/Music bars after
// Generate Preview because /api/sfx_preview never writes timeline.json to disk,
// so sfxLoadTimeline() fetches a 404 and returns early without rendering bars.
const { test, expect } = require('@playwright/test');
const fs   = require('fs');
const path = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, EP_DIR } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(() => {
  resetKW2();
  // Remove any leftover SfxPreviewPack from a previous run
  const packDir = path.join(EP_DIR, 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
});

async function openSfxTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  // Wait for onSfxEpChange() to finish (status bar changes from default text)
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );
  // Force-enable the preview button — test fixture has no sfx_search_results.json
  // so _sfxLoadExisting() returns early without enabling it.
  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });
}

test('KW-11a: /api/sfx_preview returns ok:true with vo_items in timeline', async ({ page }) => {
  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);
  expect(body.timeline).toBeDefined();
  expect(body.timeline.vo_items.length).toBeGreaterThan(0);
});

test('KW-11b: VO bars render in sfx-tl-vo after Generate Preview', async ({ page }) => {
  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);

  // Wait for sfxLoadTimeline() to fetch timeline.json and sfxRenderTimeline() to run.
  // With the bug: timeline.json never written → 404 → sfxRenderTimeline() skipped → 0 bars.
  // With the fix: timeline.json written → fetch OK → bars rendered.
  await page.waitForFunction(
    () => {
      const voDiv = document.getElementById('sfx-tl-vo');
      return voDiv && voDiv.children.length > 0;
    },
    { timeout: 8000 }
  );

  const voBarCount = await page.evaluate(() => {
    const voDiv = document.getElementById('sfx-tl-vo');
    return voDiv ? voDiv.children.length : 0;
  });
  // Two VO WAVs exist in the fixture (vo-sc01-001.wav, vo-sc01-002.wav)
  expect(voBarCount).toBeGreaterThanOrEqual(2);
});

test('KW-11c: timeline.json is written to disk by /api/sfx_preview', async ({ page }) => {
  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Verify timeline.json was written to disk (required by sfxLoadTimeline)
  const tlPath = path.join(EP_DIR, 'assets', 'sfx', 'SfxPreviewPack', 'timeline.json');
  expect(fs.existsSync(tlPath)).toBe(true);
  const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
  expect(tl.total_dur_sec).toBeGreaterThan(0);
  expect(tl.vo_items.length).toBeGreaterThan(0);
});
