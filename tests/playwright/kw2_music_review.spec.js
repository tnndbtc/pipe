// TEST COVERAGE: KW-2
// Source: prompts/regression.txt § "KW-2: Music Review Preview"
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, musicplan } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW2(); });

async function openMusicTab(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  // Wait for Generate Preview button to become enabled
  await page.waitForFunction(
    () => !document.getElementById('music-btn-review').disabled,
    { timeout: 12000 }
  );
}

test('KW-2a: Generate Preview fires music_review_pack and gets 200', async ({ page }) => {
  await openMusicTab(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/music_review_pack'), { timeout: 15000 }),
    page.click('#music-btn-review'),
  ]);
  if (resp.status() !== 200) {
    const errBody = await resp.json().catch(() => ({}));
    console.error('[KW-2a] music_review_pack error:', JSON.stringify(errBody));
  }
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);
  expect(body.timeline.shots.length).toBeGreaterThan(0);
});

test('KW-2b: Confirm Plan writes MusicPlan.json with required fields', async ({ page }) => {
  await openMusicTab(page);
  await page.click('#music-btn-review');
  await page.waitForResponse(r => r.url().includes('/api/music_review_pack'), { timeout: 15000 });

  // Wait for Confirm Plan button to be clickable
  await page.waitForFunction(
    () => !document.getElementById('music-btn-confirm').disabled,
    { timeout: 8000 }
  ).catch(() => {}); // button may already be enabled

  const [saveResp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/music_plan_save'), { timeout: 10000 }),
    page.click('#music-btn-confirm'),
  ]);
  expect(saveResp.status()).toBe(200);

  const mp = musicplan();
  expect(mp).toHaveProperty('shot_overrides');
  expect(Array.isArray(mp.shot_overrides)).toBe(true);
  expect(mp).toHaveProperty('loop_selections');
});
