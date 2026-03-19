// TEST COVERAGE: KW-9
// Music tab "no clip" persistence bug
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW13, musicplan } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW13(); });

async function openMusicTabAndGenerate(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => !document.getElementById('music-btn-review').disabled,
    { timeout: 12000 }
  );
  await page.click('#music-btn-review');
  await page.waitForResponse(r => r.url().includes('/api/music_review_pack'), { timeout: 15000 });
  // Wait for shot override selects to render in music body
  await page.locator('select[id^="music-clip-"]').first().waitFor({ state: 'visible', timeout: 8000 });
}

test('KW-9: no-clip selection persists after tab switch', async ({ page }) => {
  await openMusicTabAndGenerate(page);

  // Find first clip dropdown
  const firstDropdown = page.locator('select[id^="music-clip-"]').first();
  await firstDropdown.waitFor({ state: 'visible', timeout: 5000 });

  // Verify it starts with an auto-assigned clip
  const originalValue = await firstDropdown.inputValue();
  expect(originalValue).not.toBe('');

  // Select "— no clip —"
  await firstDropdown.selectOption('');
  expect(await firstDropdown.inputValue()).toBe('');

  // Click Confirm Plan
  const [saveResp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/music_plan_save'), { timeout: 10000 }),
    page.click('#music-btn-confirm'),
  ]);
  expect(saveResp.status()).toBe(200);

  // Assert sentinel in MusicPlan.json
  const mp = musicplan();
  const sentinel = (mp.shot_overrides || []).find(o => o.music_clip_id === '');
  expect(sentinel).toBeDefined();
  expect(sentinel.music_clip_id).toBe('');

  // Tab switch: Run → Music
  await page.click('button.tab[data-tab="run"]');
  await page.waitForTimeout(400);
  await page.click('button.tab[data-tab="music"]');

  // Wait for music body to re-render and shot override selects to appear after reload
  const afterDropdown = page.locator('select[id^="music-clip-"]').first();
  await afterDropdown.waitFor({ state: 'visible', timeout: 12000 });
  const afterValue = await afterDropdown.inputValue();
  expect(afterValue).toBe('');
});
