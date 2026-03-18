// TEST COVERAGE: Music bar end position after start/end override
// Regression for: green bar ignores end_sec override and extends to full shot duration
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW2(); });

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
  // Wait for shot override selects to render
  await page.locator('select[id^="music-clip-"]').first().waitFor({ state: 'visible', timeout: 8000 });
}

test('KW-10: green music bar ends at end_sec after start/end override', async ({ page }) => {
  await openMusicTabAndGenerate(page);

  // Set start=5, end=12 (episode-absolute) on the first shot (music-sc01-sh01)
  const shotBlock = page.locator('.music-shot-block').first();
  const startInput = shotBlock.locator('input[type="number"]').first();
  const endInput   = shotBlock.locator('input[type="number"]').nth(1);

  await startInput.fill('5');
  await startInput.press('Tab');   // trigger onchange → _musicSetStartEnd
  await endInput.fill('12');
  await endInput.press('Tab');     // trigger onchange → _musicSetStartEnd

  // Click Generate Preview again — sends shot_overrides with start=5, end=12
  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/music_review_pack'), { timeout: 15000 }),
    page.click('#music-btn-review'),
  ]);
  expect(resp.status()).toBe(200);

  // Wait for re-render
  await page.locator('select[id^="music-clip-"]').first().waitFor({ state: 'visible', timeout: 8000 });

  // Read green music bar (first segment in #music-tl-music)
  const barData = await page.evaluate(() => {
    const musDiv = document.getElementById('music-tl-music');
    if (!musDiv || !musDiv.children.length) return null;
    const bar = musDiv.children[0];
    const parseP = (s) => parseFloat((s || '0').replace('%', ''));
    const leftPct  = parseP(bar.style.left);
    const widthPct = parseP(bar.style.width);
    return { leftPct, widthPct, endPct: leftPct + widthPct };
  });

  expect(barData).not.toBeNull();

  // totalDur ≈ 60.63s (from fixture ShotList).
  // start=5 → leftPct ≈ 8.2% (= 5/60.63*100)
  // CORRECT: end=12 → endPct ≈ 19.8% (= 12/60.63*100)
  // BUG:     end uses shot.duration_sec=28.089 → endPct ≈ 46.3%
  expect(barData.endPct).toBeGreaterThan(15);  // sanity: bar actually starts
  expect(barData.endPct).toBeLessThan(30);     // FAILS with bug (~46%), passes when fixed (~20%)
});
