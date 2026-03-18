// TEST COVERAGE: KW-12
// Regression: sfxRenderShotOverrides onchange subtracts epStart from the user's
// episode-absolute input before calling sfxSetTiming().  This causes _sfxTiming
// to store shot-relative offsets which sfx_preview_pack.py then treats as
// episode-absolute → SFX is placed at the wrong time in the audio buffer.
//
// Repro: open SFX tab, set timing for a shot-2 item to start=50 end=55,
//        click Generate Preview.  Before fix the POST body has
//        timing.start ≈ 21.14 (= 50 – 28.86) instead of 50.0.
//
// epStart for sc02-sh02 = 28.86s  (ShotList.json, shot[1].duration_sec cumulative)
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2 } = require('../helpers/fixture_state');

const SC02_EP_START = 28.86;   // episode-absolute start of sc02-sh02 (from ShotList.json)
const INPUT_START   = 50.0;    // episode-absolute start the user types in
const INPUT_END     = 55.0;    // episode-absolute end   the user types in

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW2(); });

async function openSfxAndWaitForOverrides(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  // Wait for Shot Overrides section to populate (needs AssetManifest + ShotList)
  await page.locator('#sfx-overrides .music-shot-block').first().waitFor({
    state: 'visible',
    timeout: 12000,
  });
}

test('KW-12: Generate Preview POST body carries episode-absolute timing (not shot-relative)', async ({ page }) => {
  await openSfxAndWaitForOverrides(page);

  // sc02-sh02 is the 2nd shot block (index 1).
  // sfx-sc02-sh02-001 is the 1st SFX item in that block → .music-shot-params nth(0).
  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  const startInput = firstItemParams.locator('input[type="number"]').nth(0);
  const endInput   = firstItemParams.locator('input[type="number"]').nth(1);

  // User fills in episode-absolute timing: 50s → 55s
  await startInput.fill(String(INPUT_START));
  await startInput.press('Tab');   // fires onchange → sfxSetTiming
  await endInput.fill(String(INPUT_END));
  await endInput.press('Tab');     // fires onchange → sfxSetTiming

  // Force-enable preview button (no sfx_search_results.json in test fixture)
  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  // Capture the POST body sent to /api/sfx_preview
  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 },
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body   = JSON.parse(req.postData());
  const timing = body.timing && body.timing['sfx-sc02-sh02-001'];

  expect(timing).toBeDefined();

  // ── BUG (before fix) ────────────────────────────────────────────────────────
  // onchange subtracts epStart:  sfxSetTiming(iid, 'start', 50 - 28.86) = 21.14
  // so the POST carries {start: 21.14, end: 26.14} — shot-relative, wrong.
  //
  // ── FIX (after fix) ─────────────────────────────────────────────────────────
  // onchange passes the value directly:  sfxSetTiming(iid, 'start', 50)
  // so the POST carries {start: 50.0, end: 55.0} — episode-absolute, correct.
  // ─────────────────────────────────────────────────────────────────────────────
  expect(timing.start).toBeCloseTo(INPUT_START, 0); // FAILS before fix (≈21.14)
  expect(timing.end).toBeCloseTo(INPUT_END,   0);   // FAILS before fix (≈26.14)
});

test('KW-12b: shot-relative value (epStart subtracted) is NOT sent', async ({ page }) => {
  await openSfxAndWaitForOverrides(page);

  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  const startInput = firstItemParams.locator('input[type="number"]').nth(0);
  const endInput   = firstItemParams.locator('input[type="number"]').nth(1);

  await startInput.fill(String(INPUT_START));
  await startInput.press('Tab');
  await endInput.fill(String(INPUT_END));
  await endInput.press('Tab');

  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 },
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body   = JSON.parse(req.postData());
  const timing = body.timing && body.timing['sfx-sc02-sh02-001'];

  expect(timing).toBeDefined();

  const buggyStart = INPUT_START - SC02_EP_START;  // ≈ 21.14 (shot-relative)
  const buggyEnd   = INPUT_END   - SC02_EP_START;  // ≈ 26.14

  // These should NOT be the values in the request
  expect(Math.abs(timing.start - buggyStart)).toBeGreaterThan(1); // FAILS before fix
  expect(Math.abs(timing.end   - buggyEnd  )).toBeGreaterThan(1); // FAILS before fix
});
