// TEST COVERAGE: KW-18
// Regression: Music and SFX Shot Overrides render empty after episode select
// even when saved plan files are present on disk.
//
// IMPORTANT — each test is designed to FAIL when the bug is present:
//
// KW-18a/c (Music): fixture has NO VOPlan → /api/music_timeline returns 500 →
//   _musicTimeline stays null → _musicRenderBody MUST use the _musicOverrides
//   fallback path. If the fallback filter drops overrides (e.g. wrong filter key,
//   missing shot_id), blockCount = 0 → expect(0).toBeGreaterThanOrEqual(2) FAILS.
//
// KW-18b/d (SFX): fixture has VOPlan with sfx_items but no SfxPlan → SFX tab
//   must build shot timeline from ShotList + VOPlan sfx_items. If _sfxShotTimeline
//   is empty or sfx_items not mapped to shots, blockCount = 0 → test FAILS.
//
// These tests would have caught every regression introduced in this session to
// the load/render path (fallback filter change, _musicShotMap population,
// _sfxShotTimeline build from ShotList).

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW18Music, resetKW18Sfx } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });

// ── Helpers ───────────────────────────────────────────────────────────────────

async function openMusicTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  // _musicLoadExisting() is async and calls _musicRenderBody() at the end.
  // Wait until the music-body has at least one .music-card (always created by
  // _musicRenderBody, even for the placeholder "no shots" state).
  await page.waitForFunction(
    () => document.querySelector('#music-body .music-card') !== null,
    { timeout: 10000 }
  );
}

async function openSfxTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  // sfxRenderShotOverrides() is called at end of onSfxEpChange chain.
  // Wait until sfx-overrides has been written (it always gets innerHTML set).
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-overrides');
      return el && el.innerHTML.trim().length > 0;
    },
    { timeout: 10000 }
  );
}

// ── KW-18a: Music Shot Overrides shows saved blocks via fallback path ─────────
// Fixture: NO VOPlan → timeline 500 → _musicTimeline=null → fallback.
// Bug: filter(o => o.shot_id) drops overrides with no shot_id → 0 blocks → FAIL.
// Fix: MusicPlan 1.1 has shot_id on every override → 2 blocks → PASS.

test('KW-18a: Music Shot Overrides shows ≥2 shot blocks from MusicPlan fallback', async ({ page }) => {
  resetKW18Music();
  await openMusicTabAndSelectEp(page);

  const blockCount = await page.evaluate(() =>
    document.querySelectorAll('#music-body .music-shot-block').length
  );

  // 0 here means the fallback dropped all overrides — the regression is present.
  expect(blockCount).toBeGreaterThanOrEqual(2);
});

// ── KW-18b: SFX Shot Overrides shows shot blocks from shot timeline ───────────
// Fixture: VOPlan with sfx_items, no SfxPlan → shot timeline must be built from
// ShotList + VOPlan sfx_items. 0 blocks means _sfxShotTimeline not populated.

test('KW-18b: SFX Shot Overrides shows ≥2 shot blocks from shot timeline', async ({ page }) => {
  resetKW18Sfx();
  await openSfxTabAndSelectEp(page);

  const blockCount = await page.evaluate(() =>
    document.querySelectorAll('#sfx-overrides .music-shot-block').length
  );

  // 0 here means sfx_items were not mapped to shots — regression present.
  expect(blockCount).toBeGreaterThanOrEqual(2);
});

// ── KW-18c: Music Shot Overrides shows the correct shot IDs ──────────────────
// Deeper check: correct shot IDs appear, not just any block count.
// Catches: overrides present but mapped to wrong/missing shot IDs.

test('KW-18c: Music Shot Overrides contains sc01-sh01 and sc02-sh02', async ({ page }) => {
  resetKW18Music();
  await openMusicTabAndSelectEp(page);

  const shotIds = await page.evaluate(() =>
    Array.from(document.querySelectorAll('#music-body .music-shot-hdr-id'))
      .map(el => el.textContent.trim())
  );

  expect(shotIds.some(id => id.includes('sc01-sh01'))).toBe(true);
  expect(shotIds.some(id => id.includes('sc02-sh02'))).toBe(true);
});

// ── KW-18d: SFX Shot Overrides shows the correct shot IDs ────────────────────

test('KW-18d: SFX Shot Overrides contains sc01-sh01 and sc02-sh02', async ({ page }) => {
  resetKW18Sfx();
  await openSfxTabAndSelectEp(page);

  const shotIds = await page.evaluate(() =>
    Array.from(document.querySelectorAll('#sfx-overrides .music-shot-hdr-id'))
      .map(el => el.textContent.trim())
  );

  expect(shotIds.some(id => id.includes('sc01-sh01'))).toBe(true);
  expect(shotIds.some(id => id.includes('sc02-sh02'))).toBe(true);
});
