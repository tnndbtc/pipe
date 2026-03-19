// TEST COVERAGE: KW-11
// Regression: SFX tab PREVIEW AUDIO section shows no VO/SFX/Music bars after
// Generate Preview because the frontend did not use d.timeline from the POST
// response, causing sfxRenderTimeline() to never be called.
//
// KW-11d: sfx_preview with include_music=true must return music_items in the
//         timeline response. Catches: music stripped from SFX preview when
//         include_music flag is sent (regression or missing MusicPlan loading).
//
// KW-11e: music bars must render in sfx-tl-music after Generate Preview when
//         include_music checkbox is checked. Catches: frontend sfxRenderTimeline
//         ignoring music_items from the timeline JSON.
const { test, expect } = require('@playwright/test');
const fs   = require('fs');
const path = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, resetKW13, getEpDir } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(() => {
  resetKW2();
  // Remove any leftover SfxPreviewPack from a previous run
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
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

  // Wait for sfxRenderTimeline() to process d.timeline from the POST response
  // and render VO bars into #sfx-tl-vo.
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

// KW-11c: /api/sfx_preview response contains a valid timeline object.
// timeline.json is no longer written to disk — the timeline is returned
// directly in the POST response body and used in-memory by the frontend.
test('KW-11c: /api/sfx_preview response contains valid timeline with total_dur_sec and vo_items', async ({ page }) => {
  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Timeline is returned in the response body — no file on disk needed.
  expect(body.timeline).toBeDefined();
  expect(body.timeline.total_dur_sec).toBeGreaterThan(0);
  expect(body.timeline.vo_items.length).toBeGreaterThan(0);

  // Confirm timeline.json is NOT written to disk.
  const tlPath = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack', 'timeline.json');
  expect(fs.existsSync(tlPath)).toBe(false);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-11d: sfx_preview with include_music=true returns music_items in timeline.
//
// Requires MusicPlan.json (resetKW13) so sfx_preview_pack can load music stems.
// If the backend strips music (wrong include_music handling, missing MusicPlan
// load, or PIPE_DIR path-security blocking music WAVs), music_items is empty.
// ─────────────────────────────────────────────────────────────────────────────
test('KW-11d: sfx_preview with include_music=true returns music_items in timeline', async ({ request }) => {
  resetKW13();
  // Remove any leftover SfxPreviewPack from a previous run
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });

  const resp = await request.post('http://localhost:19999/api/sfx_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selected:      {},
      include_music: true,   // ← the flag under test
      timing:        {},
      volumes:       {},
      duck_fade:     {},
      cut_clips:     [],
      cut_assign:    {},
      clip_volumes:  {},
    },
  });

  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // The timeline must contain music_items — one per shot that has a music track.
  // Fixture VOPlan has music_items for sc01-sh01 and sc02-sh02.
  expect(body.timeline).toBeDefined();
  expect(Array.isArray(body.timeline.music_items)).toBe(true);
  expect(body.timeline.music_items.length).toBeGreaterThan(0);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-11e: music bars render in sfx-tl-music after Generate Preview when
//         include_music checkbox is checked.
//
// Catches: sfxRenderTimeline() ignoring music_items in the timeline JSON so
// the DOM div stays empty even though the API returned music_items correctly.
// ─────────────────────────────────────────────────────────────────────────────
test('KW-11e: music bars render in sfx-tl-music when include_music is checked', async ({ page }) => {
  resetKW13();
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });

  // Open SFX tab and select episode
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );

  // Check the include_music checkbox and force-enable the preview button
  await page.evaluate(() => {
    const cb = document.getElementById('sfx-include-music');
    if (cb) cb.checked = true;
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Wait for sfxRenderTimeline() to process music_items and render music bars
  await page.waitForFunction(
    () => {
      const musicDiv = document.getElementById('sfx-tl-music');
      return musicDiv && musicDiv.children.length > 0;
    },
    { timeout: 8000 }
  );

  const musicBarCount = await page.evaluate(() => {
    const musicDiv = document.getElementById('sfx-tl-music');
    return musicDiv ? musicDiv.children.length : 0;
  });
  // Fixture has 2 shots, each with a music track → expect ≥ 1 music bar
  expect(musicBarCount).toBeGreaterThan(0);
});
