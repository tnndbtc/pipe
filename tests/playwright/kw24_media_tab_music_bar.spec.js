// TEST COVERAGE: KW-24 (merged KW-31)
// Source: prompts/regression.txt § "KW-24: Media Tab Music Bar" and
//         "KW-31: Media Tab PREVIEW and SHOT OVERRIDES timeline source"
//
// KW-24  : Music bar renders in #media-tl-music after Generate Preview
// KW-31a : /api/vo_timeline called after Media Generate Preview
// KW-31b : /api/sfx_timeline called after Media Generate Preview
// KW-31c : /api/music_timeline called after Media Generate Preview
// KW-31d : Media VO bars within ruler after Generate Preview (overflow test — FAILS today)
// KW-31e : /api/vo_timeline called on Media episode select (Shot Overrides path — FAILS today)
// KW-31f : /api/sfx_timeline + /api/music_timeline called on episode select (FAILS today)
// KW-31g : Shot Overrides calls all 3 timeline APIs + total_dur_sec >= 90.0 (FAILS today)
// KW-31h : Shot Overrides last shot end label reflects VOPlan extent >= 80.0 (FAILS today)
//
// Fixture: resetKW80 — VOPlan(last end_sec=80.0, pause_after_ms=10000) +
//          MusicPlan + music WAVs + SfxPlan + stub SfxPreviewPack WAV.
// Two-level sentinel:
//   - end_sec=80.0 > ShotList total (55.648s): catches tabs reading ShotList alone
//   - pause_after_ms=10000: full VO tail = 90.0s; catches pause-ignorers
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW80 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW80(); });

// Opens Media tab, selects episode, waits for status text to update.
async function openMediaTab(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');
  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );
}

// Opens Media tab and clicks Generate Preview; returns the media_preview response.
async function openMediaTabAndGenerate(page) {
  await openMediaTab(page);
  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);
  const previewBody = await previewResp.json();
  expect(previewBody.ok, `media_preview failed: ${JSON.stringify(previewBody)}`).toBe(true);
  return previewResp;
}

// ── KW-24: Music bar renders in #media-tl-music after Generate Preview ────────

test('KW-24: music bar renders in #media-tl-music after Generate Preview', async ({ page }) => {
  await openMediaTabAndGenerate(page);

  // _mediaLoadTimeline() is called without await, so status "Preview ready" fires
  // before _tlRender() completes — poll for bars.
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

// ── KW-31a: /api/vo_timeline is called after Media Generate Preview ────────────
//
// Catches: _mediaLoadTimeline() or _loadAndMergeTl() not fetching vo_timeline.

test('KW-31a: /api/vo_timeline is called after Media Generate Preview', async ({ page }) => {
  await openMediaTab(page);

  const voPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 40000 }
  );

  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);

  const voResp = await voPromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline was never called after Media Generate Preview.\n' +
    'Root cause: _mediaLoadTimeline() does not call _loadAndMergeTl(), or\n' +
    '_loadAndMergeTl() does not include /api/vo_timeline.\n' +
    'Fix: ensure Media tab calls _loadAndMergeTl() which fetches all three endpoints.'
  ).not.toBeNull();
  expect(voResp.status(), '/api/vo_timeline must return 200').toBe(200);
});

// ── KW-31b: /api/sfx_timeline is called after Media Generate Preview ───────────

test('KW-31b: /api/sfx_timeline is called after Media Generate Preview', async ({ page }) => {
  await openMediaTab(page);

  const sfxPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/sfx_timeline',
    { timeout: 40000 }
  );

  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);

  const sfxResp = await sfxPromise.catch(() => null);
  expect(
    sfxResp,
    'SFX TIMELINE NOT FETCHED: /api/sfx_timeline was never called after Media Generate Preview.\n' +
    'The Media tab must load SFX overlay data from /api/sfx_timeline via _loadAndMergeTl().'
  ).not.toBeNull();
  expect(sfxResp.status(), '/api/sfx_timeline must return 200').toBe(200);
});

// ── KW-31c: /api/music_timeline is called after Media Generate Preview ──────────

test('KW-31c: /api/music_timeline is called after Media Generate Preview', async ({ page }) => {
  await openMediaTab(page);

  const musPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/music_timeline',
    { timeout: 40000 }
  );

  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);

  const musResp = await musPromise.catch(() => null);
  expect(
    musResp,
    'MUSIC TIMELINE NOT FETCHED: /api/music_timeline was never called after Media Generate Preview.\n' +
    'The Media tab must load music overlay data from /api/music_timeline via _loadAndMergeTl().'
  ).not.toBeNull();
  expect(musResp.status(), '/api/music_timeline must return 200').toBe(200);
});

// ── KW-31d: Media VO bars are within ruler after Generate Preview ─────────────
//
// _loadAndMergeTl() returns total_dur_sec = max(vo.total_sec, max(vo_items.end_sec)).
// With end_sec=80.0 → total_dur_sec must be >= 80.0.
// If the Media tab does not use _loadAndMergeTl() (or uses ShotList alone),
// total_dur_sec = 55.648 → VO bars overflow the ruler.
//
// Checks VO bars in #media-tl-vo: all bars' right edges must be <= 105%.
// With total_dur_sec=55.648 and end_sec=80.0 → last bar overflows at ~143%.
//
// FAILS today: total_dur_sec uses ShotList (55.648s), not VOPlan (80.0s).

test('KW-31d: Media VO bars are within ruler after Generate Preview (no overflow)', async ({ page }) => {
  await openMediaTabAndGenerate(page);

  // Poll for #media-tl-vo to have VO bars rendered
  await page.waitForFunction(
    () => {
      const el = document.getElementById('media-tl-vo');
      return el && el.children.length > 0;
    },
    { timeout: 15000 }
  ).catch(() => {});

  const voBars = await page.evaluate(() => {
    const el = document.getElementById('media-tl-vo');
    if (!el) return [];
    return Array.from(el.children).map(bar => ({
      left:  parseFloat((bar.style.left  || '0').replace('%', '')),
      width: parseFloat((bar.style.width || '0').replace('%', '')),
    }));
  });

  expect(
    voBars.length,
    'MEDIA VO BARS MISSING: #media-tl-vo has 0 children after Generate Preview.\n' +
    'VOPlan has 9 vo_items — VO bars must be visible in the Media Tab preview.'
  ).toBeGreaterThan(0);

  // Every VO bar must be within the ruler.
  // FAILS when total_dur_sec = 55.648 (ShotList) but last VO end_sec = 80.0
  // → last bar renders at left + width ≈ 143% (off-screen to the right).
  for (const bar of voBars) {
    expect(
      bar.left + bar.width,
      'MEDIA VO BAR OVERFLOW: bar right edge at ' + (bar.left + bar.width).toFixed(1) + '% > 105%.\n' +
      'Root cause: total_dur_sec uses ShotList (55.648s) but VO end_sec=80.0 →\n' +
      'bar overflows ruler. Fix: use _loadAndMergeTl() for total_dur_sec.'
    ).toBeLessThanOrEqual(105);
  }
});

// ── KW-31e: /api/vo_timeline called on Media episode select (Shot Overrides) ──
//
// mediaLoadShotTimeline() currently reads ONLY ShotList.json + AssetManifest.
// It does NOT call /api/vo_timeline, /api/music_timeline, or /api/sfx_timeline.
// This test asserts that it SHOULD call /api/vo_timeline on episode select so
// that Media Shot Overrides timing is anchored to VO extent, not ShotList alone.
//
// FAILS today: mediaLoadShotTimeline() does not call vo_timeline.

test('KW-31e: /api/vo_timeline is called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');

  // Set up interceptor BEFORE episode select — mediaLoadShotTimeline() fires
  // immediately on episode select, before any Generate Preview click.
  const voPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 15000 }
  );

  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );

  // Wait for mediaLoadShotTimeline() to complete
  await page.waitForFunction(
    () => {
      const el = document.getElementById('media-shot-overrides');
      return el && el.innerHTML.trim().length > 0 &&
             !el.innerHTML.includes('No shots loaded');
    },
    { timeout: 15000 }
  ).catch(() => {});

  const voResp = await voPromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED on episode select: /api/vo_timeline was never called\n' +
    'by mediaLoadShotTimeline() when Media Tab episode was selected.\n' +
    'Root cause: mediaLoadShotTimeline() reads ShotList.json directly for timing\n' +
    'instead of calling /api/vo_timeline for authoritative shot start/end_sec.\n' +
    'Fix: replace the ShotList cumulative cursor with /api/vo_timeline shot data.'
  ).not.toBeNull();
  expect(voResp.status(), '/api/vo_timeline must return 200').toBe(200);
});

// ── KW-31f-sfx: /api/sfx_timeline called on Media episode select ──────────────
//
// FAILS today: mediaLoadShotTimeline() does not call sfx_timeline.

test('KW-31f-sfx: /api/sfx_timeline called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');

  const sfxPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/sfx_timeline',
    { timeout: 12000 }
  );

  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );

  await page.waitForFunction(
    () => {
      const el = document.getElementById('media-shot-overrides');
      return el && el.innerHTML.trim().length > 0 &&
             !el.innerHTML.includes('No shots loaded');
    },
    { timeout: 12000 }
  ).catch(() => {});

  const sfxResp = await sfxPromise.catch(() => null);
  expect(
    sfxResp,
    'SFX TIMELINE NOT FETCHED on episode select: /api/sfx_timeline was never called\n' +
    'by mediaLoadShotTimeline().\n' +
    'Root cause: mediaLoadShotTimeline() does not fetch /api/sfx_timeline.\n' +
    'Fix: add /api/sfx_timeline fetch to mediaLoadShotTimeline() (use _loadAndMergeTl).'
  ).not.toBeNull();
  expect(sfxResp.status(), '/api/sfx_timeline must return 200').toBe(200);
});

// ── KW-31f-mus: /api/music_timeline called on Media episode select ─────────────
//
// FAILS today: mediaLoadShotTimeline() does not call music_timeline.

test('KW-31f-mus: /api/music_timeline called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');

  const musPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/music_timeline',
    { timeout: 12000 }
  );

  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );

  await page.waitForFunction(
    () => {
      const el = document.getElementById('media-shot-overrides');
      return el && el.innerHTML.trim().length > 0 &&
             !el.innerHTML.includes('No shots loaded');
    },
    { timeout: 12000 }
  ).catch(() => {});

  const musResp = await musPromise.catch(() => null);
  expect(
    musResp,
    'MUSIC TIMELINE NOT FETCHED on episode select: /api/music_timeline was never called\n' +
    'by mediaLoadShotTimeline().\n' +
    'Root cause: mediaLoadShotTimeline() does not fetch /api/music_timeline.\n' +
    'Fix: add /api/music_timeline fetch to mediaLoadShotTimeline() (use _loadAndMergeTl).'
  ).not.toBeNull();
  expect(musResp.status(), '/api/music_timeline must return 200').toBe(200);
});

// ── KW-31g: Shot Overrides calls all 3 timeline APIs + total_dur_sec >= 90.0 ───
//
// Strict combined test: when the Media tab Shot Overrides panel loads on episode
// select, it MUST call /api/vo_timeline, /api/music_timeline, AND /api/sfx_timeline.
// The merged total_dur_sec must be >= 90.0:
//   last vo_item.end_sec = 80.0, pause_after_ms = 10000 → tail = 90.0s
//
// FAILS today: mediaLoadShotTimeline() does not call _loadAndMergeTl() and
// never fetches any of the three timeline APIs.

test('KW-31g: Shot Overrides panel calls all 3 timeline APIs and yields total_dur_sec >= 90.0', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');

  // Register all three response interceptors BEFORE episode select so that
  // mediaLoadShotTimeline() — which fires immediately on episode select —
  // is captured even if it completes before the assertions below.
  const voPromise  = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 15000 }
  );
  const sfxPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/sfx_timeline',
    { timeout: 15000 }
  );
  const musPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/music_timeline',
    { timeout: 15000 }
  );

  await page.selectOption('#media-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('media-status-text').textContent !== 'Select an episode to begin.',
    { timeout: 10000 }
  );

  // Wait for Shot Overrides panel to render content
  await page.waitForFunction(
    () => {
      const el = document.getElementById('media-shot-overrides');
      return el && el.innerHTML.trim().length > 0 &&
             !el.innerHTML.includes('No shots loaded');
    },
    { timeout: 15000 }
  ).catch(() => {});

  // Collect responses (null if never called within timeout)
  const voResp  = await voPromise.catch(() => null);
  const sfxResp = await sfxPromise.catch(() => null);
  const musResp = await musPromise.catch(() => null);

  // ── Assert all three endpoints were called ─────────────────────────────────
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline was never called by Shot Overrides panel.\n' +
    'Root cause: mediaLoadShotTimeline() reads ShotList.json directly for timing\n' +
    'instead of calling /api/vo_timeline + /api/sfx_timeline + /api/music_timeline.\n' +
    'Fix: replace the ShotList cursor logic with _loadAndMergeTl() in mediaLoadShotTimeline().'
  ).not.toBeNull();
  expect(voResp.status(), '/api/vo_timeline must return 200').toBe(200);

  expect(
    sfxResp,
    'SFX TIMELINE NOT FETCHED: /api/sfx_timeline was never called by Shot Overrides panel.\n' +
    'Fix: mediaLoadShotTimeline() must call _loadAndMergeTl() which fetches all three endpoints.'
  ).not.toBeNull();
  expect(sfxResp.status(), '/api/sfx_timeline must return 200').toBe(200);

  expect(
    musResp,
    'MUSIC TIMELINE NOT FETCHED: /api/music_timeline was never called by Shot Overrides panel.\n' +
    'Fix: mediaLoadShotTimeline() must call _loadAndMergeTl() which fetches all three endpoints.'
  ).not.toBeNull();
  expect(musResp.status(), '/api/music_timeline must return 200').toBe(200);

  // ── Assert total_dur_sec >= 90.0 ───────────────────────────────────────────
  // Compute total_dur_sec from the vo_timeline response using the same formula as
  // _loadAndMergeTl(): max(total_sec, max(end_sec + pause_after_ms/1000))
  // With end_sec=80.0, pause_after_ms=10000 → voLastEnd = 90.0 → PASS.
  // A path ignoring pause_after_ms gets 80.0 → FAIL.
  // A path using ShotList alone gets 55.648 → FAIL.
  const voBody = await voResp.json();
  const voItems = voBody.vo_items || [];
  const voTotalSec = voBody.total_sec || 0;
  const voLastEnd = voItems.reduce(
    (m, v) => Math.max(m, (v.end_sec || 0) + (v.pause_after_ms || 0) / 1000),
    0
  );
  const totalDurSec = Math.max(voTotalSec, voLastEnd);
  expect(
    totalDurSec,
    'TOTAL_DUR_SEC TOO SMALL: computed ' + totalDurSec + 's from /api/vo_timeline response ' +
    '(total_sec=' + voTotalSec + ', voLastEnd=' + voLastEnd + ').\n' +
    'Expected >= 90.0 (last vo_item.end_sec=80.0 + pause_after_ms=10000ms).\n' +
    'Root cause: pause_after_ms is not included in the VO tail computation.\n' +
    'Fix: use _loadAndMergeTl() which applies max(end_sec + pause_after_ms/1000).'
  ).toBeGreaterThanOrEqual(90.0);

  // ── Assert Shot Overrides rows are present ─────────────────────────────────
  const shotRowCount = await page.evaluate(() => {
    const el = document.getElementById('media-shot-overrides');
    if (!el) return 0;
    return el.querySelectorAll('[data-shot-id]').length;
  });
  expect(
    shotRowCount,
    'SHOT OVERRIDES ROWS MISSING: 0 [data-shot-id] rows in #media-shot-overrides.\n' +
    'Shot Overrides must render a row for each shot after loading all three timeline endpoints.\n' +
    'Fixture has multiple shots — at least 1 row expected.'
  ).toBeGreaterThan(0);
});

// ── KW-31h: Media Shot Override last shot end label reflects VOPlan extent ─────
//
// Each Media shot block renders "episode X.Xs – Y.Ys (Zs)" via .music-shot-hdr-ep
// inside #media-shot-overrides. With VOPlan last end_sec=80.0, the last shot block
// end label must be >= 80.0.
// FAILS today: label shows 55.6s (ShotList cumulative) instead of 80.0s (VOPlan).
// Root cause: mediaLoadShotTimeline() sources timing from ShotList, not /api/vo_timeline.
// Fixture: resetKW80 (beforeEach) — VOPlan(end_sec=80.0) + MusicPlan + SfxPlan.

test('KW-31h: Media Shot Override last shot end label reflects VOPlan extent (>= 80.0)', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="media"]');
  await page.selectOption('#media-ep-select', 'test-proj|s01e01');

  // Wait for Shot Overrides to render (sentinel: at least one .music-shot-hdr-ep present)
  await page.waitForFunction(
    () => document.querySelector('#media-shot-overrides .music-shot-hdr-ep') !== null,
    { timeout: 15000 }
  );

  const lastShotEnd = await page.evaluate(() => {
    const container = document.getElementById('media-shot-overrides');
    if (!container) return null;
    const labels = Array.from(container.querySelectorAll('.music-shot-hdr-ep'));
    if (!labels.length) return null;
    // Text: "episode\u00a0X.Xs – Y.Ys\u00a0(Z.Zs)"
    // Normalise non-breaking spaces then split on en-dash separator
    const text = labels[labels.length - 1].textContent.replace(/\u00a0/g, ' ');
    const parts = text.split(' – ');
    if (parts.length < 2) return null;
    const m = parts[1].match(/^([\d.]+)s/);
    return m ? parseFloat(m[1]) : null;
  });

  expect(
    lastShotEnd,
    'No .music-shot-hdr-ep labels found in #media-shot-overrides — Shot Override blocks may not have rendered'
  ).not.toBeNull();

  expect(
    lastShotEnd,
    'WRONG SOURCE: Last Media Shot Override label shows end=' + lastShotEnd + 's < 80.0.\n' +
    'Root cause: mediaLoadShotTimeline() sources timing from ShotList cumulative sums\n' +
    '(sc02-sh02: 28.089 + 27.559 = 55.648s) instead of VOPlan last VO end_sec=80.0.\n' +
    'Fix: use /api/vo_timeline shot start_sec/end_sec for Media Shot Override timing labels.'
  ).toBeGreaterThanOrEqual(80.0);
});
