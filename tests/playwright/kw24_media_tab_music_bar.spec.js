// TEST COVERAGE: KW-24
// Source: prompts/regression.txt § "KW-24: Media Tab Music Bar" and
//         "KW-24: Media Tab PREVIEW and SHOT OVERRIDES timeline source"
//
// KW-24  : Music bar renders in #media-tl-music after Generate Preview
// KW-24a : /api/vo_timeline called after Media Generate Preview
// KW-24b : /api/sfx_timeline called after Media Generate Preview
// KW-24c : /api/music_timeline called after Media Generate Preview
// KW-24d : Media VO bars within ruler after Generate Preview (overflow test — FAILS today)
// KW-24e : /api/vo_timeline called on Media episode select (Shot Overrides path — FAILS today)
// KW-24f : /api/sfx_timeline + /api/music_timeline called on episode select (FAILS today)
// KW-24g : Shot Overrides calls all 3 timeline APIs + total_dur_sec >= 90.0 (FAILS today)
// KW-24h : Shot Overrides last shot end label reflects VOPlan extent >= 80.0 (FAILS today)
// KW-24i : preview_video.mp4 duration reflects scene_heads gap (FAILS today)
//          With bug:  ~65.3s (raw ShotList cumulative, no scene_heads gap)
//          With fix:  ~80.3s (15s scene_heads["sc01"] gap included in concat)
//
// Fixture: resetKW80 — VOPlan(last end_sec=80.0, pause_after_ms=10000) +
//          MusicPlan + music WAVs + SfxPlan + stub SfxPreviewPack WAV.
// Two-level sentinel:
//   - end_sec=80.0 > ShotList total (55.648s): catches tabs reading ShotList alone
//   - pause_after_ms=10000: full VO tail = 90.0s; catches pause-ignorers
const { test, expect } = require('@playwright/test');
const { execSync }      = require('child_process');
const fs                = require('fs');
const path              = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW80, getEpDir } = require('../helpers/fixture_state');

const FIXTURE_EP = path.join(__dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01');

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

// ── KW-24a: /api/vo_timeline is called after Media Generate Preview ────────────
//
// Catches: _mediaLoadTimeline() or _loadAndMergeTl() not fetching vo_timeline.

test('KW-24a: /api/vo_timeline is called after Media Generate Preview', async ({ page }) => {
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

// ── KW-24b: /api/sfx_timeline is called after Media Generate Preview ───────────

test('KW-24b: /api/sfx_timeline is called after Media Generate Preview', async ({ page }) => {
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

// ── KW-24c: /api/music_timeline is called after Media Generate Preview ──────────

test('KW-24c: /api/music_timeline is called after Media Generate Preview', async ({ page }) => {
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

// ── KW-24d: Media VO bars have correct start positions after Generate Preview ──
//
// Intercepts /api/vo_timeline to get authoritative start_sec / end_sec per item,
// then computes total_dur_sec = max(total_sec, max(end_sec + pause_after_ms/1000)).
// Each rendered bar's left% must match start_sec / total_dur_sec * 100 within ±0.5%.
//
// Fixture sentinel values (resetKW80):
//   sc02-sh02 start_sec = 43.1s  (VOPlan)  → left ≈ 43.1/90.0 = 47.9%
//   sc02-sh02 start_sec = 28.1s  (ShotList cumulative) → left ≈ 28.1/55.648 = 50.5%
//   delta = 18.8% — far outside the ±0.5% tolerance, so the bug is caught.
//
// Also retains the right-edge overflow guard (right edge <= 105%).

test('KW-24d: Media VO bars have correct start positions after Generate Preview', async ({ page }) => {
  await openMediaTab(page);

  // Intercept /api/vo_timeline BEFORE clicking Generate Preview so the response
  // is captured even if it completes before the assertions below.
  const voPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 40000 }
  );

  const [previewResp] = await Promise.all([
    page.waitForResponse(r => new URL(r.url()).pathname === '/api/media_preview', { timeout: 60000 }),
    page.click('#media-btn-preview'),
  ]);
  expect(previewResp.status(), 'media_preview must return 200').toBe(200);

  // /api/vo_timeline must have been called (KW-24a covers this in depth; here we
  // need its body to compute expected bar positions).
  const voResp = await voPromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline was never called after Generate Preview.\n' +
    'Cannot verify bar positions without authoritative start_sec values.\n' +
    'Fix: ensure _mediaLoadTimeline() calls _loadAndMergeTl().'
  ).not.toBeNull();
  expect(voResp.status(), '/api/vo_timeline must return 200').toBe(200);

  // Derive total_dur_sec using the same formula as _loadAndMergeTl().
  const voBody    = await voResp.json();
  const voItems   = voBody.vo_items || [];
  const voLastEnd = voItems.reduce(
    (m, v) => Math.max(m, (v.end_sec || 0) + (v.pause_after_ms || 0) / 1000),
    0
  );
  const totalDurSec = Math.max(voBody.total_sec || 0, voLastEnd);

  // Poll for #media-tl-vo to have VO bars rendered.
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
    'VOPlan has ' + voItems.length + ' vo_items — VO bars must be visible in the Media Tab preview.'
  ).toBeGreaterThan(0);

  expect(
    voBars.length,
    'BAR COUNT MISMATCH: #media-tl-vo has ' + voBars.length + ' bars but\n' +
    '/api/vo_timeline returned ' + voItems.length + ' vo_items.\n' +
    'Each vo_item must produce exactly one bar.'
  ).toBe(voItems.length);

  // Per-bar position and overflow checks.
  for (let i = 0; i < voItems.length; i++) {
    const item        = voItems[i];
    const bar         = voBars[i];
    const expectedLeft = (item.start_sec / totalDurSec) * 100;

    // ±0.5% tolerance absorbs floating-point/CSS rounding.
    // Delta between ShotList (28.1s) and VOPlan (43.1s) for sc02-sh02 is ~18.8% —
    // orders of magnitude larger than this tolerance.
    expect(
      Math.abs(bar.left - expectedLeft),
      'WRONG BAR START at vo_item[' + i + ']: start_sec=' + item.start_sec + 's\n' +
      '  expected left = ' + expectedLeft.toFixed(2) + '%  (start_sec / totalDurSec * 100)\n' +
      '  actual  left = ' + bar.left.toFixed(2) + '%\n' +
      'Root cause: bar position sourced from ShotList cumulative instead of\n' +
      '/api/vo_timeline start_sec. Fix: use _loadAndMergeTl() start_sec for bar left.'
    ).toBeLessThan(0.5);

    // Right edge must not overflow the ruler.
    expect(
      bar.left + bar.width,
      'MEDIA VO BAR OVERFLOW at vo_item[' + i + ']: right edge = ' +
      (bar.left + bar.width).toFixed(1) + '% > 105%.\n' +
      'Root cause: total_dur_sec uses ShotList (55.648s) but VO end_sec=80.0 →\n' +
      'bar overflows ruler. Fix: use _loadAndMergeTl() for total_dur_sec.'
    ).toBeLessThanOrEqual(105);
  }
});

// ── KW-24e: /api/vo_timeline called on Media episode select (Shot Overrides) ──
//
// mediaLoadShotTimeline() currently reads ONLY ShotList.json + AssetManifest.
// It does NOT call /api/vo_timeline, /api/music_timeline, or /api/sfx_timeline.
// This test asserts that it SHOULD call /api/vo_timeline on episode select so
// that Media Shot Overrides timing is anchored to VO extent, not ShotList alone.
//
// FAILS today: mediaLoadShotTimeline() does not call vo_timeline.

test('KW-24e: /api/vo_timeline is called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
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

// ── KW-24f-sfx: /api/sfx_timeline called on Media episode select ──────────────
//
// FAILS today: mediaLoadShotTimeline() does not call sfx_timeline.

test('KW-24f-sfx: /api/sfx_timeline called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
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

// ── KW-24f-mus: /api/music_timeline called on Media episode select ─────────────
//
// FAILS today: mediaLoadShotTimeline() does not call music_timeline.

test('KW-24f-mus: /api/music_timeline called on Media Tab episode select (Shot Overrides path)', async ({ page }) => {
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

// ── KW-24g: Shot Overrides calls all 3 timeline APIs + total_dur_sec >= 90.0 ───
//
// Strict combined test: when the Media tab Shot Overrides panel loads on episode
// select, it MUST call /api/vo_timeline, /api/music_timeline, AND /api/sfx_timeline.
// The merged total_dur_sec must be >= 90.0:
//   last vo_item.end_sec = 80.0, pause_after_ms = 10000 → tail = 90.0s
//
// FAILS today: mediaLoadShotTimeline() does not call _loadAndMergeTl() and
// never fetches any of the three timeline APIs.

test('KW-24g: Shot Overrides panel calls all 3 timeline APIs and yields total_dur_sec >= 90.0', async ({ page }) => {
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

// ── KW-24h: Media Shot Override last shot end label reflects VOPlan extent ─────
//
// Each Media shot block renders "episode X.Xs – Y.Ys (Zs)" via .music-shot-hdr-ep
// inside #media-shot-overrides. With VOPlan last end_sec=80.0, the last shot block
// end label must be >= 80.0.
// FAILS today: label shows 55.6s (ShotList cumulative) instead of 80.0s (VOPlan).
// Root cause: mediaLoadShotTimeline() sources timing from ShotList, not /api/vo_timeline.
// Fixture: resetKW80 (beforeEach) — VOPlan(end_sec=80.0) + MusicPlan + SfxPlan.

test('KW-24h: Media Shot Override last shot end label reflects VOPlan extent (>= 80.0)', async ({ page }) => {
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

// ── KW-24i: preview_video.mp4 duration includes scene_heads gap ───────────────
//
// media_preview_pack.py derives scene slot durations from VOPlan.vo_items
// start_sec values + scene_heads offsets (no ShotList read).
//
// Fixture: VOPlan with scene_heads={"sc01":15}, last end_sec=55.348s.
//   VOPlan is "unaware" (sc01 first VO start_sec=0.0 < head=15) so the code
//   takes the unaware path: sc01 slot = 28.089 + 15 = 43.089s.
//
// Sentinel arithmetic:
//   Bug (old code, naive ShotList cumsum, no gap): 28.089 + 27.559 = 55.648s
//   Fix (VOPlan scene timeline, head extends sc01): 43.089 + 27.259 = 70.348s
//
// Threshold 62.5s is the midpoint — no rounding can bridge the 15s delta.
// ffprobe reads the container duration directly from the mp4 header.

test('KW-24i: preview_video.mp4 duration includes scene_heads gap from VOPlan', async ({ page }) => {
  // Override the resetKW80 fixture state for this test.
  // We need: VOPlan with scene_heads={"sc01":15} and last end_sec=55.348s
  // (the unaware fixture: sc01 first VO start_sec=0.0, head=15 → unaware mode).
  // ShotList is NOT read by media_preview_pack.py — no restore needed.
  const ep = getEpDir();
  // Restore the original (unpatched) VOPlan — fixture already has scene_heads={"sc01":15}
  // and last end_sec=55.348 (no 80s patch from resetKW80).
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  // Keep last vo_item end_sec at 55.348 (fixture default) — no 80s patch.
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
  // Clear any stale preview so Generate Preview actually runs.
  const packDir = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });

  await openMediaTabAndGenerate(page);

  const videoPath = path.join(ep, 'assets', 'media', 'MediaPreviewPack', 'preview_video.mp4');

  // ffprobe extracts the container duration (seconds) from the mp4 header.
  let durationSec;
  try {
    const out = execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videoPath}"`,
      { encoding: 'utf8' }
    ).trim();
    durationSec = parseFloat(out);
  } catch (e) {
    throw new Error(
      'ffprobe failed on preview_video.mp4 — file may not exist or ffprobe not installed.\n' +
      'Path checked: ' + videoPath + '\n' +
      'ffprobe error: ' + e.message
    );
  }

  expect(
    isNaN(durationSec),
    'ffprobe returned non-numeric duration for preview_video.mp4: ' + durationSec
  ).toBe(false);

  // Bug  (~55.6s): old code used ShotList cumsum, ignored scene_heads → FAILS.
  // Fix  (~70.3s): VOPlan scene timeline, sc01 slot = 28.089 + 15 = 43.089s → PASSES.
  // Threshold 62.5s is the midpoint — impossible to straddle by rounding.
  expect(
    durationSec,
    'WRONG VIDEO DURATION: preview_video.mp4 is ' + durationSec.toFixed(3) + 's.\n' +
    'Expected >= 62.5s — scene_heads["sc01"]=15 must extend sc01 slot by 15s.\n' +
    'Bug value:  ~55.6s  (ignored scene_heads: 28.089 + 27.559 = 55.648s).\n' +
    'Fix value:  ~70.3s  (VOPlan scene timeline: sc01=43.089s + sc02=27.259s).\n' +
    'Root cause: media_preview_pack.py must derive shot slot durations from\n' +
    'VOPlan scene boundaries (vo_item.start_sec + scene_heads), not ShotList.'
  ).toBeGreaterThanOrEqual(62.5);
});
