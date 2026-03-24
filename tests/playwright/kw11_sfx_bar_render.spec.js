// TEST COVERAGE: KW-11 (merged KW-12, KW-16, KW-20, KW-26, KW-29, KW-30)
// All SFX tab tests consolidated here.
//
// Sections:
//   KW-11a–e  : Generate Preview — basic (API response, bar render)
//   KW-11f    : Generate Preview — VO bar positions must not overflow ruler (FAILS today)
//   KW-11g    : Shot Overrides — last shot end label reflects VOPlan extent (FAILS today)
//   KW-12     : Shot Overrides params (duck_db, fade_sec, cut clip, clip volumes, WAV creation)
//   KW-16     : SFX adelay timing coordinate space (render_video.py, no server needed)
//   KW-20     : sfx_save_all schema — volume_db in cut_clips, not sfx_entries
//   KW-26     : Preview restore on episode select + sfx_timeline bar positions
//   KW-29     : Preview restore reads all 3 timeline endpoints; total_dur_sec >= 80.0
//   KW-30     : Shot Overrides reads /api/vo_timeline; totalDur reflects VO extent

const { test, expect }  = require('@playwright/test');
const { spawnSync }     = require('child_process');
const fs                = require('fs');
const path              = require('path');
const os                = require('os');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, resetKW12, resetKW13, resetKW26, resetKW80, getEpDir } = require('../helpers/fixture_state');

const BASE_URL = 'http://localhost:19999';

// ── KW-16 constants (derived from fixture ShotList + VOPlan) ──────────────────
const SHOT2_CUMULATIVE_MS  = 28860;
const SHOT2_DURATION_MS    = 57348;
const SFX_EPISODE_ABS_SEC  = 60.0;
const SFX_CORRECT_DELAY_MS = Math.round((SFX_EPISODE_ABS_SEC - SHOT2_CUMULATIVE_MS / 1000) * 1000);
const SFX_BUG_DELAY_MS     = Math.round(SFX_EPISODE_ABS_SEC * 1000);

// ── KW-20 constants ───────────────────────────────────────────────────────────
const KW20_ITEM_ID    = 'sfx-sc01-sh01-001';
const KW20_CLIP_ID    = 'Button_Click_Sharp_0.5s-2.0s';
const KW20_CLIP_VOL   = 6;

// ── Paths ─────────────────────────────────────────────────────────────────────
const FIXTURE_EP = path.join(__dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01');
const RENDER_PY  = path.join(__dirname, '..', '..', 'code', 'http', 'render_video.py');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(() => {
  resetKW2();
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

// Opens SFX tab, selects episode, waits for status bar to change.
// Force-enables the preview button (no sfx_search_results.json in test fixture).
async function openSfxTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );
  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });
}

// Opens SFX tab and waits for Shot Overrides section to populate.
async function openSfxAndWaitForOverrides(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.locator('#sfx-overrides .music-shot-block').first().waitFor({
    state: 'visible',
    timeout: 12000,
  });
}

// Opens SFX tab and waits for sfx-overrides innerHTML to be non-empty.
async function openSfxAndWait(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-overrides');
      return el && el.innerHTML.trim().length > 0;
    },
    { timeout: 12000 }
  );
}

// resetKW80NoWav: same as resetKW80 but without the stub SfxPreviewPack WAV.
// _sfxTryRestorePreview() HEAD check fails → _loadAndMergeTl() is NOT called.
// Isolates the Shot Overrides load path.
function resetKW80NoWav() {
  resetKW80();
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

// KW-16: fake ffmpeg that logs full arg list so we can inspect -filter_complex.
const FAKE_FFMPEG_SRC = `#!/usr/bin/env python3
import sys, json, os, pathlib
log  = os.environ.get('FFMPEG_LOG', '')
args = sys.argv[1:]
if log:
    with open(log, 'a') as f:
        f.write(json.dumps({'args': args}) + '\\n')
if args and not args[-1].startswith('-'):
    out = pathlib.Path(args[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'\\x00' * 100)
sys.exit(0)
`;

// KW-16: set up temp episode dir with VOPlan, ShotList, music WAVs, SfxPlan.
function setupEpDir(epDir) {
  fs.mkdirSync(epDir, { recursive: true });
  fs.copyFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(epDir, 'VOPlan.en.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'ShotList.json'),  path.join(epDir, 'ShotList.json'));
  const musicDir = path.join(epDir, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(f =>
    fs.copyFileSync(path.join(FIXTURE_EP, 'assets', 'music', f), path.join(musicDir, f))
  );
  const sfxWav = path.join(FIXTURE_EP, 'assets', 'music', 'music-sc02-sh02.wav');
  const sfxPlan = {
    timing_format: 'episode_absolute',
    sfx_entries: [{
      item_id:     'sfx-sc02-001',
      shot_id:     'sc02-sh02',
      source_file: sfxWav,
      start_sec:   SFX_EPISODE_ABS_SEC,
    }],
  };
  fs.writeFileSync(path.join(epDir, 'SfxPlan.json'), JSON.stringify(sfxPlan, null, 2));
}

// KW-20 helpers
function writeFakeSearchResults(epDir) {
  const sfxDir = path.join(epDir, 'assets', 'sfx');
  fs.mkdirSync(sfxDir, { recursive: true });
  fs.writeFileSync(
    path.join(sfxDir, 'sfx_search_results.json'),
    JSON.stringify({ saved_at: '2026-01-01T00:00:00Z', results: {}, selected: {} })
  );
}

function createFakeClipWav(epDir) {
  const clipDir = path.join(epDir, 'assets', 'sfx', KW20_ITEM_ID);
  fs.mkdirSync(clipDir, { recursive: true });
  const wavPath = path.join(clipDir, KW20_CLIP_ID + '.wav');
  const buf = Buffer.alloc(44);
  buf.write('RIFF', 0); buf.writeUInt32LE(36, 4);
  buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(44100, 24);
  buf.writeUInt32LE(88200, 28); buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34); buf.write('data', 36);
  buf.writeUInt32LE(0, 40);
  fs.writeFileSync(wavPath, buf);
  return `assets/sfx/${KW20_ITEM_ID}/${KW20_CLIP_ID}.wav`;
}

// KW-16 helper: parse first adelay=X value from filter_complex string.
function parseAdelay(filterComplex) {
  const m = filterComplex.match(/adelay=(\d+)\|/);
  return m ? parseInt(m[1], 10) : null;
}

// ─────────────────────────────────────────────────────────────────────────────
// KW-11a–e: Generate Preview — basic
// ─────────────────────────────────────────────────────────────────────────────

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

  await page.waitForFunction(
    () => {
      const voDiv = document.getElementById('sfx-tl-vo');
      return voDiv && voDiv.children.length > 0;
    },
    { timeout: 8000 }
  );

  const voBarCount = await page.evaluate(
    () => document.getElementById('sfx-tl-vo')?.children.length ?? 0
  );
  expect(voBarCount).toBeGreaterThanOrEqual(2);
});

test('KW-11c: /api/sfx_preview response contains valid timeline with total_dur_sec and vo_items', async ({ page }) => {
  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);
  expect(body.timeline).toBeDefined();
  expect(body.timeline.total_dur_sec).toBeGreaterThan(0);
  expect(body.timeline.vo_items.length).toBeGreaterThan(0);

  const tlPath = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack', 'timeline.json');
  expect(fs.existsSync(tlPath)).toBe(false);
});

test('KW-11d: sfx_preview with include_music=true returns music_items in timeline', async ({ request }) => {
  resetKW13();
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });

  const resp = await request.post('http://localhost:19999/api/sfx_preview', {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      selected: {}, include_music: true,
      timing: {}, volumes: {}, duck_fade: {},
      cut_clips: [], cut_assign: {}, clip_volumes: {},
    },
  });
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);
  expect(body.timeline).toBeDefined();
  expect(Array.isArray(body.timeline.music_items)).toBe(true);
  expect(body.timeline.music_items.length).toBeGreaterThan(0);
});

test('KW-11e: music bars render in sfx-tl-music when include_music is checked', async ({ page }) => {
  resetKW13();
  const packDir = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );
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

  await page.waitForFunction(
    () => {
      const musicDiv = document.getElementById('sfx-tl-music');
      return musicDiv && musicDiv.children.length > 0;
    },
    { timeout: 8000 }
  );
  const musicBarCount = await page.evaluate(
    () => document.getElementById('sfx-tl-music')?.children.length ?? 0
  );
  expect(musicBarCount).toBeGreaterThan(0);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-11f: VO bars must not overflow the ruler after Generate Preview (NEW)
//
// KW-11b passes today — it only checks bar COUNT > 0.
// This test checks BAR POSITIONS: with VOPlan end_sec=80.0 and total_dur_sec
// sourced from ShotList (55.648), the last VO bar renders at ≈143% — off-screen.
//
// FAILS today: sfx_preview total_dur_sec = 55.648 (ShotList), not 80.0 (VOPlan).
// FIX: total_dur_sec = max(ShotList_total, last_vo_end_sec).
// ─────────────────────────────────────────────────────────────────────────────

test('KW-11f: VO bars in sfx-tl-vo must all be within ruler after Generate Preview (no overflow)', async ({ page }) => {
  resetKW80();

  await openSfxTabAndSelectEp(page);

  const [resp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/sfx_preview'), { timeout: 30000 }),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-tl-vo');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  const voBars = await page.evaluate(() => {
    const el = document.getElementById('sfx-tl-vo');
    if (!el) return [];
    return Array.from(el.children).map(bar => ({
      left:  parseFloat((bar.style.left  || '0').replace('%', '')),
      width: parseFloat((bar.style.width || '0').replace('%', '')),
    }));
  });

  expect(voBars.length, '#sfx-tl-vo has 0 bars — KW-11b already catches this').toBeGreaterThan(0);

  // Every VO bar's right edge must be within the ruler (≤ 105%).
  // Fixture: last VO end_sec=80.0. If total_dur_sec=55.648 (ShotList),
  // last bar right = 80/55.648*100 ≈ 143% → FAILS. Fix: total_dur_sec=80.0 → ≈100%.
  for (const bar of voBars) {
    expect(
      bar.left + bar.width,
      'VO BAR OVERFLOW: right edge at ' + (bar.left + bar.width).toFixed(1) + '% > 105%.\n' +
      'Root cause: sfx_preview total_dur_sec sourced from ShotList (55.648s),\n' +
      'not from VOPlan last end_sec (80.0s). Last bar renders at ≈143%.\n' +
      'Fix: total_dur_sec = max(ShotList_total, last_vo_end_sec).'
    ).toBeLessThanOrEqual(105);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-12: Shot Overrides params (migrated from kw12_sfx_timing_absolute.spec.js)
// ─────────────────────────────────────────────────────────────────────────────

test('KW-12: duck_db from Shot Overrides appears in sfx_preview POST body duck_fade field', async ({ page }) => {
  resetKW2();
  await openSfxAndWaitForOverrides(page);

  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  const duckInput = firstItemParams.locator('input[type="number"][min="-30"]');

  await duckInput.fill('-6');
  await duckInput.press('Tab');

  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body = JSON.parse(req.postData());
  expect(body.duck_fade).toBeDefined();
  const df = body.duck_fade['sfx-sc02-sh02-001'];
  expect(df).toBeDefined();
  expect(df.duck_db).toBe(-6);
  expect(body.volumes).toBeDefined();
  expect(body.clip_volumes).toBeDefined();
});

test('KW-12b: fade_sec from Shot Overrides appears in sfx_preview POST body duck_fade field', async ({ page }) => {
  resetKW2();
  await openSfxAndWaitForOverrides(page);

  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  const fadeInput = firstItemParams.locator('input[type="number"][step="0.05"]');

  await fadeInput.fill('0.5');
  await fadeInput.press('Tab');

  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body = JSON.parse(req.postData());
  expect(body.duck_fade).toBeDefined();
  const df = body.duck_fade['sfx-sc02-sh02-001'];
  expect(df).toBeDefined();
  expect(df.fade_sec).toBeCloseTo(0.5, 2);
});

test('KW-12c: sfx_cut_clip returns title-based clip_id and Generated Clips section appears after reload', async ({ request, page }) => {
  resetKW12();

  const epDir       = getEpDir();
  const sourceFile  = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const TITLE       = 'Button Click Sharp';
  const START       = 0.5;
  const END         = 2.0;
  const EXPECTED_ID = 'Button_Click_Sharp_0.5s-2.0s';
  const EXPECTED_WAV = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', EXPECTED_ID + '.wav');

  const resp = await request.post(`${BASE_URL}/api/sfx_cut_clip`, {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      item_id: 'sfx-sc01-sh01-001', candidate_idx: 0,
      source_file: sourceFile, title: TITLE,
      start_sec: START, end_sec: END,
    },
  });
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.clip_id).toBe(EXPECTED_ID);
  expect(body.duration_sec).toBeCloseTo(END - START, 1);
  expect(fs.existsSync(EXPECTED_WAV)).toBe(true);

  const cutJson = path.join(epDir, 'assets', 'sfx', 'sfx_cut_clips.json');
  expect(fs.existsSync(cutJson)).toBe(true);
  const clips = JSON.parse(fs.readFileSync(cutJson, 'utf8'));
  expect(clips.some(c => c.clip_id === EXPECTED_ID)).toBe(true);

  await openSfxAndWaitForOverrides(page);
  const genClipsHdr = page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ });
  await expect(genClipsHdr).toBeVisible({ timeout: 6000 });
  const genClipsTable = page.locator('.music-card', {
    has: page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ }),
  });
  await expect(genClipsTable.locator('td').first()).toContainText(EXPECTED_ID);
});

test('KW-12d: clip volume set in Generated Clips table appears in sfx_preview POST body under clip_volumes[clip_id]', async ({ request, page }) => {
  resetKW12();

  const epDir      = getEpDir();
  const sourceFile = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const TITLE      = 'Button Click Sharp';
  const START      = 0.5;
  const END        = 2.0;
  const CLIP_ID    = 'Button_Click_Sharp_0.5s-2.0s';
  const ITEM_ID    = 'sfx-sc01-sh01-001';

  const cutResp = await request.post(`${BASE_URL}/api/sfx_cut_clip`, {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      item_id: ITEM_ID, candidate_idx: 0,
      source_file: sourceFile, title: TITLE,
      start_sec: START, end_sec: END,
    },
  });
  expect(cutResp.status()).toBe(200);
  const cutBody = await cutResp.json();
  expect(cutBody.clip_id).toBe(CLIP_ID);

  await openSfxAndWaitForOverrides(page);
  const genClipsHdr = page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ });
  await expect(genClipsHdr).toBeVisible({ timeout: 6000 });

  const genClipsCard = page.locator('.music-card', {
    has: page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ }),
  });
  const clipVolInput = genClipsCard.locator('input[type="number"][min="-18"]').first();
  await clipVolInput.fill('-6');
  await clipVolInput.press('Tab');

  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 }
    ),
    page.waitForResponse(
      r => r.url().includes('/api/sfx_preview'),
      { timeout: 30000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body = JSON.parse(req.postData());
  expect(body.clip_volumes).toBeDefined();
  expect(body.clip_volumes[CLIP_ID]).toBe(-6);
  const slotVol = (body.volumes || {})[ITEM_ID];
  expect(slotVol === undefined || slotVol === 0).toBe(true);
});

test('KW-12e: Generate Preview creates preview_audio.wav on disk (non-empty)', async ({ request, page }) => {
  resetKW12();

  const epDir      = getEpDir();
  const sourceFile = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const CLIP_ID    = 'Button_Click_Sharp_0.5s-2.0s';
  const ITEM_ID    = 'sfx-sc01-sh01-001';

  const cutResp = await request.post(`${BASE_URL}/api/sfx_cut_clip`, {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      item_id: ITEM_ID, candidate_idx: 0,
      source_file: sourceFile, title: 'Button Click Sharp',
      start_sec: 0.5, end_sec: 2.0,
    },
  });
  expect(cutResp.status()).toBe(200);
  const cutBody = await cutResp.json();
  const clipRelPath = cutBody.path;

  async function callPreview() {
    return request.post(`${BASE_URL}/api/sfx_preview`, {
      data: {
        slug: 'test-proj', ep_id: 's01e01',
        selected: {}, include_music: false,
        timing: {}, volumes: {}, duck_fade: {},
        cut_clips: [{
          clip_id: CLIP_ID, item_id: ITEM_ID,
          path: clipRelPath, duration_sec: 1.5,
        }],
        cut_assign: { [ITEM_ID]: CLIP_ID },
        clip_volumes: {},
      },
    });
  }

  let previewResp = await callPreview();
  if (previewResp.status() === 409) {
    await page.waitForTimeout(4000);
    previewResp = await callPreview();
  }
  expect(previewResp.status()).toBe(200);
  const previewBody = await previewResp.json();
  expect(previewBody.ok).toBe(true);

  const previewWav = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack', 'preview_audio.wav');
  expect(fs.existsSync(previewWav)).toBe(true);
  const stats = fs.statSync(previewWav);
  expect(stats.size).toBeGreaterThan(1024);

  const timeline = previewBody.timeline;
  expect(timeline).toBeDefined();
  expect(Array.isArray(timeline.sfx_items)).toBe(true);
  expect(timeline.sfx_items.length).toBeGreaterThan(0);
  const sfxEntry = timeline.sfx_items.find(si => si.item_id === ITEM_ID);
  expect(sfxEntry).toBeDefined();
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-16: SFX adelay timing coordinate space (migrated from kw16_sfx_timing_coordinate_space.spec.js)
// Runs render_video.py directly with a fake ffmpeg — no browser needed.
// ─────────────────────────────────────────────────────────────────────────────

test('KW-16: SFX adelay must use shot-relative timing, not episode-absolute', () => {
  const tmpRoot   = fs.mkdtempSync(path.join(os.tmpdir(), 'kw16-'));
  const epDir     = path.join(tmpRoot, 'ep');
  const binDir    = path.join(tmpRoot, 'bin');
  const ffmpegLog = path.join(tmpRoot, 'ffmpeg_calls.jsonl');

  setupEpDir(epDir);

  fs.mkdirSync(binDir);
  const fakeFfmpeg = path.join(binDir, 'ffmpeg');
  fs.writeFileSync(fakeFfmpeg, FAKE_FFMPEG_SRC);
  fs.chmodSync(fakeFfmpeg, 0o755);

  const result = spawnSync(
    'python3',
    [RENDER_PY, '--plan', path.join(epDir, 'VOPlan.en.json'), '--locale', 'en'],
    {
      env: { ...process.env, PATH: `${binDir}:${process.env.PATH}`, FFMPEG_LOG: ffmpegLog },
      encoding: 'utf8',
      timeout: 60_000,
    }
  );

  expect(
    result.status,
    `render_video.py crashed.\nstderr: ${result.stderr}`
  ).toBe(0);

  const logLines = fs.existsSync(ffmpegLog)
    ? fs.readFileSync(ffmpegLog, 'utf8').trim().split('\n').filter(Boolean)
    : [];

  const shot2Call = logLines
    .map(l => { try { return JSON.parse(l); } catch { return null; } })
    .filter(Boolean)
    .find(entry => (entry.args || []).some(a => a.includes('sc02-sh02')));

  expect(
    shot2Call,
    'Could not find ffmpeg invocation for sc02-sh02 — render_video.py may have skipped the shot.'
  ).not.toBeNull();

  const args          = shot2Call.args;
  const fcIdx         = args.indexOf('-filter_complex');
  const filterComplex = fcIdx !== -1 ? args[fcIdx + 1] : '';

  expect(filterComplex, 'No -filter_complex in sc02-sh02 ffmpeg invocation.').toBeTruthy();

  const adelayMs = parseAdelay(filterComplex);

  expect(
    adelayMs,
    'No adelay= in filter_complex — SFX not wired into ffmpeg.\n' +
    `filter_complex: ${filterComplex.slice(0, 400)}`
  ).not.toBeNull();

  expect(
    adelayMs,
    `SFX TIMING BUG: adelay=${adelayMs}ms but shot sc02-sh02 is only ${SHOT2_DURATION_MS}ms.\n` +
    `SFX placed past shot end → SILENT.\n` +
    `Root cause: render_video.py uses episode-absolute start_sec directly as shot-relative delay.\n` +
    `Fix: subtract cumulative shot offset (${SHOT2_CUMULATIVE_MS}ms).\n` +
    `Expected ≈${SFX_CORRECT_DELAY_MS}ms, got ${adelayMs}ms.`
  ).toBeLessThanOrEqual(SHOT2_DURATION_MS);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-20: sfx_save_all schema (migrated from kw20_sfx_clip_volume_roundtrip.spec.js)
// ─────────────────────────────────────────────────────────────────────────────

test('KW-20: /api/sfx_save_all writes volume_db in cut_clips, not clip_volume_db in sfx_entries', async ({ request }) => {
  resetKW12();

  const epDir       = getEpDir();
  writeFakeSearchResults(epDir);
  const clipRelPath = createFakeClipWav(epDir);

  const resp = await request.post(`${BASE_URL}/api/sfx_save_all`, {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      selected: {}, timing: {}, volumes: {}, duck_fade: {},
      cut_clips: [{
        clip_id: KW20_CLIP_ID, item_id: KW20_ITEM_ID,
        candidate_idx: 0, start_sec: 0.5, end_sec: 2.0,
        duration_sec: 1.5,
        source_file: path.join(epDir, 'assets', 'sfx', KW20_ITEM_ID, 'sfx_source_fixture.wav'),
        path: clipRelPath, volume_db: 0,
      }],
      cut_assign:   { [KW20_ITEM_ID]: KW20_CLIP_ID },
      clip_volumes: { [KW20_CLIP_ID]: KW20_CLIP_VOL },
    },
  });

  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  const planPath = path.join(epDir, 'SfxPlan.json');
  expect(fs.existsSync(planPath)).toBe(true);
  const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));

  expect(Array.isArray(plan.cut_clips)).toBe(true);
  expect(plan.cut_clips.length).toBeGreaterThan(0);
  const savedClip = plan.cut_clips.find(c => c.clip_id === KW20_CLIP_ID);
  expect(savedClip).toBeDefined();
  expect(savedClip.volume_db).toBe(KW20_CLIP_VOL);

  for (const entry of (plan.sfx_entries || [])) {
    expect(entry).not.toHaveProperty('clip_volume_db');
  }
  expect(plan).not.toHaveProperty('clip_volumes');
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-26: Preview restore on episode select (migrated from kw26_sfx_preview_restore.spec.js)
// ─────────────────────────────────────────────────────────────────────────────

test('KW-26: SFX preview wrap is visible on episode select when preview_audio.wav exists', async ({ page }) => {
  resetKW26();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);

  const headRespPromise = page.waitForResponse(
    r => r.url().includes('SfxPreviewPack') && r.url().includes('preview_audio.wav'),
    { timeout: 15000 }
  );

  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');

  const headResp = await headRespPromise;
  expect(
    headResp.status(),
    'HEAD /serve_media returned ' + headResp.status() + ' — expected 200.\n' +
    'Root cause: Handler has no do_HEAD; Python returns 501.\n' +
    'Fix: add do_HEAD = do_GET to the Handler class.'
  ).toBe(200);

  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-preview-wrap');
      return el !== null && el.style.display !== 'none';
    },
    { timeout: 12000 }
  ).catch(() => {});

  const isVisible = await page.evaluate(() => {
    const el = document.getElementById('sfx-preview-wrap');
    return el !== null && el.style.display !== 'none';
  });
  expect(
    isVisible,
    'SFX PREVIEW WRAP HIDDEN after episode select.\n' +
    'preview_audio.wav exists — the preview section must be restored automatically.'
  ).toBe(true);

  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-tl-sfx');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  ).catch(() => {});

  const barPositions = await page.evaluate(() => {
    const el = document.getElementById('sfx-tl-sfx');
    if (!el) return [];
    return Array.from(el.children).map(bar => ({
      left:  parseFloat((bar.style.left  || '0').replace('%', '')),
      width: parseFloat((bar.style.width || '0').replace('%', '')),
    }));
  });

  expect(
    barPositions.length,
    'SFX BARS MISSING: #sfx-tl-sfx has 0 children after restore.'
  ).toBeGreaterThan(0);

  for (const bar of barPositions) {
    expect(
      bar.left,
      'SFX BAR OFF-SCREEN: bar.left=' + bar.left + '% — expected < 100%.\n' +
      'Root cause: /api/sfx_timeline double-adds shot offset to episode-absolute start_sec.\n' +
      'Fix: remove _off_stl + from start_sec/end_sec in /api/sfx_timeline.'
    ).toBeLessThan(100);
  }

  // With scene_heads: {"sc01": 15} in the fixture VOPlan, /api/vo_timeline returns
  // total_sec = 70.648 (15s head + 28.089s sc01 + 27.559s sc02).
  // sfx-sc02-sh02-001 start_sec=35.0 → left = 35.0/70.648*100 ≈ 49.5%.
  // Previously (scene_heads: {"sc01": 0}): total_sec=55.648, left ≈ 62.9%.
  const maxLeft = Math.max(...barPositions.map(b => b.left));
  expect(maxLeft).toBeGreaterThan(43);
  expect(maxLeft).toBeLessThan(57);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-29: Preview restore reads all 3 timeline endpoints; total_dur_sec >= 80.0
// (migrated from kw29_sfx_preview_vo_source.spec.js)
// ─────────────────────────────────────────────────────────────────────────────

test('KW-29a: /api/vo_timeline, /api/sfx_timeline, /api/music_timeline all called in SFX restore', async ({ page }) => {
  resetKW80();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);

  const voPromise  = page.waitForResponse(r => new URL(r.url()).pathname === '/api/vo_timeline',  { timeout: 20000 });
  const sfxPromise = page.waitForResponse(r => new URL(r.url()).pathname === '/api/sfx_timeline', { timeout: 20000 });
  const musPromise = page.waitForResponse(r => new URL(r.url()).pathname === '/api/music_timeline',{ timeout: 20000 });

  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );

  const [voResp, sfxResp, musResp] = await Promise.all([
    voPromise.catch(() => null),
    sfxPromise.catch(() => null),
    musPromise.catch(() => null),
  ]);

  expect(voResp,  'VO TIMELINE NOT FETCHED in SFX preview restore.').not.toBeNull();
  expect(voResp.status()).toBe(200);
  expect(sfxResp, 'SFX TIMELINE NOT FETCHED in SFX preview restore.').not.toBeNull();
  expect(sfxResp.status()).toBe(200);
  expect(musResp, 'MUSIC TIMELINE NOT FETCHED in SFX preview restore.').not.toBeNull();
  expect(musResp.status()).toBe(200);
});

test('KW-29b: SFX timeline total_dur_sec is >= 80.0 after restore (reads VO extent)', async ({ page }) => {
  resetKW80();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-preview-wrap');
      return el !== null && el.style.display !== 'none';
    },
    { timeout: 20000 }
  ).catch(() => {});

  const totalDurSec = await page.evaluate(() => {
    const voDiv     = document.getElementById('sfx-tl-vo');
    const container = voDiv && voDiv.closest('[data-total-dur]');
    if (container) return parseFloat(container.dataset.totalDur);
    try { return (typeof _sfxTimeline !== 'undefined') ? _sfxTimeline.total_dur_sec : null; } catch(_) {}
    return null;
  });

  if (totalDurSec !== null) {
    expect(
      totalDurSec,
      'WRONG DURATION: SFX timeline total_dur_sec=' + totalDurSec + ' < 80.0.\n' +
      'Root cause: _loadAndMergeTl() uses ShotList (55.648) not max(vo.total_sec, vo_items end_sec).'
    ).toBeGreaterThanOrEqual(80.0);
  }

  await page.waitForFunction(
    () => { const el = document.getElementById('sfx-tl-vo'); return el && el.children.length > 0; },
    { timeout: 10000 }
  ).catch(() => {});

  const voBars = await page.evaluate(() => {
    const el = document.getElementById('sfx-tl-vo');
    if (!el) return [];
    return Array.from(el.children).map(bar => ({
      left:  parseFloat((bar.style.left  || '0').replace('%', '')),
      width: parseFloat((bar.style.width || '0').replace('%', '')),
    }));
  });

  if (voBars.length > 0) {
    const maxRight = Math.max(...voBars.map(b => b.left + b.width));
    expect(
      maxRight,
      'VO BAR OVERFLOW: last bar right edge at ' + maxRight + '% — ruler does not cover VO extent.\n' +
      'Correct: total_dur_sec=80.0 → ≈100%. Bug: total_dur_sec=55.648 → ≈143%.'
    ).toBeLessThanOrEqual(105);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-30: Shot Overrides reads /api/vo_timeline; totalDur reflects VO extent
// (migrated from kw30_sfx_shot_overrides_vo_source.spec.js)
// ─────────────────────────────────────────────────────────────────────────────

test('KW-30a: /api/vo_timeline is called on SFX Tab episode select', async ({ page }) => {
  resetKW80();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);

  const voPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 20000 }
  );

  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => { const el = document.getElementById('sfx-overrides'); return el && el.innerHTML.trim().length > 0; },
    { timeout: 12000 }
  );

  const voResp = await voPromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline never called when SFX Tab loaded.\n' +
    'Root cause: _sfxLoadShotTimeline() does not fetch /api/vo_timeline.\n' +
    'Fix: _sfxLoadShotTimeline() must call /api/vo_timeline for authoritative shot timing.'
  ).not.toBeNull();
  expect(voResp.status()).toBe(200);
});

test('KW-30b: SFX Shot Overrides totalDur reflects VO extent when /api/vo_timeline is used', async ({ page }) => {
  resetKW80();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-overrides');
      return el && el.innerHTML.trim().length > 0 && el.querySelectorAll('.music-shot-block').length > 0;
    },
    { timeout: 15000 }
  );

  await page.waitForFunction(
    () => { const el = document.getElementById('sfx-preview-wrap'); return el && el.style.display !== 'none'; },
    { timeout: 20000 }
  ).catch(() => {});

  await page.waitForFunction(
    () => { const el = document.getElementById('sfx-tl-vo'); return el && el.children.length > 0; },
    { timeout: 12000 }
  ).catch(() => {});

  const voBars = await page.evaluate(() => {
    const el = document.getElementById('sfx-tl-vo');
    if (!el) return [];
    return Array.from(el.children).map(bar => ({
      left:  parseFloat((bar.style.left  || '0').replace('%', '')),
      width: parseFloat((bar.style.width || '0').replace('%', '')),
    }));
  });

  if (voBars.length > 0) {
    const maxRight = Math.max(...voBars.map(b => b.left + b.width));
    expect(
      maxRight,
      'SFX VO BAR OVERFLOW: right edge at ' + maxRight + '% > 105%.\n' +
      'VO end_sec=80.0, total_dur_sec should be 80.0 → last bar ≈100%.\n' +
      'Bug: total_dur_sec=55.648 (ShotList) → ≈143%.'
    ).toBeLessThanOrEqual(105);
  }
});

test('KW-30c: /api/music_timeline NOT called on SFX episode select (Shot Overrides path only)', async ({ page }) => {
  resetKW80NoWav();

  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);

  const musPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/music_timeline',
    { timeout: 12000 }
  );

  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => { const el = document.getElementById('sfx-overrides'); return el && el.innerHTML.trim().length > 0; },
    { timeout: 12000 }
  );

  const musResp = await musPromise.catch(() => null);

  // Documents current broken state: music_timeline is NOT fetched during Shot Overrides load.
  // Once fixed, change this assertion to not.toBeNull().
  expect(
    musResp,
    'MUSIC TIMELINE WAS FETCHED unexpectedly on SFX episode select (Shot Overrides path).\n' +
    'This test documents the current broken state: music_timeline is NOT fetched\n' +
    'during Shot Overrides load. Once fixed, change this assertion to not.toBeNull().'
  ).toBeNull();
});

// ── KW-11g: SFX Shot Override last shot end label reflects VOPlan extent ──────
//
// Each SFX shot block renders "episode X.Xs – Y.Ys (Zs)" via .music-shot-hdr-ep.
// With VOPlan last end_sec=80.0, the last shot block end label must be >= 80.0.
// FAILS today: label shows 55.6s (ShotList cumulative) instead of 80.0s (VOPlan).
// Root cause: SFX Shot Overrides timing sourced from ShotList, not /api/vo_timeline.
// Fixture: resetKW80NoWav — VOPlan(end_sec=80.0) + MusicPlan + SfxPlan, NO stub WAV
// (no preview pack → _sfxTryRestorePreview fails → pure Shot Overrides load path).

test('KW-11g: SFX Shot Override last shot end label reflects VOPlan extent (>= 80.0)', async ({ page }) => {
  resetKW80NoWav();
  await openSfxAndWaitForOverrides(page);

  const lastShotEnd = await page.evaluate(() => {
    const container = document.getElementById('sfx-overrides');
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
    'No .music-shot-hdr-ep labels found in #sfx-overrides — Shot Override blocks may not have rendered'
  ).not.toBeNull();

  expect(
    lastShotEnd,
    'WRONG SOURCE: Last SFX Shot Override label shows end=' + lastShotEnd + 's < 80.0.\n' +
    'Root cause: SFX Shot Overrides timing sourced from ShotList cumulative sums\n' +
    '(sc02-sh02: 28.089 + 27.559 = 55.648s) instead of VOPlan last VO end_sec=80.0.\n' +
    'Fix: use /api/vo_timeline shot start_sec/end_sec for SFX Shot Override timing labels.'
  ).toBeGreaterThanOrEqual(80.0);
});
