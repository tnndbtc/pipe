// TEST COVERAGE: KW-2 (merged KW-9, KW-10, KW-14, KW-25, KW-27, KW-28)
// Source: prompts/regression.txt § "KW-2: Music Review Preview"
//
// KW-2a/2b : Generate Preview path      — fixture: resetKW2 (no MusicPlan, end_sec=80.0)
// KW-2c–2f : Shot Overrides path        — fixture: resetKW80 (MusicPlan present, inline)
// KW-9     : no-clip selection persists — fixture: resetKW13 (MusicPlan, original VOPlan, inline)
// KW-10    : music bar end position     — fixture: resetKW13 (inline)
// KW-14    : music stops at end_sec     — fixture: resetKW13 (inline)
// KW-25    : VO bars render in #music-tl-vo on tab load — fixture: resetKW13 (inline)
const { test, expect } = require('@playwright/test');
const { execSync }     = require('child_process');
const fs               = require('fs');
const os               = require('os');
const path             = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, resetKW80, resetKW13, getEpDir, musicplan } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW2(); });

// Opens Music tab and waits for Generate Preview button to be enabled.
// Requires NO MusicPlan (resetKW2 fixture — fresh start).
async function openMusicTab(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => !document.getElementById('music-btn-review').disabled,
    { timeout: 12000 }
  );
}

// Opens Music tab, clicks Generate Preview, and waits for clip select dropdowns.
// Requires MusicPlan to exist — call resetKW13() or resetKW80() before using this helper.
async function openMusicTabGenerateAndWaitForClips(page) {
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
  // Wait for shot override clip selects to render
  await page.locator('select[id^="music-clip-"]').first().waitFor({ state: 'visible', timeout: 8000 });
}

// Opens Music tab and waits for _musicLoadExisting() to render Shot Override cards.
// Sets up response interceptors BEFORE episode select so no calls are missed.
// Requires MusicPlan to exist — call resetKW80() before using this helper.
async function openMusicTabAndWaitForLoad(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);

  const voPromise  = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 20000 }
  );
  const musPromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/music_timeline',
    { timeout: 20000 }
  );

  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.querySelector('#music-body .music-card') !== null,
    { timeout: 15000 }
  );

  return { voPromise, musPromise };
}

// ── KW-2a: Generate Preview fires music_review_pack ───────────────────────────
//
// Also checks (migrated from KW-27a/27b):
//   KW-27a: /api/vo_timeline is fetched during Generate Preview
//   KW-27b: total_duration_sec in response reflects VO extent (>= 90.0)

test('KW-2a: Generate Preview fires music_review_pack and gets 200', async ({ page }) => {
  await openMusicTab(page);

  // KW-27a: arm vo_timeline intercept before clicking so the call is not missed
  const voTimelinePromise = page.waitForResponse(
    r => new URL(r.url()).pathname === '/api/vo_timeline',
    { timeout: 20000 }
  );

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

  // KW-27a: /api/vo_timeline must have been called during Generate Preview.
  // Fails if musicGenerateReview() never fetches vo_timeline — VO extent unknown.
  const voResp = await voTimelinePromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline was never called during Generate Preview.\n' +
    'musicGenerateReview() must fetch /api/vo_timeline to know the VO extent (end_sec=80.0).'
  ).not.toBeNull();
  expect(voResp.status()).toBe(200);

  // KW-27b: total_duration_sec must reflect VO extent including pause_after_ms.
  // Fixture: last VO end_sec=80.0, pause_after_ms=10000 → full tail = 90.0s.
  // build_timeline() ignores pause_after_ms → returns max(55.648, 80.0)=80.0 → FAIL.
  expect(
    body.timeline.total_duration_sec,
    'WRONG DURATION: total_duration_sec=' + body.timeline.total_duration_sec +
    ' < 90.0. build_timeline ignores pause_after_ms=10000 (10s tail). ' +
    'Fix: total_duration_sec = max(ShotList_cum, end_sec + pause_after_ms/1000) = 90.0.'
  ).toBeGreaterThanOrEqual(90.0);
});

// ── KW-2b: Confirm Plan writes MusicPlan.json ─────────────────────────────────

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

// ── KW-2c: Shot Overrides load calls /api/vo_timeline (migrated KW-28a) ───────
//
// _musicLoadExisting() must fetch /api/vo_timeline on episode select so that
// VO extent (end_sec=80.0) is known before rendering Shot Override blocks.

test('KW-2c: /api/vo_timeline is called on Music Tab episode select', async ({ page }) => {
  resetKW80();
  const { voPromise } = await openMusicTabAndWaitForLoad(page);

  const voResp = await voPromise.catch(() => null);
  expect(
    voResp,
    'VO TIMELINE NOT FETCHED: /api/vo_timeline was never called when Music Tab loaded.\n' +
    'Root cause: _musicLoadExisting() skips /api/vo_timeline fetch.\n' +
    'Fix: add fetch(/api/vo_timeline) in _musicLoadExisting().'
  ).not.toBeNull();
  expect(voResp.status()).toBe(200);
});

// ── KW-2d: Shot Overrides load calls /api/music_timeline (migrated KW-28c) ───

test('KW-2d: /api/music_timeline is called on Music Tab episode select', async ({ page }) => {
  resetKW80();
  const { musPromise } = await openMusicTabAndWaitForLoad(page);

  const musResp = await musPromise.catch(() => null);
  expect(
    musResp,
    'MUSIC TIMELINE NOT FETCHED: /api/music_timeline was never called when Music Tab loaded.\n' +
    'Shot Overrides section requires music_timeline for MusicPlan assignments.'
  ).not.toBeNull();
  expect(musResp.status()).toBe(200);
});

// ── KW-2e: Shot Timeline "Total:" text reflects VO extent (migrated KW-27c/28b) ─
//
// The "Shot Timeline" card sub-heading "Total: X.Xs — N shots" must be >= 90.0.
// If total_duration_sec is sourced from ShotList (55.648) it fails.

test('KW-2e: Music Shot Timeline "Total:" shows >= 90.0s after load', async ({ page }) => {
  resetKW80();
  await openMusicTabAndWaitForLoad(page);

  const totalText = await page.evaluate(() => {
    const subs = Array.from(document.querySelectorAll('.music-card-sub'));
    const tl = subs.find(s => s.textContent.includes('Total:'));
    return tl ? tl.textContent.trim() : null;
  });

  expect(
    totalText,
    'Shot Timeline card sub-heading not found — _musicLoadExisting() may not have rendered'
  ).not.toBeNull();

  const match = totalText.match(/Total:\s*([\d.]+)s/);
  const totalSec = match ? parseFloat(match[1]) : 0;

  expect(
    totalSec,
    'WRONG DURATION: Music Shot Timeline shows Total=' + totalSec + 's < 90.0.\n' +
    'Root cause: total_duration_sec = max(ShotList_cum=55.648, end_sec=80.0) = 80.0,\n' +
    'ignoring pause_after_ms=10000 → 80.0 < 90.0.\n' +
    'Fix: total_duration_sec = max(ShotList_cum, end_sec + pause_after_ms/1000) = 90.0.'
  ).toBeGreaterThanOrEqual(90.0);
});

// ── KW-2f: Shot Override per-shot labels must reflect VOPlan extent ───────────
//
// Each shot block renders "episode X.Xs – Y.Ys (Zs)" sourced from
// _musicTimeline.shots[].offset_sec / duration_sec, which are overwritten
// from ShotList cumulative sums in JS (sc02-sh02 end = 28.089+27.559 = 55.648s).
//
// With last VO end_sec=80.0, the last shot block end label must be >= 80.0.
// FAILS today: label shows 55.6s (ShotList) instead of 80.0s (VOPlan). → BUG CAUGHT.

test('KW-2f: Shot Override last shot end label reflects VOPlan extent (>= 80.0)', async ({ page }) => {
  resetKW80();
  await openMusicTabAndWaitForLoad(page);

  const lastShotEnd = await page.evaluate(() => {
    const labels = Array.from(document.querySelectorAll('.music-shot-hdr-ep'));
    if (!labels.length) return null;
    // Text: "episode\u00a028.1s – 55.6s\u00a0(27.6s)"
    // Normalise non-breaking spaces then split on en-dash separator
    const text = labels[labels.length - 1].textContent.replace(/\u00a0/g, ' ');
    const parts = text.split(' – ');
    if (parts.length < 2) return null;
    const m = parts[1].match(/^([\d.]+)s/);
    return m ? parseFloat(m[1]) : null;
  });

  expect(
    lastShotEnd,
    'No .music-shot-hdr-ep labels found — Shot Override blocks may not have rendered'
  ).not.toBeNull();

  expect(
    lastShotEnd,
    'WRONG SOURCE: Last Shot Override label shows end=' + lastShotEnd + 's < 80.0.\n' +
    'Root cause: _musicTimeline.shots offset_sec/duration_sec overwritten from ShotList\n' +
    '(sc02-sh02: 28.089 + 27.559 = 55.648s) instead of VOPlan last VO end_sec=80.0.\n' +
    'Fix: use /api/vo_timeline shot start_sec/end_sec for Shot Override timing labels.'
  ).toBeGreaterThanOrEqual(80.0);
});

// ── KW-9: no-clip selection persists after tab switch (migrated from kw9) ─────
//
// Regression: selecting "— no clip —" (music_clip_id="") in a shot override,
// then Confirm Plan, then tab-switch back to Music tab should show "" still selected.
// Fixture: resetKW13 (MusicPlan + music WAVs present, VOPlan unpatched).

test('KW-9: no-clip selection persists after tab switch', async ({ page }) => {
  resetKW13();
  await openMusicTabGenerateAndWaitForClips(page);

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

// ── KW-10: green music bar ends at end_sec after start/end override ────────────
//
// Regression: green bar ignores end_sec override and extends to full shot duration.
// Set start=5, end=12 on first shot, re-generate preview, assert bar endPct < 30%.
// FAILS with bug: end uses shot.duration_sec=28.089 → endPct ≈ 46.3%.
// PASSES with fix: end=12 → endPct ≈ 19.8%.
// Fixture: resetKW13 (MusicPlan + music WAVs present, VOPlan unpatched).

test('KW-10: green music bar ends at end_sec after start/end override', async ({ page }) => {
  resetKW13();
  await openMusicTabGenerateAndWaitForClips(page);

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

// ── KW-14: music stops at end_sec=35 (not at wav_duration 59.6s) ─────────────
//
// Regression: media_preview_pack.py builds a delay_ms from start_sec and plays
// the WAV for its FULL duration. end_sec and clip_duration_sec from MusicPlan
// are never used to trim the clip.
//
//   music-sc01-sh01  start_sec=5   end_sec=20   WAV=28.089s → plays until 33.1s (13s overshoot)
//   music-sc02-sh02  start_sec=30  end_sec=35   WAV=29.559s → plays until 59.6s (24s overshoot)
//
// Detection: run media_preview twice (with/without music). Measure RMS diff:
//   Window A t=31s: INSIDE sc02 music window  → diff >> 0 (sanity check)
//   Window B t=36s: OUTSIDE sc02 music window → diff ≈ 0 after fix, >> 0 before fix.
//
// KEY ASSERTION: diff_after < RMS_THRESHOLD — FAILS before fix.
// Fixture: resetKW13 (MusicPlan + music WAVs present, VOPlan unpatched).

// Threshold in 16-bit PCM amplitude units.
// Music at BASE_MUSIC_DB_PREVIEW = -6 dB → amplitude ≈ 16 000.
// Silence / rounding noise stays well under 300.
const RMS_THRESHOLD = 500;

// Python helper: RMS amplitude of a 1-second window inside an MP4/WAV.
// Writes a temp script file to avoid shell quoting issues with multi-line code.
const RMS_SCRIPT = path.join(os.tmpdir(), '_kw14_rms_helper.py');
fs.writeFileSync(RMS_SCRIPT, `
import subprocess, wave, struct, math, sys

def rms_seg(p, t, dur=1.0):
    tmp = '/tmp/_kw14_rms.wav'
    r = subprocess.run(
        ['ffmpeg', '-y', '-ss', str(t), '-t', str(dur),
         '-i', p, '-ac', '1', '-ar', '44100', '-f', 'wav', tmp],
        capture_output=True)
    if r.returncode != 0:
        return 0.0
    with wave.open(tmp) as w:
        frames = w.readframes(w.getnframes())
    if not frames:
        return 0.0
    s = [struct.unpack_from('<h', frames, i * 2)[0] for i in range(len(frames) // 2)]
    return math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0

print(f'{rms_seg(sys.argv[1], float(sys.argv[2])):.2f}')
`);

function rmsAt(filePath, startSec) {
  return parseFloat(
    execSync(`python3 ${RMS_SCRIPT} ${JSON.stringify(filePath)} ${startSec}`)
      .toString().trim()
  );
}

// ── KW-25: VO timeline bars render in #music-tl-vo on Music Tab load ─────────
//
// _musicLoadExisting() fetches only /api/music_timeline, which passes empty
// dicts for VOPlan → shots have no vo_lines field → voItems = [] → 0 bars in
// #music-tl-vo.
// Fix: fetch /api/vo_timeline in _musicLoadExisting() and seed voItems from
// tl.vo_items in _musicRenderTimeline().
// No Generate Preview click needed — _musicLoadExisting() runs on episode select.
// Fixture: resetKW13 (VOPlan + MusicPlan + music WAVs, 9 vo_items).

test('KW-25: VO timeline bars render in #music-tl-vo on Music Tab load', async ({ page }) => {
  resetKW13();
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');

  // Sentinel: _musicRenderBody() creates #music-tl-vo in the same innerHTML as
  // #music-tl-music. Once it exists, _musicRenderTimeline() has already run.
  await page.waitForFunction(
    () => document.getElementById('music-tl-vo') !== null,
    { timeout: 15000 }
  );

  const voBarCount = await page.evaluate(
    () => document.getElementById('music-tl-vo')?.children?.length ?? 0
  );

  expect(
    voBarCount,
    'VO TIMELINE MISSING: #music-tl-vo has 0 children after Music Tab load.\n' +
    'VOPlan.en.json has 9 vo_items with start_sec/end_sec — VO bars must be\n' +
    'visible in the Music Tab preview.\n' +
    'Root cause: _musicLoadExisting() never fetches /api/vo_timeline, so\n' +
    '_musicRenderTimeline() has no vo_items to render.'
  ).toBeGreaterThanOrEqual(1);
});

// ── KW-14: music stops at end_sec=35 (not at wav_duration 59.6s) ─────────────

test('KW-14: music stops at end_sec=35 (not at wav_duration 59.6s)', async ({ request }) => {
  resetKW13();

  // ── Run preview WITH music ─────────────────────────────────────────────────
  const withResp = await request.post('/api/media_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selections:    {},
      include_music: true,
      include_sfx:   false,
      out_name:      'kw14_with_music.mp4',
    },
  });
  expect(withResp.ok()).toBe(true);
  const withBody = await withResp.json();
  expect(withBody.ok).toBe(true);

  // ── Run preview WITHOUT music (VO-only baseline) ───────────────────────────
  const noResp = await request.post('/api/media_preview', {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      selections:    {},
      include_music: false,
      include_sfx:   false,
      out_name:      'kw14_no_music.mp4',
    },
  });
  expect(noResp.ok()).toBe(true);
  const noBody = await noResp.json();
  expect(noBody.ok).toBe(true);

  const packDir  = path.join(getEpDir(), 'assets', 'media', 'MediaPreviewPack');
  const withMp4  = path.join(packDir, 'kw14_with_music.mp4');
  const noMp4    = path.join(packDir, 'kw14_no_music.mp4');

  // ── Measure RMS amplitude at two time windows ──────────────────────────────
  // Window A: t=31s — INSIDE sc02 music window (start_sec=30, end_sec=35).
  // Sanity check: with_music should be louder than no_music here.
  const rmsWithInside = rmsAt(withMp4, 31);
  const rmsNoInside   = rmsAt(noMp4,   31);
  const diffInside    = Math.abs(rmsWithInside - rmsNoInside);

  // Window B: t=36s — OUTSIDE sc02 music window (1s after end_sec=35).
  // After fix  : music has stopped → diff ≈ 0.
  // Before fix : music still playing (WAV runs to ~59.6s) → diff ≈ 11 000+.
  const rmsWithAfter  = rmsAt(withMp4, 36);
  const rmsNoAfter    = rmsAt(noMp4,   36);
  const diffAfter     = Math.abs(rmsWithAfter - rmsNoAfter);

  // Sanity: music IS present before end_sec (ensures the fixture is correct)
  expect(diffInside).toBeGreaterThan(RMS_THRESHOLD);

  // KEY ASSERTION — FAILS before fix:
  // Music must NOT be audible after end_sec=35.
  // (Before fix: diffAfter ≈ 11 000+, so this fails.)
  expect(diffAfter).toBeLessThan(RMS_THRESHOLD);
});
