// KW-3: VO tab — Generate Preview must use unsaved scene_head value from DOM.
//
// Bug: _voPreviewAll() sends only ep_dir + locale to /api/vo_preview_concat.
// The backend reads scene_heads from VOPlan on disk.  If the user changed the
// scene_head input without clicking Save, the unsaved value is silently ignored:
// the concatenated audio and the clips timing both reflect the stale saved value.
//
// KW-3a: After the user types 5 into the scene_head input (without saving) and
//         clicks Generate Preview, clips[0].start_sec in the response must be 5.0.
//         FAILS today: backend reads VOPlan scene_heads=0 → clips[0].start_sec=0.

const { test, expect } = require('@playwright/test');
const fs   = require('fs');
const path = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { getEpDir, resetKW3 } = require('../helpers/fixture_state');

const BASE_URL   = 'http://localhost:19999';

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(() => { resetKW3(); });

// ── KW-3a: Generate Preview clips[0].start_sec reflects unsaved scene_head ──────
//
// Setup:  VOPlan on disk has scene_heads={"sc01":0}.
// Action: user types 5 in #vo-head-sc01 (no Save click) → clicks Generate Preview.
// Expect: clips[0].start_sec === 5.0  (DOM value used, not stale VOPlan value).
// Fails:  clips[0].start_sec === 0.0  (backend ignores DOM, reads VOPlan=0).

test('KW-3a: Generate Preview clips include unsaved scene_head offset from DOM', async ({ page }) => {
  await page.goto(BASE_URL + '/');

  // Open VO tab (triggers populateVoEpSelect which fetches /list_projects)
  await page.click('button.tab[data-tab="vo"]');

  // Wait for the episode option to appear in the select
  await page.waitForFunction(
    () => [...document.querySelectorAll('#vo-ep-select option')]
            .some(o => o.value.includes('test-proj')),
    { timeout: 8000 }
  );

  // Select the test episode — value is the relative ep_dir path
  await page.selectOption('#vo-ep-select', 'projects/test-proj/episodes/s01e01');

  // Wait for VO items to render so the scene_head input exists
  await page.waitForFunction(
    () => document.getElementById('vo-head-sc01') !== null,
    { timeout: 8000 }
  );

  // Type 5 into the scene_head input — do NOT click Save
  await page.fill('#vo-head-sc01', '5');

  // Click Generate Preview and atomically capture the response
  const [previewResp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/vo_preview_concat'), { timeout: 15000 }),
    page.click('#vo-preview-all-btn'),
  ]);
  const previewData = await previewResp.json();
  expect(previewData, 'No response captured from /api/vo_preview_concat').not.toBeNull();
  expect(previewData.clips, 'Response missing clips array').toBeDefined();
  expect(previewData.clips.length, 'clips array is empty').toBeGreaterThan(0);

  // The first clip of sc01 must start at 5.0s (the unsaved DOM head value).
  // Bug value:  0.0  — backend reads VOPlan scene_heads=0, ignores the DOM.
  // Fixed value: 5.0 — frontend sends scene_heads in the request body/params.
  expect(
    previewData.clips[0].start_sec,
    'clips[0].start_sec should be 5.0 (unsaved DOM scene_head=5), ' +
    'got 0 because _voPreviewAll() does not send scene_heads to the backend.'
  ).toBeGreaterThanOrEqual(5.0);
});

// ── KW-3c: VO bars must reflect unsaved scene_head DOM value after Generate Preview ─
//
// Bug (two problems compounded):
//   1. _voVisLoadTimeline() is fired at the top of _voPreviewAll(), before the
//      preview fetch, and is not awaited — it races to /api/vo_timeline.
//   2. /api/vo_timeline has no scene_heads override param — it always reads
//      scene_heads from VOPlan on disk, ignoring any unsaved DOM value.
//
// Result: audio preview uses DOM scene_head (0s), bars use disk scene_head (5s).
//         /api/vo_timeline total_sec is ~5s more than previewData.total_sec.
//
// Setup:  VOPlan disk has scene_heads={"sc01": 5}. DOM set to 0 (no Save).
// Action: Click Generate Preview.
// Expect: The /api/vo_timeline request that _voVisLoadTimeline makes (with the
//         scene_heads override param) returns total_sec ≈ previewData.total_sec (delta < 1s).
// Fails:  Without the fix, _voVisLoadTimeline calls /api/vo_timeline without
//         scene_heads, so bars read disk=5s while audio used DOM=0s → delta ≈ 5s.
//
// NOTE: The test captures the browser's actual /api/vo_timeline call (which
//       _voPreviewAll fires after the preview completes with scene_heads in the URL).
//       It does NOT call /api/vo_timeline directly — that would bypass the fix.

test('KW-3c: VO bars reflect unsaved scene_head DOM value after Generate Preview', async ({ page }) => {
  // beforeEach ran resetKW3() → disk scene_heads={"sc01":0}, vo_items[0].start_sec=0.
  // Patch disk so scene_heads=5 AND vo_items[0].start_sec=5 (simulating a saved plan
  // that was originally generated with sc01=5).  The user then sets DOM to 0 without
  // saving — so disk says 5s head, DOM says 0s head.
  //
  // The visual bug: _tlRender reads vo_items[0].start_sec from /api/vo_timeline, which
  // reads VOPlan on disk.  With disk vo_items[0].start_sec=5, the bar draws at ~9%
  // even though the audio (DOM sc01=0) starts immediately at 0s.
  //
  // The fix: after Generate Preview, bars must be driven from previewData.clips
  // (which already carry DOM-based start_sec=0), not from VOPlan disk start_sec.
  const ep = getEpDir();
  const vp = JSON.parse(fs.readFileSync(path.join(ep, 'VOPlan.en.json'), 'utf8'));
  vp.scene_heads = { sc01: 5 };
  // Shift all sc01 vo_items forward by 5s to match the saved scene_head=5.
  // This is what VOPlan looks like when scene_heads was 5 at plan-generation time.
  const HEAD_DELTA = 5.0;
  vp.vo_items = vp.vo_items.map(it => {
    if (!it.item_id.startsWith('vo-sc01-')) return it;
    return { ...it, start_sec: it.start_sec + HEAD_DELTA, end_sec: it.end_sec + HEAD_DELTA };
  });
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));

  // Open VO tab and select episode
  await page.goto(BASE_URL + '/');
  await page.click('button.tab[data-tab="vo"]');
  await page.waitForFunction(
    () => [...document.querySelectorAll('#vo-ep-select option')]
            .some(o => o.value.includes('test-proj')),
    { timeout: 8000 }
  );
  await page.selectOption('#vo-ep-select', 'projects/test-proj/episodes/s01e01');
  await page.waitForFunction(
    () => document.getElementById('vo-head-sc01') !== null,
    { timeout: 8000 }
  );

  // Set DOM scene_head to 0 — do NOT click Save (disk still has 5)
  await page.fill('#vo-head-sc01', '0');

  // Click Generate Preview and atomically capture:
  //   1. The preview concat response (total_sec based on DOM scene_heads=0)
  //   2. The /api/vo_timeline call that _voVisLoadTimeline makes AFTER the
  //      preview completes — must include scene_heads in the URL.
  //   3+4. /api/sfx_timeline and /api/music_timeline — _loadAndMergeTl fetches
  //      all three in parallel; _tlRender only runs after ALL THREE complete.
  //      We must wait for 3+4 too, otherwise we read the bar before _tlRender fires.
  const [, tlResp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/vo_preview_concat'),
      { timeout: 15000 }
    ),
    page.waitForResponse(
      r => r.url().includes('/api/vo_timeline') && r.url().includes('scene_heads'),
      { timeout: 15000 }
    ),
    page.waitForResponse(
      r => r.url().includes('/api/sfx_timeline'),
      { timeout: 15000 }
    ),
    page.waitForResponse(
      r => r.url().includes('/api/music_timeline'),
      { timeout: 15000 }
    ),
    page.click('#vo-preview-all-btn'),
  ]);
  // All three parallel fetches inside _loadAndMergeTl have now returned.
  // Flush the browser's microtask queue so _tlRender() has run before we read the DOM.
  await page.evaluate(() => new Promise(r => setTimeout(r, 0)));

  // The /api/vo_timeline URL must carry the DOM value (sc01=0), not the disk value (sc01=5).
  // If _voPreviewAll did NOT pass DOM scene_heads to _voVisLoadTimeline, the URL would
  // have no scene_heads param at all → waitForResponse above would time out (KW-3c fails).
  // If it passed the wrong (disk) value, sc01 would be 5, not 0.
  const tlUrl        = new URL(tlResp.url());
  const shParam      = tlUrl.searchParams.get('scene_heads');
  const sceneHeadsInUrl = JSON.parse(decodeURIComponent(shParam || '{}'));

  expect(
    sceneHeadsInUrl.sc01,
    `scene_heads in /api/vo_timeline URL has sc01=${sceneHeadsInUrl.sc01}, ` +
    'expected 0 (unsaved DOM value). ' +
    'Bug: _voPreviewAll passed disk value (5) instead of DOM value (0) to _voVisLoadTimeline.'
  ).toBe(0);

  // Also confirm: the DOM-based timeline (sc01=0) is shorter than the disk-based one (sc01=5).
  // This proves the param actually changes the response, not just the URL.
  const tlData   = await tlResp.json();
  const diskResp = await page.request.get(
    `${BASE_URL}/api/vo_timeline?slug=test-proj&ep_id=s01e01` +
    `&scene_heads=${encodeURIComponent(JSON.stringify({ sc01: 5 }))}`
  );
  const diskData = await diskResp.json();

  expect(
    diskData.total_sec - tlData.total_sec,
    `Disk timeline (sc01=5, ${diskData.total_sec}s) should be ~5s longer than ` +
    `DOM timeline (sc01=0, ${tlData.total_sec}s). ` +
    'The scene_heads override param has no effect on the backend.'
  ).toBeGreaterThan(4.0);

  // ── Visual bar check ─────────────────────────────────────────────────────────
  // After _voVisLoadTimeline runs and calls _tlRender('vo-vis', ...), each bar div
  // in #vo-vis-tl-vo gets:  left = v.start_sec / total_dur_sec * 100%.
  //
  // Bug:  v.start_sec comes from VOPlan disk (5.0s) → first bar left ≈ 9%.
  //       The audio preview starts at 0s but the bar shows 5s of silence.
  //
  // Fix:  bars must be driven from previewData.clips (DOM-based start_sec=0)
  //       → first bar left ≈ 0%.
  //
  const firstBarLeftPct = await page.evaluate(() => {
    const bar = document.querySelector('#vo-vis-tl-vo > div');
    return bar ? parseFloat(bar.style.left) : null;
  });

  // DOM clips[0].start_sec=0 → after fix bar left ≈ 0%.
  // Bug: VOPlan disk start_sec=5 → bar left ≈ 5/total_dur*100 ≈ 9%.
  expect(
    firstBarLeftPct,
    `First VO bar in #vo-vis-tl-vo is at ${firstBarLeftPct?.toFixed(2)}% — ` +
    'it reads VOPlan disk vo_items[0].start_sec=5 (9%) instead of ' +
    'previewData.clips[0].start_sec=0 (0%). ' +
    'Fix: drive #vo-vis-tl-vo bars from previewData.clips after Generate Preview.'
  ).toBeLessThan(1.0);
});

// ── KW-3b: Generate Preview actual audio duration matches reported total_sec ──
//
// Real bug: vo_preview_concat sets SAMPLE_RATE from the FIRST WAV only, then
// dumps raw PCM bytes from every subsequent WAV into that container without
// resampling.  When one item was re-generated at a different rate (e.g. 16000 Hz
// while the rest are 24000 Hz), its PCM segment is the wrong byte-count for the
// output rate → that item plays at the wrong speed (cartoon / fast-forward).
//
// Concretely: a 16000 Hz item with 16000 frames = 1.000 s at 16000 Hz.
// Dumped into a 24000 Hz container: 32000 bytes / (24000×2) = 0.667 s actual.
// The backend reports duration_sec=1.000 s (correct) but the WAV only contains
// 0.667 s of audio for that item → the segment plays 1.5× too fast.
//
// Setup:  Plant WAVs with MIXED sample rates:
//           - first item:  24000 Hz, 24000 frames → 1.000 s
//           - second item: 16000 Hz, 16000 frames → 1.000 s (same reported dur)
//           - remaining:   24000 Hz, 24000 frames → 1.000 s each
// Action: Click Generate Preview.
// Expect: actual WAV audio duration ≈ previewData.total_sec (delta < 0.05 s).
// Fails:  actual WAV audio duration < total_sec by ~0.333 s (the 16k item is
//         0.333 s shorter in the output than total_sec claims).

test('KW-3b: Generate Preview actual audio duration matches reported total_sec for mixed-rate WAVs', async ({ page }) => {
  const ep    = getEpDir();
  const voDir = path.join(ep, 'assets', 'en', 'audio', 'vo');

  // Helper: build a minimal mono 16-bit PCM WAV with `nFrames` silent frames at `rate` Hz.
  function makeWav(rate, nFrames) {
    const dataSize = nFrames * 2;  // 16-bit mono = 2 bytes/frame
    const buf = Buffer.alloc(44 + dataSize);
    buf.write('RIFF', 0, 'ascii');  buf.writeUInt32LE(36 + dataSize, 4);
    buf.write('WAVE', 8, 'ascii');  buf.write('fmt ', 12, 'ascii');
    buf.writeUInt32LE(16, 16);      buf.writeUInt16LE(1, 20);   // PCM
    buf.writeUInt16LE(1,  22);      buf.writeUInt32LE(rate, 24);
    buf.writeUInt32LE(rate * 2, 28); buf.writeUInt16LE(2, 32);
    buf.writeUInt16LE(16, 34);      buf.write('data', 36, 'ascii');
    buf.writeUInt32LE(dataSize, 40);
    // data bytes already zero (silent)
    return buf;
  }

  // Read VOPlan to get the ordered item list — we need to know which file is "second"
  const vp       = JSON.parse(fs.readFileSync(path.join(ep, 'VOPlan.en.json'), 'utf8'));
  const voItems  = vp.vo_items || [];
  expect(voItems.length, 'Need at least 2 VO items for this test').toBeGreaterThanOrEqual(2);

  // Plant WAVs: all items get 24000 Hz / 24000 frames (= 1.000 s) …
  fs.mkdirSync(voDir, { recursive: true });
  const wav24k = makeWav(24000, 24000);  // 1.000 s at 24000 Hz
  const wav16k = makeWav(16000, 16000);  // 1.000 s at 16000 Hz (same reported duration)
  voItems.forEach((item, idx) => {
    const dest = path.join(voDir, `${item.item_id}.wav`);
    // Second item gets 16000 Hz — mimics "re-generated at a different rate"
    fs.writeFileSync(dest, idx === 1 ? wav16k : wav24k);
  });

  // Navigate to VO tab and select episode
  await page.goto(BASE_URL + '/');
  await page.click('button.tab[data-tab="vo"]');
  await page.waitForFunction(
    () => [...document.querySelectorAll('#vo-ep-select option')]
            .some(o => o.value.includes('test-proj')),
    { timeout: 8000 }
  );
  await page.selectOption('#vo-ep-select', 'projects/test-proj/episodes/s01e01');
  await page.waitForFunction(
    () => document.getElementById('vo-preview-all-btn') !== null,
    { timeout: 8000 }
  );

  // Click Generate Preview and atomically capture the response
  const [previewResp] = await Promise.all([
    page.waitForResponse(r => r.url().includes('/api/vo_preview_concat'), { timeout: 15000 }),
    page.click('#vo-preview-all-btn'),
  ]);
  const previewData = await previewResp.json();

  expect(previewData,           'No response from /api/vo_preview_concat').not.toBeNull();
  expect(previewData.wav_url,   'Response missing wav_url').toBeDefined();
  expect(previewData.total_sec, 'Response missing total_sec').toBeDefined();

  // Fetch the generated WAV
  const wavResp = await page.request.get(BASE_URL + previewData.wav_url);
  expect(wavResp.ok(), 'Failed to fetch generated preview WAV').toBe(true);
  const wavBytes = await wavResp.body();

  // Actual audio duration = PCM data bytes / (outputSampleRate × bytesPerSample × channels)
  const outSampleRate   = wavBytes.readUInt32LE(24);   // fmt chunk, bytes 24–27
  const pcmDataSize     = wavBytes.readUInt32LE(40);   // data chunk size, bytes 40–43
  const actualAudioSec  = pcmDataSize / (outSampleRate * 2);  // 16-bit mono

  // total_sec is computed from getnframes()/getframerate() per item — correct regardless of mismatch.
  // actualAudioSec is what the browser actually plays.
  // If the 16k item's PCM was dumped into a 24k container without resampling:
  //   that item contributes 32000 bytes instead of 48000 bytes,
  //   so actualAudioSec < total_sec by ≈ 0.333 s.
  expect(
    Math.abs(actualAudioSec - previewData.total_sec),
    `Actual WAV audio duration (${actualAudioSec.toFixed(3)}s) differs from ` +
    `reported total_sec (${previewData.total_sec}s) by ` +
    `${Math.abs(actualAudioSec - previewData.total_sec).toFixed(3)}s. ` +
    'The second item (16000 Hz) was written into a 24000 Hz container without resampling — ' +
    'its segment is 0.333s shorter than claimed, causing cartoon/fast-forward playback.'
  ).toBeLessThan(0.05);
});
