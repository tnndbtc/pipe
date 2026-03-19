// TEST COVERAGE: KW-12
// Source: prompts/regression.txt § "KW-12: SFX shot-override params reach sfx_preview POST body"
//
// History: this file originally tested that sfxRenderShotOverrides did NOT subtract
// epStart from the user's timing input before calling sfxSetTiming().
//
// Current layout of .music-shot-params (one per SFX item inside each .music-shot-block):
//   input[0] — start timing   (step=0.1, min=0)      onchange → sfxSetTiming(iid,'start',…)
//   input[1] — end timing     (step=0.1, min=0)      onchange → sfxSetTiming(iid,'end',…)
//   input[2] — duck_db        (step=1,   min=-30)    onchange → _sfxSetDuckFade(iid,'duck_db',…)
//   input[3] — fade_sec       (step=0.05, min=0)     onchange → _sfxSetDuckFade(iid,'fade_sec',…)
//
// Tests MUST use attribute selectors (min=, step=) — NOT positional nth() — because the
// positional order changed when start/end timing inputs were added to the same container.
//
// KW-12:  duck_db set in Shot Override UI appears in sfx_preview POST body
//         duck_fade[item_id].duck_db
//
// KW-12b: fade_sec set in Shot Override UI appears in sfx_preview POST body
//         duck_fade[item_id].fade_sec
//
// KW-12c: sfx_cut_clip produces clip_id following music naming convention:
//         {safe_title}_{start:.1f}s-{end:.1f}s
//         After reload, Generated Clips section renders and shows clip_id in Clip column.
//
// KW-12d: clip_volumes in sfx_preview POST body is keyed by clip_id (not item_id).
//         Volume set in the Generated Clips table must travel with the clip, so that
//         re-using the same clip in a future slot picks up the right volume.
//
// KW-12e: clicking Generate Preview creates preview_audio.wav on disk (non-empty).
//         Catches: sfx_preview_pack crashes / returns early without writing audio.

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW2, resetKW12, getEpDir } = require('../helpers/fixture_state');
const fs   = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:19999';

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });

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

// ─────────────────────────────────────────────────────────────────────────────
// KW-12: duck_db value set in Shot Overrides reaches sfx_preview POST body
// ─────────────────────────────────────────────────────────────────────────────
test('KW-12: duck_db from Shot Overrides appears in sfx_preview POST body duck_fade field', async ({ page }) => {
  resetKW2();
  await openSfxAndWaitForOverrides(page);

  // sc02-sh02 is the 2nd shot block (index 1).
  // sfx-sc02-sh02-001 is the 1st SFX item in that block → 1st .music-shot-params.
  //
  // IMPORTANT: .music-shot-params has 4 inputs: start, end, duck_db, fade_sec.
  // Use attribute selectors so the test survives reordering:
  //   duck_db input has min="-30" (unique in this container)
  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  const duckInput = firstItemParams.locator('input[type="number"][min="-30"]');

  // User sets duck attenuation to -6 dB
  await duckInput.fill('-6');
  await duckInput.press('Tab');   // fires onchange → _sfxSetDuckFade(iid, 'duck_db', -6)

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

  const body = JSON.parse(req.postData());

  // duck_fade key must be present in POST body
  expect(body.duck_fade).toBeDefined();

  // The item we changed must have a duck_fade entry
  const df = body.duck_fade['sfx-sc02-sh02-001'];
  expect(df).toBeDefined();
  expect(df.duck_db).toBe(-6);

  // volumes and clip_volumes keys must always be present (even if empty objects)
  expect(body.volumes).toBeDefined();
  expect(body.clip_volumes).toBeDefined();
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-12b: fade_sec value set in Shot Overrides reaches sfx_preview POST body
// ─────────────────────────────────────────────────────────────────────────────
test('KW-12b: fade_sec from Shot Overrides appears in sfx_preview POST body duck_fade field', async ({ page }) => {
  resetKW2();
  await openSfxAndWaitForOverrides(page);

  const secondShotBlock = page.locator('#sfx-overrides .music-shot-block').nth(1);
  const firstItemParams = secondShotBlock.locator('.music-shot-params').first();
  // fade_sec input has step="0.05" (unique in this container)
  const fadeInput = firstItemParams.locator('input[type="number"][step="0.05"]');

  // User sets fade to 0.5s
  await fadeInput.fill('0.5');
  await fadeInput.press('Tab');   // fires onchange → _sfxSetDuckFade(iid, 'fade_sec', 0.5)

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

  const body = JSON.parse(req.postData());

  expect(body.duck_fade).toBeDefined();
  const df = body.duck_fade['sfx-sc02-sh02-001'];
  expect(df).toBeDefined();
  expect(df.fade_sec).toBeCloseTo(0.5, 2);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-12c: /api/sfx_cut_clip produces a clip_id and filename that follow the
//         music naming convention: {safe_title}_{start:.1f}s-{end:.1f}s
//
//         Also exercises the reload regression fix: after the cut, reload the
//         page and confirm the "Generated Clips" section appears (proving
//         sfx_cut_clips.json is now whitelisted in /api/episode_file).
// ─────────────────────────────────────────────────────────────────────────────
test('KW-12c: sfx_cut_clip returns title-based clip_id and Generated Clips section appears after reload', async ({ request, page }) => {
  resetKW12();

  const epDir        = getEpDir();
  const sourceFile   = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const TITLE        = 'Button Click Sharp';
  const START        = 0.5;
  const END          = 2.0;
  const EXPECTED_ID  = 'Button_Click_Sharp_0.5s-2.0s';
  const EXPECTED_WAV = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', EXPECTED_ID + '.wav');

  // ── Part 1: call the real Cut Clip endpoint ──────────────────────────────
  const resp = await request.post(`${BASE_URL}/api/sfx_cut_clip`, {
    data: {
      slug:          'test-proj',
      ep_id:         's01e01',
      item_id:       'sfx-sc01-sh01-001',
      candidate_idx: 0,
      source_file:   sourceFile,
      title:         TITLE,
      start_sec:     START,
      end_sec:       END,
    },
  });

  expect(resp.status()).toBe(200);
  const body = await resp.json();

  // clip_id must follow {safe_title}_{start:.1f}s-{end:.1f}s (NOT item_id-based)
  expect(body.clip_id).toBe(EXPECTED_ID);
  expect(body.duration_sec).toBeCloseTo(END - START, 1);

  // WAV file must actually exist on disk with the right name
  expect(fs.existsSync(EXPECTED_WAV)).toBe(true);

  // sfx_cut_clips.json must have been written and contain this clip
  const cutJson = path.join(epDir, 'assets', 'sfx', 'sfx_cut_clips.json');
  expect(fs.existsSync(cutJson)).toBe(true);
  const clips = JSON.parse(fs.readFileSync(cutJson, 'utf8'));
  expect(clips.some(c => c.clip_id === EXPECTED_ID)).toBe(true);

  // ── Part 2: reload the page — Generated Clips section must appear ────────
  //    (regression: sfx_cut_clips.json was 403'd by episode_file whitelist,
  //     so _sfxLoadCutClips() returned early and the section never rendered)
  await openSfxAndWaitForOverrides(page);

  const genClipsHdr = page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ });
  await expect(genClipsHdr).toBeVisible({ timeout: 6000 });

  // The Clip column (first <td> in Generated Clips table) must show clip_id as text,
  // not item_id — matches the music tab's Generated Clips column behaviour.
  const genClipsTable = page.locator('.music-card', { has: page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ }) });
  await expect(genClipsTable.locator('td').first()).toContainText(EXPECTED_ID);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-12d: clip_volumes in sfx_preview POST body is keyed by clip_id, not item_id.
//
// Volume set on a Generated Clip must travel with the clip_id so that re-using
// the same clip in a different shot slot automatically picks up the right volume.
//
// Test flow:
//   1. Cut a real clip → clip_id = Button_Click_Sharp_0.5s-2.0s
//   2. Load the SFX tab; Generated Clips section appears.
//   3. Set the clip's volume input to -6 dB.
//   4. Click Generate Preview; intercept the POST body.
//   5. Assert clip_volumes[clip_id] = -6  (volume keyed by clip_id).
//   6. Assert volumes[item_id] is NOT set to -6  (slot-level vol is separate).
// ─────────────────────────────────────────────────────────────────────────────
test('KW-12d: clip volume set in Generated Clips table appears in sfx_preview POST body under clip_volumes[clip_id]', async ({ request, page }) => {
  resetKW12();

  const epDir      = getEpDir();
  const sourceFile = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const TITLE      = 'Button Click Sharp';
  const START      = 0.5;
  const END        = 2.0;
  const CLIP_ID    = 'Button_Click_Sharp_0.5s-2.0s';
  const ITEM_ID    = 'sfx-sc01-sh01-001';

  // ── Step 1: create a real clip ───────────────────────────────────────────
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

  // ── Step 2: load SFX tab — Generated Clips section must appear ───────────
  await openSfxAndWaitForOverrides(page);
  const genClipsHdr = page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ });
  await expect(genClipsHdr).toBeVisible({ timeout: 6000 });

  // ── Step 3: set the clip volume in the Generated Clips table ─────────────
  // The volume input is inside the Generated Clips card, keyed to clip_id via
  // onchange="_sfxSetClipVolume(clip_id, this.value)" → _sfxClipVolumes[clip_id] = val
  const genClipsCard = page.locator('.music-card', {
    has: page.locator('.music-card-hdr', { hasText: /^Generated Clips$/ }),
  });
  // Volume input: step=1, min=-18, max=18 (unique in this card)
  const clipVolInput = genClipsCard.locator('input[type="number"][min="-18"]').first();
  await clipVolInput.fill('-6');
  await clipVolInput.press('Tab');   // fires _sfxSetClipVolume(clip_id, -6)

  // ── Step 4: capture Preview POST body AND wait for response ──────────────
  // waitForRequest captures the body; waitForResponse ensures the server
  // finishes before KW-12e starts (avoids 409 "already generating" race).
  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [req] = await Promise.all([
    page.waitForRequest(
      r => r.url().includes('/api/sfx_preview') && r.method() === 'POST',
      { timeout: 10000 },
    ),
    page.waitForResponse(
      r => r.url().includes('/api/sfx_preview'),
      { timeout: 30000 },
    ),
    page.click('#sfx-btn-preview'),
  ]);

  const body = JSON.parse(req.postData());

  // ── Step 5: clip_volumes must be keyed by clip_id ─────────────────────────
  expect(body.clip_volumes).toBeDefined();
  expect(body.clip_volumes[CLIP_ID]).toBe(-6);   // volume travels with clip_id

  // ── Step 6: the slot-level volumes dict must NOT carry the clip volume ────
  // volumes[item_id] should be 0 or absent — clip volume ≠ slot volume
  const slotVol = (body.volumes || {})[ITEM_ID];
  expect(slotVol === undefined || slotVol === 0).toBe(true);
});

// ─────────────────────────────────────────────────────────────────────────────
// KW-12e: clicking Generate Preview creates preview_audio.wav on disk.
//
// Catches: sfx_preview_pack crashes or returns early without writing the audio
// file, leaving the user with a silent/broken preview.
//
// Test uses a real cut clip so the render path exercises actual WAV I/O.
// The audio file must exist AND be non-empty (> 1 KB).
// ─────────────────────────────────────────────────────────────────────────────
test('KW-12e: Generate Preview creates preview_audio.wav on disk (non-empty)', async ({ request, page }) => {
  resetKW12();

  const epDir      = getEpDir();
  const sourceFile = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const TITLE      = 'Button Click Sharp';
  const START      = 0.5;
  const END        = 2.0;
  const CLIP_ID    = 'Button_Click_Sharp_0.5s-2.0s';
  const ITEM_ID    = 'sfx-sc01-sh01-001';

  // ── Step 1: create a real clip ───────────────────────────────────────────
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
  const clipRelPath = cutBody.path;   // ep_dir-relative path returned by sfx_cut_clip

  // ── Step 2: call sfx_preview via API with the cut clip ───────────────────
  // This mirrors exactly what the browser sends when the user clicks
  // Generate Preview after assigning a cut clip.
  //
  // Retry once on 409 "already generating": a prior test's preview may still
  // be finishing on the server when this test starts.
  async function callPreview() {
    return request.post(`${BASE_URL}/api/sfx_preview`, {
      data: {
        slug:          'test-proj',
        ep_id:         's01e01',
        selected:      {},
        include_music: false,
        timing:        {},
        volumes:       {},
        duck_fade:     {},
        cut_clips:     [{
          clip_id:      CLIP_ID,
          item_id:      ITEM_ID,
          path:         clipRelPath,
          duration_sec: END - START,
        }],
        cut_assign:    { [ITEM_ID]: CLIP_ID },
        clip_volumes:  {},
      },
    });
  }

  let previewResp = await callPreview();
  if (previewResp.status() === 409) {
    // Wait for the prior preview to finish, then retry once.
    await page.waitForTimeout(4000);
    previewResp = await callPreview();
  }

  expect(previewResp.status()).toBe(200);
  const previewBody = await previewResp.json();
  expect(previewBody.ok).toBe(true);

  // ── Step 3: preview_audio.wav must exist on disk and be non-empty ─────────
  // getEpDir() returns the server's temp episode dir (set via _pipeTestDir),
  // so fs reads go to the same directory the server writes to — same pattern
  // as other SfxPreviewPack file checks.
  const previewWav = path.join(getEpDir(), 'assets', 'sfx', 'SfxPreviewPack', 'preview_audio.wav');
  expect(fs.existsSync(previewWav)).toBe(true);
  const stats = fs.statSync(previewWav);
  // A valid WAV with even 1 s of audio at 44100 Hz / 16-bit mono is > 80 KB.
  // Guard against empty file or WAV header-only stub (< 1 KB).
  expect(stats.size).toBeGreaterThan(1024);

  // ── Step 4: timeline must include the cut clip as an sfx_item ────────────
  // If sfx_preview_pack silently skips the SFX (e.g. clip path can't be found),
  // ok:true is still returned but sfx_items is empty — no SFX in the audio mix.
  // This catches "Generate Preview produces VO-only audio, SFX is missing."
  const timeline = previewBody.timeline;
  expect(timeline).toBeDefined();
  expect(Array.isArray(timeline.sfx_items)).toBe(true);
  expect(timeline.sfx_items.length).toBeGreaterThan(0);
  // The specific cut clip item must appear in the timeline
  const sfxEntry = timeline.sfx_items.find(si => si.item_id === ITEM_ID);
  expect(sfxEntry).toBeDefined();
});
