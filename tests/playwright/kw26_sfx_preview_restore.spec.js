// KW-26: SFX tab PREVIEW section must auto-restore on episode select when
//        preview_audio.wav already exists on disk from a previous generate.
//
// BUG A — missing do_HEAD:
//   _sfxTryRestorePreview() sends HEAD /serve_media?path=... to check WAV
//   existence.  Handler has no do_HEAD → Python returns 501 → audioCheck.ok
//   is false → function returns early → #sfx-preview-wrap stays hidden.
//   Fix: add do_HEAD = do_GET to the Handler class.
//
// BUG B — sfx_timeline double-adds shot offset:
//   After the HEAD check passes, _loadAndMergeTl() calls /api/sfx_timeline
//   which reads SfxPlan.sfx_entries[].start_sec/end_sec.  Those values are
//   episode-absolute (timing_format:"episode_absolute"), but the endpoint
//   treats them as shot-relative and adds the shot's episode offset on top:
//     "start_sec": round(_off_stl + float(entry["start_sec"]), 3)
//   For sc02-sh02 (shot offset 28.089s, start_sec=35.0):
//     BUG:   28.089 + 35.0 = 63.089s → bar left = 113.4%  (off-screen)
//     CORRECT:          35.0s        → bar left =  62.9%
//   sc01-sh01 is accidentally correct because its shot offset is 0.0s.
//   Fix: remove _off_stl + from the start_sec / end_sec expressions.
//
// Sentinel: intercept the HEAD request to SfxPreviewPack/preview_audio.wav.
//   Without BUG A fix: HEAD → 501 → fails at status assertion.
//   With both fixes: bars render at correct positions.
//
// Fixture: resetKW26 — VOPlan + MusicPlan + SfxPlan + music WAVs +
//          stub SfxPreviewPack/preview_audio.wav.

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW26 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW26(); });

test('KW-26: SFX preview wrap is visible on episode select when preview_audio.wav exists', async ({ page }) => {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);

  // Intercept the HEAD request _sfxTryRestorePreview() sends to check WAV existence.
  // Must be set up before the episode select that triggers onSfxEpChange().
  const headRespPromise = page.waitForResponse(
    r => r.url().includes('SfxPreviewPack') && r.url().includes('preview_audio.wav'),
    { timeout: 15000 }
  );

  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');

  const headResp = await headRespPromise;
  expect(
    headResp.status(),
    'HEAD /serve_media returned ' + headResp.status() + ' — expected 200.\n' +
    'Root cause: Handler has no do_HEAD; Python returns 501 Not Implemented.\n' +
    'Fix: add do_HEAD = do_GET to the Handler class.'
  ).toBe(200);

  // HEAD returned 200 → _sfxTryRestorePreview() calls _loadAndMergeTl() which
  // awaits three parallel fetches (vo_timeline, sfx_timeline, music_timeline).
  // Only after ALL three complete does it set style.display = ''.
  // Use waitForFunction on the actual visibility — this is the observable end state.
  // (waitForResponse on any single fetch resolves too early, before the Promise.all
  // settles and before display is updated.)
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-preview-wrap');
      return el !== null && el.style.display !== 'none';
    },
    { timeout: 12000 }
  ).catch(() => {});   // timeout → fall through to the expect for a clear message

  const isVisible = await page.evaluate(
    () => {
      const el = document.getElementById('sfx-preview-wrap');
      return el !== null && el.style.display !== 'none';
    }
  );
  expect(
    isVisible,
    'SFX PREVIEW WRAP HIDDEN: #sfx-preview-wrap is still display:none after episode select.\n' +
    'preview_audio.wav exists on disk — the preview section must be restored automatically.'
  ).toBe(true);

  // ── BUG B: sfx_timeline double-adds shot offset to episode-absolute values ──
  //
  // SfxPlan.json stores episode-absolute start/end_sec (timing_format:
  // "episode_absolute").  /api/sfx_timeline must return them as-is.
  // If it adds the shot offset on top, bars for non-zero-offset shots land
  // off-screen (left > 100%).
  //
  // Fixture values (episode-absolute, from SfxPlan.json):
  //   sc01-sh01-001: start=5.0s  → left = 5.0/55.648*100 =  8.98%  (shot offset 0 → same either way)
  //   sc02-sh02-001: start=35.0s → left = 35.0/55.648*100 = 62.9%  CORRECT
  //                 BUG gives:    (28.089+35.0)/55.648*100 = 113.4% OFF-SCREEN
  //
  // Wait for #sfx-tl-sfx bars to be rendered by _tlRender() after restore.
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
    'SFX BARS MISSING: #sfx-tl-sfx has 0 children after restore.\n' +
    'SfxPlan.json has 2 sfx_entries — bars must be rendered on restore.'
  ).toBeGreaterThan(0);

  // Every bar must be within the timeline (left < 100%).
  // FAILS with double-add bug: sc02-sh02 bar lands at left ≈ 113.4%.
  for (const bar of barPositions) {
    expect(
      bar.left,
      'SFX BAR OFF-SCREEN: bar.left=' + bar.left + '% — expected < 100%.\n' +
      'Root cause: /api/sfx_timeline adds shot offset to already-episode-absolute\n' +
      'SfxPlan start_sec values (sc02-sh02 offset=28.089s, start_sec=35.0 → 113.4%).\n' +
      'Fix: remove _off_stl + from start_sec/end_sec in /api/sfx_timeline.'
    ).toBeLessThan(100);
  }

  // sc02-sh02 bar must be at ~62.9% (35.0 / 55.648 * 100), not ~113.4%.
  // Tolerance ±5%.  sc01 is at ~9% so maxLeft identifies the sc02 bar.
  const maxLeft = Math.max(...barPositions.map(b => b.left));
  expect(maxLeft).toBeGreaterThan(55);   // correct ≈ 62.9%; bug gives 113.4%
  expect(maxLeft).toBeLessThan(70);      // sanity upper bound
});
