// TEST COVERAGE: KW-19
// Regression: Music and SFX bars do not appear in the preview timeline
// after clicking Generate Preview, even when the API returns items correctly.
//
// KW-19a (SFX tab): after Generate Preview, #sfx-tl-sfx must have ≥1 child bar.
//   Catches: sfxRenderTimeline() not called, or sfx_items not rendered into DOM.
//
// KW-19b (SFX tab): after Generate Preview with include_music=true,
//   #sfx-tl-music must have ≥1 child bar.
//   Catches: music_items present in API response but not rendered into DOM.
//
// KW-19c (Music tab): after Generate Preview, #music-tl-music must have ≥1 child bar.
//   Catches: _musicRenderTimeline() not called, or music_items not rendered.
//
// KW-19d (Music tab): after Generate Preview, #music-tl-vo must have ≥1 child bar.
//   Catches: vo_items present in API response but not rendered into DOM.
//
// KW-19e (SFX tab): after Generate Preview with a cut clip that has a non-zero
//   timing start, #sfx-tl-sfx bar must be positioned at that start time, not at 0.
//   Catches: /api/sfx_preview Pass 1 (cut clips) hardcodes start:0.0, ignoring
//   the timing dict sent by the frontend — bars always render at left≈0% instead
//   of their SfxPlan episode-absolute positions.
//
// KW-19a/b assert PRESENCE; KW-19c/d/e assert correct POSITION.

const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW12, resetKW13, resetKW19c, getEpDir } = require('../helpers/fixture_state');
const path = require('path');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });

// ── Helpers ───────────────────────────────────────────────────────────────────

async function openSfxTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="sfx"]');
  await page.waitForTimeout(400);
  await page.selectOption('#sfx-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.getElementById('sfx-status-bar').textContent !== 'Select an episode to begin.',
    { timeout: 12000 }
  );
}

async function openMusicTabAndSelectEp(page) {
  await page.goto('/');
  await page.click('button.tab[data-tab="music"]');
  await page.waitForTimeout(400);
  await page.selectOption('#music-ep-select', 'test-proj|s01e01');
  await page.waitForFunction(
    () => document.querySelector('#music-body .music-card') !== null,
    { timeout: 10000 }
  );
}

// ── KW-19a: SFX bars render in #sfx-tl-sfx after Generate Preview ─────────────
//
// Root cause of original failure: test sent selected:{} → sfx_items:[] → no bars.
// Fix: cut a real clip via API first, inject it into frontend state, then click
// Generate Preview. sfx_preview_pack will include the clip in sfx_items → bars render.

test('KW-19a: SFX bars render in #sfx-tl-sfx after Generate Preview', async ({ request, page }) => {
  resetKW12();   // provides VOPlan + sfx_source_fixture.wav

  const epDir    = getEpDir();
  const srcFile  = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const ITEM_ID  = 'sfx-sc01-sh01-001';
  const CLIP_ID  = 'Button_Click_Sharp_0.5s-2.0s';

  // ── Step 1: cut a real clip so there is a local source file ─────────────────
  const cutResp = await request.post('http://localhost:19999/api/sfx_cut_clip', {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      item_id: ITEM_ID, candidate_idx: 0,
      source_file: srcFile,
      title: 'Button Click Sharp', start_sec: 0.5, end_sec: 2.0,
    },
  });
  expect(cutResp.status()).toBe(200);
  const cutBody = await cutResp.json();
  expect(cutBody.clip_id).toBe(CLIP_ID);

  // ── Step 2: open SFX tab — _sfxLoadCutClips() will load the cut clip ────────
  await openSfxTabAndSelectEp(page);

  // ── Step 3: inject cut assignment into frontend state ───────────────────────
  // _sfxCutAssign and _sfxCutClips are let variables in page scope (not window).
  // page.evaluate runs in global context so direct assignment reaches them.
  await page.evaluate(([iid, cid, clip]) => {
    // eslint-disable-next-line no-undef
    _sfxCutAssign = { [iid]: cid };
    // eslint-disable-next-line no-undef
    _sfxCutClips  = [clip];
  }, [ITEM_ID, CLIP_ID, cutBody]);

  // ── Step 4: generate preview via UI ─────────────────────────────────────────
  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [resp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/sfx_preview'),
      { timeout: 30000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // ── Step 5: verify SFX bars rendered in #sfx-tl-sfx ────────────────────────
  // sfxLoadTimeline() reads timeline.json → sfxRenderTimeline() → _tlRender writes bars
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-tl-sfx');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  const sfxBarCount = await page.evaluate(
    () => document.getElementById('sfx-tl-sfx')?.children.length ?? 0
  );

  // 0 means sfx_items were in the timeline but _tlRender did not render them
  expect(sfxBarCount).toBeGreaterThan(0);
});

// ── KW-19b: Music bars render in #sfx-tl-music after Generate Preview ─────────

test('KW-19b: Music bars render in #sfx-tl-music after Generate Preview with include_music', async ({ page }) => {
  resetKW13();

  await openSfxTabAndSelectEp(page);

  // Check include_music and force-enable preview button
  await page.evaluate(() => {
    const cb = document.getElementById('sfx-include-music');
    if (cb) cb.checked = true;
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [resp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/sfx_preview'),
      { timeout: 30000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Wait for sfxRenderTimeline() to render music bars into the DOM
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-tl-music');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  const musicBarCount = await page.evaluate(
    () => document.getElementById('sfx-tl-music')?.children.length ?? 0
  );

  // 0 means music_items were returned by API but not rendered — bug present
  expect(musicBarCount).toBeGreaterThan(0);
});

// ── KW-19c: Music bars render in #music-tl-music after Generate Preview ───────
//
// HOW THE BUG IS TRIGGERED
//
// The fallback path in _musicRenderTimeline() (lines 11885–11894) runs only
// when sh.music_item_id is "" for a shot.  That only happens when the backend
// returns an empty music_item_id, which only happens when MusicPlan overrides
// lack shot_id (so the backend _music_index stays empty).
//
// resetKW19c() writes a MusicPlan whose overrides have NO shot_id field.
// The backend therefore sends back music_item_id="" for every shot.
// The frontend's _musicOverrides (loaded from the same plan) also has no
// shot_id — so the fallback find() returns undefined and bars would not draw.
//
// To reach the double-add bug at line 11889, we inject _musicOverrides WITH
// shot_id via page.evaluate() after the tab loads.  Now the fallback find()
// hits, and the broken code does: musicStart = shStart + ovr.start_sec
// where ovr.start_sec is episode-absolute (30s) and shStart is also
// episode-absolute (28.089s) → double-add → sc02 bar at 58s / 55.648s ≈ 104%
// → off-screen.  The position assertion below catches this.

test('KW-19c: Music bars render in #music-tl-music after Generate Preview', async ({ page }) => {
  // VOPlan (no music_items) + MusicPlan WITHOUT shot_id on overrides
  resetKW19c();

  await openMusicTabAndSelectEp(page);

  // Inject _musicOverrides WITH shot_id so the fallback find() in
  // _musicRenderTimeline() actually fires.  start_sec / end_sec are
  // episode-absolute (same values as the real MusicPlan fixture).
  // eslint-disable-next-line no-undef
  await page.evaluate(() => {
    _musicOverrides['music-sc02-sh02'] = {
      item_id: 'music-sc02-sh02', shot_id: 'sc02-sh02',
      start_sec: 30, end_sec: 35,
    };
    _musicOverrides['music-sc01-sh01'] = {
      item_id: 'music-sc01-sh01', shot_id: 'sc01-sh01',
      start_sec: 5, end_sec: 20,
    };
  });

  // Force-enable the Generate Preview button
  await page.evaluate(() => {
    document.getElementById('music-btn-review').disabled = false;
  });

  // Stamp #music-tl-music so we detect when _musicRenderBody() rebuilds the DOM
  await page.evaluate(() => {
    const el = document.getElementById('music-tl-music');
    if (el) el.setAttribute('data-stale', '1');
  });

  const [resp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/music_review_pack'),
      { timeout: 30000 }
    ),
    page.click('#music-btn-review'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Wait for _musicRenderBody() to finish rebuilding (stale marker gone)
  await page.waitForFunction(
    () => !document.querySelector('#music-tl-music[data-stale="1"]'),
    { timeout: 8000 }
  );

  // sc01 bar is always in-bounds (shStart=0, so double-add doesn't hurt it).
  // Its presence proves _musicRenderTimeline() ran at all.
  await page.waitForFunction(
    () => {
      const el = document.getElementById('music-tl-music');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  // ── Position check — this is the killer assertion ─────────────────────────
  // sc02-sh02: episode-absolute start_sec=30, total_dur=55.648s
  //
  // CORRECT (fix applied):  musicStart = ovr.start_sec          = 30.000s → left ≈ 53.9%
  // BUG    (fallback path):  musicStart = shStart + ovr.start_sec = 28.089+30 = 58.089s → left ≈ 104%
  //
  // sc01 bar is not affected (shStart=0 → double-add is a no-op for sc01).
  const barPositions = await page.evaluate(() => {
    const musDiv = document.getElementById('music-tl-music');
    if (!musDiv) return [];
    return Array.from(musDiv.children).map(el => {
      const left  = parseFloat((el.style.left  || '0').replace('%', ''));
      const width = parseFloat((el.style.width || '0').replace('%', ''));
      return { left, width };
    });
  });

  // Every bar must sit inside the timeline (left < 100%).
  // FAILS with bug: sc02 bar lands at ≈ 104%.
  for (const bar of barPositions) {
    expect(bar.left).toBeLessThan(100);
  }

  // sc02 bar (largest left%) must be at ≈ 53.9%.  Tolerance ±5%.
  // FAILS with bug: maxLeft ≈ 104%.
  const maxLeft = Math.max(...barPositions.map(b => b.left));
  expect(maxLeft).toBeGreaterThan(48);
  expect(maxLeft).toBeLessThan(60);
});

// ── KW-19d: VO bars render in #music-tl-vo after Generate Preview ─────────────

test('KW-19d: VO bars render in #music-tl-vo after Generate Preview', async ({ page }) => {
  resetKW13();

  await openMusicTabAndSelectEp(page);

  await page.evaluate(() => {
    document.getElementById('music-btn-review').disabled = false;
  });

  // Stamp both timeline rows so we can detect when _musicRenderBody() has
  // rebuilt the DOM after Generate Preview (same timing fix as KW-19c).
  await page.evaluate(() => {
    ['music-tl-music', 'music-tl-vo'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.setAttribute('data-stale', '1');
    });
  });

  const [resp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/music_review_pack'),
      { timeout: 30000 }
    ),
    page.click('#music-btn-review'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Wait for _musicRenderBody() to rebuild the DOM
  await page.waitForFunction(
    () => !document.querySelector('#music-tl-vo[data-stale="1"]'),
    { timeout: 8000 }
  );

  // Wait for _musicRenderTimeline() to render VO bars into the DOM
  // (checks the NEW #music-tl-vo, post-Generate-Preview).
  await page.waitForFunction(
    () => {
      const el = document.getElementById('music-tl-vo');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  const voBarCount = await page.evaluate(
    () => document.getElementById('music-tl-vo')?.children.length ?? 0
  );

  // 0 means vo_items were not rendered — bug present
  expect(voBarCount).toBeGreaterThan(0);
});

// ── KW-19e: SFX bar honours timing start_sec from SfxPlan after Generate Preview ──
//
// HOW THE BUG IS TRIGGERED
//
// sfxGeneratePreview() sends timing:{item_id:{start,end}} in the POST body.
// The /api/sfx_preview handler builds sfx_selections in two passes:
//   Pass 1 (cut clips)     — hardcodes start:0.0, IGNORES timing  ← BUG
//   Pass 2 (library cands) — correctly reads timing.get(item_id).start
//
// This test uses a cut clip (Pass 1 path) and injects _sfxTiming with a
// non-zero episode-absolute start.  With the bug the bar sits at left≈0%;
// with the fix it sits at left≈TIMING_START/total_dur*100.

test('KW-19e: SFX bar position respects timing start_sec for cut clips', async ({ request, page }) => {
  resetKW12();   // VOPlan + sfx_source_fixture.wav

  const epDir   = getEpDir();
  const srcFile = path.join(epDir, 'assets', 'sfx', 'sfx-sc01-sh01-001', 'sfx_source_fixture.wav');
  const ITEM_ID = 'sfx-sc01-sh01-001';
  const CLIP_ID = 'Button_Click_Sharp_0.5s-2.0s';

  // Cut a real clip so the WAV exists on disk for sfx_preview_pack to read
  const cutResp = await request.post('http://localhost:19999/api/sfx_cut_clip', {
    data: {
      slug: 'test-proj', ep_id: 's01e01',
      item_id: ITEM_ID, candidate_idx: 0,
      source_file: srcFile,
      title: 'Button Click Sharp', start_sec: 0.5, end_sec: 2.0,
    },
  });
  expect(cutResp.status()).toBe(200);
  const cutBody = await cutResp.json();
  expect(cutBody.clip_id).toBe(CLIP_ID);

  await openSfxTabAndSelectEp(page);

  // Inject cut assignment AND timing.
  // TIMING_START is episode-absolute: the SFX clip should start at 10s.
  // Total duration = 55.648s → expected bar left ≈ 10/55.648×100 ≈ 18.0%
  //
  // With the bug: Pass 1 hardcodes start=0.0 → bar at left = 0%
  // With the fix: Pass 1 reads timing[item_id].start = 10.0 → bar at left ≈ 18%
  const TIMING_START = 10.0;
  const TIMING_END   = 13.0;

  await page.evaluate(([iid, cid, clip, tStart, tEnd]) => {
    // eslint-disable-next-line no-undef
    _sfxCutAssign = { [iid]: cid };
    // eslint-disable-next-line no-undef
    _sfxCutClips  = [clip];
    // eslint-disable-next-line no-undef
    _sfxTiming    = { [iid]: { start: tStart, end: tEnd } };
  }, [ITEM_ID, CLIP_ID, cutBody, TIMING_START, TIMING_END]);

  await page.evaluate(() => {
    document.getElementById('sfx-btn-preview').disabled = false;
  });

  const [resp] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/sfx_preview'),
      { timeout: 30000 }
    ),
    page.click('#sfx-btn-preview'),
  ]);
  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // Wait for sfxLoadTimeline() → sfxRenderTimeline() → bars in DOM
  await page.waitForFunction(
    () => {
      const el = document.getElementById('sfx-tl-sfx');
      return el && el.children.length > 0;
    },
    { timeout: 8000 }
  );

  // ── Position + width check ───────────────────────────────────────────────
  // Total duration = 55.648s.
  //
  // left  (start): TIMING_START=10s → 10/55.648×100 ≈ 18.0%
  //                BUG (timing ignored) → 0.0%
  //
  // width (window): TIMING_END-TIMING_START = 3s → 3/55.648×100 ≈ 5.4%
  //                 BUG (uses actual clip length 1.5s) → 1.5/55.648×100 ≈ 2.7%
  //                 BUG (save loop: timing not persisted; after reload end=null
  //                      → end_sec = start + clip_len = 10+1.5 = 11.5s
  //                      → width = 1.5/55.648×100 ≈ 2.7%)
  //                 BUG (0.5s shorter: end = 12.5 → width = 2.5/55.648 ≈ 4.49%)
  const bar = await page.evaluate(() => {
    const sfxDiv = document.getElementById('sfx-tl-sfx');
    if (!sfxDiv || !sfxDiv.children.length) return null;
    const el = sfxDiv.children[0];
    return {
      left:  parseFloat((el.style.left  || '0').replace('%', '')),
      width: parseFloat((el.style.width || '0').replace('%', '')),
    };
  });

  expect(bar).not.toBeNull();

  // Start position — FAILS with timing bug (left = 0)
  expect(bar.left).toBeGreaterThan(5);   // bug gives 0%  — fix gives ≈18%
  expect(bar.left).toBeLessThan(30);     // sanity

  // Width spans the full user-intended window (3s), not the 1.5s clip.
  // FAILS with width bug: width ≈ 2.7% (clip length, physical cut).  Fix gives ≈ 5.4%.
  // Threshold 4.8 catches: 2.7% (clip-length bug), 4.49% (0.5s-shorter bug),
  //   and the save-loop bug (after reload with start_sec=0,end=null → width≈2.7%).
  expect(bar.width).toBeGreaterThan(4.8);  // 5.4% (correct) — all known bugs fail this
  expect(bar.width).toBeLessThan(10);      // sanity
});
