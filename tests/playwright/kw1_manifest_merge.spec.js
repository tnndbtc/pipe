// TEST COVERAGE: KW-1
// Source: prompts/regression.txt § "KW-1: Stage 9 Step 5 — Manifest Merge"
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW1, getEpDir } = require('../helpers/fixture_state');
const fs = require('fs'), path = require('path');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW1(); });

test('KW-1b: music_review_pack returns 400 before step 5 runs', async ({ request }) => {
  const resp = await request.post('/api/music_review_pack', {
    data: { slug: 'test-proj', ep_id: 's01e01' },
  });
  expect(resp.status()).toBe(400);
  const body = await resp.json();
  expect(body.error).toMatch(/locale_scope|merged|manifest|not ready|VOPlan|stage/i);
});

test('KW-1a: Run 5 button writes VOPlan with locale_scope=merged', async ({ page }) => {
  await page.goto('/');

  // Switch to Pipeline tab
  await page.click('button.tab[data-tab="pipeline"]');
  await page.waitForTimeout(300);

  // Select the episode from the pipeline dropdown (same pattern as music tab)
  await page.selectOption('#pipe-ep-select', 'test-proj|s01e01');

  // Wait for Stage 9 row to appear (pipeline status loaded and rendered)
  const stage9Row = page.locator('.pipe-step').filter({ hasText: /Stage 9/ });
  await stage9Row.waitFor({ state: 'visible', timeout: 10000 });

  // Expand Stage 9 detail panel — click the btn-expand in that row.
  // The sub-step rows (including manifest_merge) are inside a pipe-detail div that starts
  // hidden; clicking btn-expand toggles the "open" class to make it visible.
  const expandBtn = stage9Row.locator('button.btn-expand').first();
  await expandBtn.waitFor({ state: 'visible', timeout: 5000 });
  await expandBtn.click();

  // Find Run 5 button via data-step attribute (added by Phase 1 Bug 2).
  // The button is inside the pipe-detail panel which is now open.
  const runBtn = page.locator('[data-step="manifest_merge"] .btn-pipe-run').first();
  await runBtn.waitFor({ state: 'visible', timeout: 8000 });
  await runBtn.click();

  // Poll music_review_pack — returns 200 only when VOPlan has locale_scope=merged
  await page.waitForFunction(async () => {
    try {
      const r = await fetch('/api/music_review_pack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug: 'test-proj', ep_id: 's01e01' }),
      });
      return r.ok;
    } catch { return false; }
  }, { timeout: 20000, polling: 500 });

  // Verify file on disk
  const vp = JSON.parse(fs.readFileSync(path.join(getEpDir(), 'VOPlan.en.json'), 'utf8'));
  expect(vp.locale_scope).toBe('merged');
  expect(vp.music_items.length).toBeGreaterThan(0);
});
