// TEST COVERAGE: KW-22
// Source: prompts/regression.txt § "KW-22: Stage 9 — translated locale visible before VOPlan exists"
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW22, getEpDir } = require('../helpers/fixture_state');
const fs = require('fs'), path = require('path');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(()      => { resetKW22(); });

test('KW-22: zh-Hans appears in pipeline_status locales when only AssetManifest exists', async ({ request }) => {
  // Fixture state: VOPlan.en.json present, AssetManifest.zh-Hans.json present,
  // VOPlan.zh-Hans.json absent.  This is the real user state before Step 5 runs
  // for the translated locale.
  //
  // Regression: the old code used AssetManifest as a fallback ONLY when no
  // VOPlan existed at all.  Once VOPlan.en.json appeared, zh-Hans was silently
  // dropped from the locales list and steps 5-11 were never rendered in Stage 9.

  const resp = await request.get('/pipeline_status?slug=test-proj&ep_id=s01e01');
  expect(resp.status()).toBe(200);

  const status = await resp.json();
  expect(status.locales).toBeDefined();

  // Both locales must be present
  expect(status.locales).toContain('en');
  expect(status.locales).toContain('zh-Hans');

  // VOPlan.zh-Hans.json must NOT have been created by this call (read-only check)
  const zhVoPlan = path.join(getEpDir(), 'VOPlan.zh-Hans.json');
  expect(fs.existsSync(zhVoPlan)).toBe(false);
});
