// TEST COVERAGE: KW-1
// Source: prompts/regression.txt § "KW-1: Stage 9 Step 5 — Manifest Merge"
const { test, expect } = require('@playwright/test');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW1 } = require('../helpers/fixture_state');

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });

test('KW-1b: music_review_pack returns 400 before step 5 runs', async ({ request }) => {
  resetKW1();  // no VOPlan — music_review_pack must reject
  const resp = await request.post('/api/music_review_pack', {
    data: { slug: 'test-proj', ep_id: 's01e01' },
  });
  expect(resp.status()).toBe(400);
  const body = await resp.json();
  expect(body.error).toMatch(/locale_scope|merged|manifest|not ready|VOPlan|stage/i);
});

