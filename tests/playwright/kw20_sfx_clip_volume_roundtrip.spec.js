// TEST COVERAGE: KW-20
// Regression: SfxPlan.json clip volume is saved in cut_clips[].volume_db (v1.2)
// and NOT in sfx_entries[].clip_volume_db (old schema) or a root clip_volumes key.
//
// Changed behaviour being guarded:
//   - SfxPlan schema 1.2: clip volume moves from sfx_entries[].clip_volume_db
//     to cut_clips[].volume_db.
//   - /api/sfx_save_all embeds clip_volumes[clip_id] into each cut_clip entry
//     as volume_db, not into sfx_entries.
//   - No root-level clip_volumes key is written to SfxPlan.json.
//
// Test flow (API-only, no browser needed — the data path is server-side):
//   1. Write a minimal sfx_search_results.json (required by sfx_save_all handler).
//   2. POST /api/sfx_save_all with:
//        cut_clips: [{ clip_id, item_id, ... }]
//        cut_assign: { item_id: clip_id }
//        clip_volumes: { clip_id: 6 }
//        selected: {}   (no library candidates — cut-clip-only item)
//   3. Read SfxPlan.json from disk.
//   4. Assert cut_clips[0].volume_db === 6.
//   5. Assert no sfx_entries entry has a clip_volume_db field.
//   6. Assert no root-level clip_volumes key exists in SfxPlan.json.
const { test, expect } = require('@playwright/test');
const fs   = require('fs');
const path = require('path');
const { startTestServer, stopTestServer } = require('../helpers/server');
const { resetKW12, getEpDir } = require('../helpers/fixture_state');

const BASE_URL = 'http://localhost:19999';

const ITEM_ID = 'sfx-sc01-sh01-001';
const CLIP_ID = 'Button_Click_Sharp_0.5s-2.0s';
const CLIP_VOL_DB = 6;

let serverProc;
test.beforeAll(async () => { serverProc = await startTestServer(); });
test.afterAll(async ()  => { await stopTestServer(serverProc); });
test.beforeEach(() => { resetKW12(); });

// Helper: write a minimal sfx_search_results.json so sfx_save_all does not
// abort with "sfx_search_results.json not found".  The selected dict sent in
// the POST is empty (no library candidates), so the results content does not
// need to match any real candidate — it just needs to be valid JSON.
function writeFakeSearchResults(epDir) {
  const sfxDir = path.join(epDir, 'assets', 'sfx');
  fs.mkdirSync(sfxDir, { recursive: true });
  fs.writeFileSync(
    path.join(sfxDir, 'sfx_search_results.json'),
    JSON.stringify({ saved_at: '2026-01-01T00:00:00Z', results: {}, selected: {} }),
  );
}

// Helper: create the WAV file that will be referenced as the cut clip path so
// the second sfx_save_all loop (cut-clip-only items) can build sfx_plan_entries.
function createFakeClipWav(epDir) {
  const clipDir = path.join(epDir, 'assets', 'sfx', ITEM_ID);
  fs.mkdirSync(clipDir, { recursive: true });
  const wavPath = path.join(clipDir, CLIP_ID + '.wav');
  // Minimal 44-byte WAV header (enough for fs.existsSync; not playable)
  const buf = Buffer.alloc(44);
  buf.write('RIFF', 0); buf.writeUInt32LE(36, 4);
  buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(44100, 24);
  buf.writeUInt32LE(88200, 28); buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34); buf.write('data', 36);
  buf.writeUInt32LE(0, 40);
  fs.writeFileSync(wavPath, buf);
  return `assets/sfx/${ITEM_ID}/${CLIP_ID}.wav`;  // ep_dir-relative path
}

test('KW-20: /api/sfx_save_all writes volume_db in cut_clips, not clip_volume_db in sfx_entries', async ({ request }) => {
  const epDir   = getEpDir();
  writeFakeSearchResults(epDir);
  const clipRelPath = createFakeClipWav(epDir);

  // ── POST to sfx_save_all ─────────────────────────────────────────────────
  const resp = await request.post(`${BASE_URL}/api/sfx_save_all`, {
    data: {
      slug:     'test-proj',
      ep_id:    's01e01',
      selected:     {},           // no library candidates — cut clip only
      timing:       {},
      volumes:      {},
      duck_fade:    {},
      cut_clips: [
        {
          clip_id:      CLIP_ID,
          item_id:      ITEM_ID,
          candidate_idx: 0,
          start_sec:    0.5,
          end_sec:      2.0,
          duration_sec: 1.5,
          source_file:  path.join(epDir, 'assets', 'sfx', ITEM_ID, 'sfx_source_fixture.wav'),
          path:         clipRelPath,
          volume_db:    0,        // stale field on incoming object; server must overwrite
        },
      ],
      cut_assign:   { [ITEM_ID]: CLIP_ID },
      clip_volumes: { [CLIP_ID]: CLIP_VOL_DB },  // ← volume the user set
    },
  });

  expect(resp.status()).toBe(200);
  const body = await resp.json();
  expect(body.ok).toBe(true);

  // ── Read the written SfxPlan.json ────────────────────────────────────────
  const planPath = path.join(epDir, 'SfxPlan.json');
  expect(fs.existsSync(planPath)).toBe(true);
  const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));

  // ── Assertion 1: volume lives in cut_clips[].volume_db ──────────────────
  expect(Array.isArray(plan.cut_clips)).toBe(true);
  expect(plan.cut_clips.length).toBeGreaterThan(0);
  const savedClip = plan.cut_clips.find(c => c.clip_id === CLIP_ID);
  expect(savedClip).toBeDefined();
  expect(savedClip.volume_db).toBe(CLIP_VOL_DB);

  // ── Assertion 2: sfx_entries must NOT have clip_volume_db ───────────────
  for (const entry of (plan.sfx_entries || [])) {
    expect(entry).not.toHaveProperty('clip_volume_db');
  }

  // ── Assertion 3: no root-level clip_volumes key ─────────────────────────
  expect(plan).not.toHaveProperty('clip_volumes');
});
