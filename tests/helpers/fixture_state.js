const fs   = require('fs');
const path = require('path');

const FIXTURE_EP    = path.join(__dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01');

// Mutable: updated by server.js via setPipeTestDir() before each test run.
// Falls back to the source fixture tree when running outside the server harness.
let _pipeTestDir = null;

function setPipeTestDir(dir) {
  _pipeTestDir = dir;
}

function getEpDir() {
  const fixtureRoot = _pipeTestDir
    ? path.join(_pipeTestDir, 'projects')
    : path.join(__dirname, '..', 'fixtures', 'projects');
  return path.join(fixtureRoot, 'test-proj', 'episodes', 's01e01');
}

function voplan() {
  return JSON.parse(fs.readFileSync(path.join(getEpDir(), 'VOPlan.en.json'), 'utf8'));
}

function musicplan() {
  return JSON.parse(fs.readFileSync(path.join(getEpDir(), 'MusicPlan.json'), 'utf8'));
}

// KW-1b start: no VOPlan, no MusicPlan (tests that music_review_pack returns 400)
function resetKW1() {
  const ep = getEpDir();
  const vp = path.join(ep, 'VOPlan.en.json');
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(vp)) fs.unlinkSync(vp);
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-1a start: fixture VOPlan present (manifest_merge needs it as --locale input),
// no MusicPlan.  manifest_merge accepts locale_scope=merged as input and rewrites it.
function resetKW1a() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-2/KW-9 start: merged VOPlan present, fresh MusicPlan deleted
function resetKW2() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-13 start: merged VOPlan + committed MusicPlan both present
function resetKW13() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'MusicPlan.json'),
    path.join(ep, 'MusicPlan.json')
  );
  // Copy music WAVs so sfx_preview_pack (and KW-11d/e) can find them.
  // sfx_preview_pack only adds music_items to tl_doc when the WAV exists.
  const musicDir = path.join(ep, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(wav => {
    const src = path.join(FIXTURE_EP, 'assets', 'music', wav);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(musicDir, wav));
  });
  // Remove any stale MediaPreviewPack output from a previous run
  const packDir = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

// KW-12 start: merged VOPlan present, SFX source fixture WAV in place, no prior cut output.
function resetKW12() {
  const ep = getEpDir();
  // VOPlan needed for SFX tab to load shot list
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  // Place source WAV that sfx_cut_clip will slice
  const sfxDir = path.join(ep, 'assets', 'sfx', 'sfx-sc01-sh01-001');
  fs.mkdirSync(sfxDir, { recursive: true });
  fs.copyFileSync(
    path.join(__dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01', 'assets', 'sfx', 'sfx_source_fixture.wav'),
    path.join(sfxDir, 'sfx_source_fixture.wav')
  );
  // Remove any previously cut clips so each test starts clean
  const cutJson = path.join(ep, 'assets', 'sfx', 'sfx_cut_clips.json');
  if (fs.existsSync(cutJson)) fs.unlinkSync(cutJson);
  // Remove any prior clip WAVs in the item dir (keep only the source fixture)
  fs.readdirSync(sfxDir).forEach(f => {
    if (f !== 'sfx_source_fixture.wav') fs.unlinkSync(path.join(sfxDir, f));
  });
}

// KW-15 / KW-16 / KW-17: render_video.py invoked directly — no server state to reset.
// Each spec manages its own tmpdir; these are no-ops kept for suite consistency.
function resetKW15() {}
function resetKW16() {}
function resetKW17() {}

// KW-18 Music: VOPlan present but with NO music_items + committed MusicPlan 1.1.
// The real regression: /api/music_timeline returns 200 with shots where
// music_item_id="" (no music_items in VOPlan). The broken code checks
// (_musicTimeline && _musicTimeline.shots) which is truthy → takes timeline path
// → filters to 0 shots → 0 blocks. The fix checks _tlShots.length > 0 so
// when timeline has no music shots, it falls through to the MusicPlan fallback.
function resetKW18Music() {
  const ep = getEpDir();
  // VOPlan with NO music_items → timeline returns 200 but all music_item_id=""
  // This is the exact condition that hid the fallback path in the broken code.
  fs.copyFileSync(
    path.join(__dirname, '..', 'fixtures', 'projects', 'test-proj', 'episodes', 's01e01', 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'MusicPlan.json'),
    path.join(ep, 'MusicPlan.json')
  );
}

// KW-19c: VOPlan with NO music_items + MusicPlan whose overrides have NO shot_id.
//
// Why no shot_id?  The backend seeds _music_index from MusicPlan.shot_overrides
// only when each entry carries BOTH item_id AND shot_id.  Without shot_id the
// index stays empty → build_timeline() sets music_item_id="" on every shot →
// the frontend takes the fallback path in _musicRenderTimeline().
//
// The test then injects _musicOverrides WITH shot_id via page.evaluate() so
// the fallback find(o => o.shot_id === sh.shot_id) actually hits — triggering
// the double-add bug at lines 11889–11891.  This is the only way to reach that
// path: a fixture alone cannot split what the backend sees from what the
// frontend holds.
function resetKW19c() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  // MusicPlan without shot_id — backend _music_index stays empty.
  // start_sec / end_sec are episode-absolute (same values as the real fixture)
  // so the injected _musicOverrides the test adds will carry the right coords.
  const plan = {
    schema_id: 'MusicPlan',
    schema_version: '1.1',
    loop_selections: {},
    shot_overrides: [
      {
        item_id: 'music-sc02-sh02',
        // NO shot_id — backend cannot seed _music_index from this entry
        music_clip_id: 'cher1:126.0s-155.6s',
        music_asset_id: 'music-sc02-sh02',
        clip_start_sec: 126,
        clip_duration_sec: 29.599999999999994,
        start_sec: 30,
        end_sec: 35,
        duck_db: 0,
        fade_sec: 0.15,
      },
      {
        item_id: 'music-sc01-sh01',
        // NO shot_id
        music_clip_id: 'cher2:114.0s-142.1s',
        music_asset_id: 'music-sc01-sh01',
        clip_start_sec: 114,
        clip_duration_sec: 28.099999999999994,
        start_sec: 5,
        end_sec: 20,
        duck_db: 0,
        fade_sec: 0.15,
      },
    ],
  };
  fs.writeFileSync(path.join(ep, 'MusicPlan.json'), JSON.stringify(plan, null, 2));
  // Music WAVs (render is skipped in TEST_MODE, but presence expected by server)
  const musicDir = path.join(ep, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(wav => {
    const src = path.join(FIXTURE_EP, 'assets', 'music', wav);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(musicDir, wav));
  });
  // Remove any stale MediaPreviewPack from a previous run
  const packDir = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

// KW-18 SFX: VOPlan with sfx_items (for shot timeline) + no MusicPlan/SfxPlan.
// Forces SFX tab to build shot timeline purely from ShotList + VOPlan sfx_items.
// If _sfxShotTimeline is not built or sfx_items not mapped, blockCount = 0 → fails.
function resetKW18Sfx() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
  const sfxPlan = path.join(ep, 'SfxPlan.json');
  if (fs.existsSync(sfxPlan)) fs.unlinkSync(sfxPlan);
}

module.exports = {
  getEpDir,
  setPipeTestDir,
  voplan,
  musicplan,
  resetKW1,
  resetKW1a,
  resetKW2,
  resetKW12,
  resetKW13,
  resetKW15,
  resetKW16,
  resetKW17,
  resetKW18Music,
  resetKW18Sfx,
  resetKW19c,
  // EP_DIR kept for backward compat — resolves dynamically via getEpDir()
  get EP_DIR() { return getEpDir(); },
};
