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

// KW-2/KW-9 start: merged VOPlan present, fresh MusicPlan deleted.
// The last VO end_sec is patched to 80.0 AND pause_after_ms to 10000 (10s).
// This creates a two-level sentinel:
//   - end_sec=80.0 exceeds ShotList total (55.648s) — catches tabs reading ShotList
//   - pause_after_ms=10000 means full VO tail = 80.0+10.0 = 90.0s
// Strong assertion threshold: >= 90.0.
// Paths that include pause_after_ms in total (e.g. _loadAndMergeTl) return 90.0 → PASS.
// Paths that ignore pause_after_ms (e.g. music_review_pack.build_timeline,
// _musicTimeline JS) return only max(55.648, 80.0)=80.0 → FAIL at >= 90.0 → bug caught.
function resetKW2() {
  const ep = getEpDir();
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  vp.vo_items[vp.vo_items.length - 1].end_sec = 80.0;
  vp.vo_items[vp.vo_items.length - 1].pause_after_ms = 10000;
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// resetKW80: sentinel state used by KW-27 through KW-24.
// VOPlan present with last vo_item end_sec patched to 80.0 (well above ShotList
// total of 55.648s) AND pause_after_ms patched to 10000 (10s).
// MusicPlan + SfxPlan + music WAVs present so all three timeline endpoints
// (/api/vo_timeline, /api/music_timeline, /api/sfx_timeline) can respond with 200.
// A stub SfxPreviewPack/preview_audio.wav is planted so KW-29 (SFX restore path)
// can fire _loadAndMergeTl() without generating audio.
//
// Two-level sentinel:
//   - end_sec=80.0 > ShotList total (55.648s): catches tabs reading ShotList alone
//   - pause_after_ms=10000: full VO tail = 80.0+10.0 = 90.0s
// Strong assertion threshold: >= 90.0.
//
// Paths that include pause_after_ms (_loadAndMergeTl):
//   _voLastEnd = max(end_sec + pause_after_ms/1000) = 90.0 → PASS at >= 90.0
// Paths that ignore pause_after_ms (music_review_pack.build_timeline, _musicTimeline JS):
//   total_duration_sec = max(55.648, 80.0) = 80.0 → FAIL at >= 90.0 → bug caught
function resetKW80() {
  const ep = getEpDir();
  // Patch last VO end_sec to 80.0 and pause_after_ms to 10000 (10s tail)
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  vp.vo_items[vp.vo_items.length - 1].end_sec = 80.0;
  vp.vo_items[vp.vo_items.length - 1].pause_after_ms = 10000;
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
  // Mirror what patch_shotlist_durations.py does in the live pipeline:
  // extend the last shot's duration_sec so ShotList agrees with VOPlan end_sec=80.0.
  // Without this the test creates a ShotList/VOPlan divergence that never occurs in
  // production (live ShotList is always patched before the UI is opened).
  const sl = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'ShotList.json'), 'utf8'));
  const lastShot = sl.shots[sl.shots.length - 1];
  lastShot.duration_sec  = parseFloat((80.0 - (lastShot.start_sec || 0)).toFixed(3));
  sl.total_duration_sec  = 80.0;
  fs.writeFileSync(path.join(ep, 'ShotList.json'), JSON.stringify(sl, null, 2));
  // MusicPlan + music WAVs (needed by music_timeline and SFX include_music path)
  fs.copyFileSync(path.join(FIXTURE_EP, 'MusicPlan.json'), path.join(ep, 'MusicPlan.json'));
  const musicDir = path.join(ep, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(wav => {
    const src = path.join(FIXTURE_EP, 'assets', 'music', wav);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(musicDir, wav));
  });
  // SfxPlan (needed by sfx_timeline)
  fs.copyFileSync(path.join(FIXTURE_EP, 'SfxPlan.json'), path.join(ep, 'SfxPlan.json'));
  // Stub SfxPreviewPack/preview_audio.wav so _sfxTryRestorePreview() HEAD check passes
  const packDir = path.join(ep, 'assets', 'sfx', 'SfxPreviewPack');
  fs.mkdirSync(packDir, { recursive: true });
  const wav = Buffer.alloc(44);
  wav.write('RIFF', 0, 'ascii');   wav.writeUInt32LE(36, 4);
  wav.write('WAVE', 8, 'ascii');   wav.write('fmt ', 12, 'ascii');
  wav.writeUInt32LE(16, 16);       wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);        wav.writeUInt32LE(44100, 24);
  wav.writeUInt32LE(88200, 28);    wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);       wav.write('data', 36, 'ascii');
  wav.writeUInt32LE(0, 40);
  fs.writeFileSync(path.join(packDir, 'preview_audio.wav'), wav);
  // Remove any stale MediaPreviewPack
  const mediaPack = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(mediaPack)) fs.rmSync(mediaPack, { recursive: true, force: true });
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
  // Restore the original ShotList fixture so that any previous resetKW80() patch
  // (which extends the last shot duration to make total=80.0s) does not affect
  // tests that depend on the unpatched ShotList (total=55.648s).
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'ShotList.json'),
    path.join(ep, 'ShotList.json')
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

// KW-2g: VOPlan with music_items present, NO MusicPlan.
// Sentinel: two shots (sc01-sh01, sc02-sh02) carry item_id in VOPlan.music_items.
// The API must fall back to VOPlan.music_items when MusicPlan.json is absent.
// FAILS today: API builds music_index only from MusicPlan.shot_overrides → empty →
//   all shots get music_item_id="" → Shot Overrides section shows nothing.
// PASSES with fix: API reads VOPlan.music_items as fallback → shots get music_item_id.
function resetKW2g() {
  const ep = getEpDir();
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  vp.music_items = [
    { shot_id: 'sc01-sh01', item_id: 'music-sc01-sh01' },
    { shot_id: 'sc02-sh02', item_id: 'music-sc02-sh02' },
  ];
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
  // Restore canonical ShotList so timing is correct
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'ShotList.json'),
    path.join(ep, 'ShotList.json')
  );
  // No MusicPlan — this is the exact state that triggers the bug
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
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

// KW-22: VOPlan.en.json exists but VOPlan.zh-Hans.json does NOT yet exist.
// AssetManifest.zh-Hans.json is present (written by Stage 7/8).
// This is the exact state a user is in before running Step 5 for zh-Hans.
// The regression: once VOPlan.en.json existed, the fallback locale scan was
// skipped, so zh-Hans never appeared in /pipeline_status locales list.
function resetKW22() {
  const ep = getEpDir();
  // VOPlan.en.json present (primary locale already merged)
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  // AssetManifest.zh-Hans.json present (Stage 7/8 output for translated locale)
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'AssetManifest.zh-Hans.json'),
    path.join(ep, 'AssetManifest.zh-Hans.json')
  );
  // VOPlan.zh-Hans.json must NOT exist — manifest_merge hasn't run yet
  const zhVoPlan = path.join(ep, 'VOPlan.zh-Hans.json');
  if (fs.existsSync(zhVoPlan)) fs.unlinkSync(zhVoPlan);
}

// KW-3: VOPlan with scene_heads={"sc01":0} so any DOM-supplied head value stands out.
// VO WAVs are already present via the full fixture copy done by startTestServer.
function resetKW3() {
  const ep = getEpDir();
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  vp.scene_heads = { sc01: 0 };
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
}

// KW-24: VOPlan + MusicPlan at episode root (correct location) + music WAVs.
function resetKW24() {
  const ep = getEpDir();
  fs.copyFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(ep, 'VOPlan.en.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'MusicPlan.json'), path.join(ep, 'MusicPlan.json'));
  const musicDir = path.join(ep, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(wav => {
    const src = path.join(FIXTURE_EP, 'assets', 'music', wav);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(musicDir, wav));
  });
  const packDir = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

// KW-26: VOPlan + MusicPlan + SfxPlan + a stub SfxPreviewPack/preview_audio.wav.
// The stub WAV simulates a previously generated SFX preview so that
// _sfxTryRestorePreview() finds the file and should show the preview wrap.
// SfxPlan.json is required for /api/sfx_timeline to return sfx_items with
// episode-absolute positions (used by the bar-position assertion in KW-26).
function resetKW26() {
  const ep = getEpDir();
  fs.copyFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(ep, 'VOPlan.en.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'MusicPlan.json'), path.join(ep, 'MusicPlan.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'SfxPlan.json'),   path.join(ep, 'SfxPlan.json'));
  // Restore the original ShotList fixture so that any previous resetKW80() patch
  // (which extends the last shot duration to make total=80.0s) does not affect
  // bar-position assertions that rely on the unpatched ShotList (total=55.648s).
  fs.copyFileSync(path.join(FIXTURE_EP, 'ShotList.json'),  path.join(ep, 'ShotList.json'));
  // Music WAVs (needed by _loadAndMergeTl → music_timeline)
  const musicDir = path.join(ep, 'assets', 'music');
  fs.mkdirSync(musicDir, { recursive: true });
  ['music-sc01-sh01.wav', 'music-sc02-sh02.wav'].forEach(wav => {
    const src = path.join(FIXTURE_EP, 'assets', 'music', wav);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(musicDir, wav));
  });
  // Plant a minimal valid WAV so /serve_media HEAD returns 200
  const packDir = path.join(ep, 'assets', 'sfx', 'SfxPreviewPack');
  fs.mkdirSync(packDir, { recursive: true });
  const wav = Buffer.alloc(44);
  wav.write('RIFF', 0, 'ascii');   wav.writeUInt32LE(36, 4);
  wav.write('WAVE', 8, 'ascii');   wav.write('fmt ', 12, 'ascii');
  wav.writeUInt32LE(16, 16);       wav.writeUInt16LE(1, 20);   // PCM
  wav.writeUInt16LE(1, 22);        wav.writeUInt32LE(44100, 24);
  wav.writeUInt32LE(88200, 28);    wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);       wav.write('data', 36, 'ascii');
  wav.writeUInt32LE(0, 40);
  fs.writeFileSync(path.join(packDir, 'preview_audio.wav'), wav);
}

module.exports = {
  getEpDir,
  setPipeTestDir,
  voplan,
  resetKW3,
  musicplan,
  resetKW1,
  resetKW1a,
  resetKW2,
  resetKW2g,
  resetKW12,
  resetKW13,
  resetKW15,
  resetKW16,
  resetKW17,
  resetKW18Music,
  resetKW18Sfx,
  resetKW19c,
  resetKW22,
  resetKW24,
  resetKW26,
  resetKW80,
  // EP_DIR kept for backward compat — resolves dynamically via getEpDir()
  get EP_DIR() { return getEpDir(); },
};
