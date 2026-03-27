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

// KW-2g: No MusicPlan present.
// Invariant: /api/music_timeline must return all shots with empty music_item_id.
// VOPlan and AssetManifest no longer carry music_items; MusicPlan is the sole source.
// When MusicPlan.json is absent the timeline shows no music bars — this is correct.
function resetKW2g() {
  const ep = getEpDir();
  fs.copyFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), path.join(ep, 'VOPlan.en.json'));
  fs.copyFileSync(path.join(FIXTURE_EP, 'ShotList.json'),  path.join(ep, 'ShotList.json'));
  // No MusicPlan — timeline must show no music bars
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

// resetKW29 — MusicPlan with two shot_overrides (free-segment UI fixture)
function resetKW29() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const plan = {
    schema_id: 'MusicPlan',
    schema_version: '1.1',
    loop_selections: {},
    track_volumes: {},
    clip_volumes: {},
    shot_overrides: [
      {
        start_sec: 5.0,
        end_sec: 20.0,
        music_asset_id: 'cher2',
        music_clip_id: 'cher2:114.0s-142.1s',
        clip_start_sec: 114.0,
        clip_duration_sec: 28.1,
        duck_db: -12.0,
        fade_sec: 0.15,
      },
      {
        start_sec: 30.0,
        end_sec: 55.0,
        music_asset_id: 'cher1',
        music_clip_id: 'cher1:126.0s-155.6s',
        clip_start_sec: 126.0,
        clip_duration_sec: 29.6,
        duck_db: 0.0,
        fade_sec: 0.15,
      },
    ],
  };
  fs.writeFileSync(path.join(ep, 'MusicPlan.json'), JSON.stringify(plan, null, 2));
}

// resetKW30 — SfxPlan with two shot_overrides (free-segment UI fixture)
function resetKW30() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const plan = {
    schema_id: 'SfxPlan',
    schema_version: '1.0',
    timing_format: 'episode_absolute',
    shot_overrides: [
      {
        start_sec: 5.0,
        end_sec: 10.0,
        source_file: 'assets/sfx/sfx-sc01-sh01-001/ai_1773848083082.mp3',
        volume_db: 0.0,
        duck_db: 0.0,
        fade_sec: 0.0,
        clip_id: null,
        clip_path: null,
      },
      {
        start_sec: 35.0,
        end_sec: 40.0,
        source_file: 'assets/sfx/sfx-sc02-sh02-001/sfx-sc02-sh02-001.mp3',
        volume_db: 0.0,
        duck_db: 0.0,
        fade_sec: 0.0,
        clip_id: null,
        clip_path: null,
      },
    ],
    cut_clips: [],
    cut_assign: {},
  };
  fs.writeFileSync(path.join(ep, 'SfxPlan.json'), JSON.stringify(plan, null, 2));
}

// resetKW31 — MediaPlan with two shot_overrides (free-segment UI fixture)
function resetKW31() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(FIXTURE_EP, 'VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const plan = {
    schema_id: 'MediaPlan',
    schema_version: '1.0',
    shot_overrides: [
      {
        type: 'image',
        url: 'https://example.com/bg-reactor.jpg',
        path: 'assets/media/bg-reactor-control-room-night.jpg',
        clip_id: 'bg-reactor-control-room-night',
        hold_sec: 28.0,
        animation_type: 'none',
      },
      {
        type: 'video',
        url: 'https://example.com/chernobyl-exterior.mp4',
        path: 'assets/media/chernobyl-exterior.mp4',
        clip_id: 'chernobyl-exterior-clip',
        clip_in: 0.0,
        clip_out: 37.0,
      },
    ],
  };
  fs.writeFileSync(path.join(ep, 'MediaPlan.json'), JSON.stringify(plan, null, 2));
}

// resetKW32 — vo_timeline sentinel: last vo_item.end_sec patched to 80.0
// (ShotList total is 65.348s) — any tab reading ShotList for duration will show
// wrong value; any tab reading vo_timeline will show >= 80.0
function resetKW32() {
  const ep = getEpDir();
  const vp = JSON.parse(fs.readFileSync(path.join(FIXTURE_EP, 'VOPlan.en.json'), 'utf8'));
  vp.vo_items[vp.vo_items.length - 1].end_sec = 80.0;
  fs.writeFileSync(path.join(ep, 'VOPlan.en.json'), JSON.stringify(vp, null, 2));
  ['MusicPlan.json', 'SfxPlan.json', 'MediaPlan.json'].forEach(f => {
    const p = path.join(ep, f);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  });
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
  resetKW19c,
  resetKW22,
  resetKW24,
  resetKW26,
  resetKW29,
  resetKW30,
  resetKW31,
  resetKW32,
  resetKW80,
  // EP_DIR kept for backward compat — resolves dynamically via getEpDir()
  get EP_DIR() { return getEpDir(); },
};
