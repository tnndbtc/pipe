const fs   = require('fs');
const path = require('path');

const STEP_OUT_DIR = path.join(__dirname, '..', 'step_outputs');

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

// KW-1 start: no VOPlan, no MusicPlan
function resetKW1() {
  const ep = getEpDir();
  const vp = path.join(ep, 'VOPlan.en.json');
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(vp)) fs.unlinkSync(vp);
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-2/KW-9 start: merged VOPlan present, fresh MusicPlan deleted
function resetKW2() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'manifest_merge.VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  const mp = path.join(ep, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-13 start: merged VOPlan + committed MusicPlan both present
function resetKW13() {
  const ep = getEpDir();
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'manifest_merge.VOPlan.en.json'),
    path.join(ep, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'music_review.MusicPlan.json'),
    path.join(ep, 'MusicPlan.json')
  );
  // Remove any stale MediaPreviewPack output from a previous run
  const packDir = path.join(ep, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

module.exports = {
  getEpDir,
  setPipeTestDir,
  voplan,
  musicplan,
  resetKW1,
  resetKW2,
  resetKW13,
  // EP_DIR kept for backward compat — resolves dynamically via getEpDir()
  get EP_DIR() { return getEpDir(); },
};
