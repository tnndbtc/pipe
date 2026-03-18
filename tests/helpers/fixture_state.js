const fs   = require('fs');
const path = require('path');

const FIXTURE_ROOT = path.join(__dirname, '..', 'fixtures', 'projects');
const STEP_OUT_DIR = path.join(__dirname, '..', 'step_outputs');
const EP_DIR = path.join(FIXTURE_ROOT, 'test-proj', 'episodes', 's01e01');

function voplan() {
  return JSON.parse(fs.readFileSync(path.join(EP_DIR, 'VOPlan.en.json'), 'utf8'));
}

function musicplan() {
  return JSON.parse(fs.readFileSync(path.join(EP_DIR, 'MusicPlan.json'), 'utf8'));
}

// KW-1 start: no VOPlan, no MusicPlan
function resetKW1() {
  const vp = path.join(EP_DIR, 'VOPlan.en.json');
  const mp = path.join(EP_DIR, 'MusicPlan.json');
  if (fs.existsSync(vp)) fs.unlinkSync(vp);
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-2/KW-9 start: merged VOPlan present, fresh MusicPlan deleted
function resetKW2() {
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'manifest_merge.VOPlan.en.json'),
    path.join(EP_DIR, 'VOPlan.en.json')
  );
  const mp = path.join(EP_DIR, 'MusicPlan.json');
  if (fs.existsSync(mp)) fs.unlinkSync(mp);
}

// KW-13 start: merged VOPlan + committed MusicPlan both present
function resetKW13() {
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'manifest_merge.VOPlan.en.json'),
    path.join(EP_DIR, 'VOPlan.en.json')
  );
  fs.copyFileSync(
    path.join(STEP_OUT_DIR, 'music_review.MusicPlan.json'),
    path.join(EP_DIR, 'MusicPlan.json')
  );
  // Remove any stale MediaPreviewPack output from a previous run
  const packDir = path.join(EP_DIR, 'assets', 'media', 'MediaPreviewPack');
  if (fs.existsSync(packDir)) fs.rmSync(packDir, { recursive: true, force: true });
}

module.exports = { voplan, musicplan, resetKW1, resetKW2, resetKW13, EP_DIR };
