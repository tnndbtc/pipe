const { spawn } = require('child_process');
const path = require('path');
const fs   = require('fs');
const os   = require('os');
const fixtureState = require('./fixture_state');

const PIPE_DIR    = path.join(__dirname, '..', '..');
const SERVER_SCRIPT = path.join(PIPE_DIR, 'code', 'http', 'test_server.py');
const FIXTURE_SRC = path.join(__dirname, '..', 'fixtures');

/** Recursively copy a directory tree (no symlinks). */
function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const dstPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, dstPath);
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
  }
}

async function startTestServer() {
  // Create a fresh temp dir for this test run and populate it with fixtures.
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'pipe-test-'));
  copyDir(FIXTURE_SRC, tmpRoot);

  // Tell fixture_state where the live ep_dir is so resetKW*() and file reads
  // target the same directory the server will use.
  fixtureState.setPipeTestDir(tmpRoot);

  const proc = spawn('python3', [SERVER_SCRIPT, '--test-mode', '--port', '19999'], {
    cwd: PIPE_DIR,
    env: { ...process.env, PIPE_TEST_DIR: tmpRoot },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc._pipeTestDir = tmpRoot;   // expose so fixture_state can read it
  proc.stdout.on('data', d => process.stdout.write('[server] ' + d));
  proc.stderr.on('data', d => process.stderr.write('[server] ' + d));

  // Wait until server responds (max 20s)
  const start = Date.now();
  while (Date.now() - start < 20000) {
    try {
      const { default: http } = await import('http');
      await new Promise((resolve, reject) => {
        const req = http.get('http://localhost:19999/', res => {
          res.resume();
          resolve(res.statusCode);
        });
        req.on('error', reject);
        req.setTimeout(1000, () => { req.destroy(); reject(new Error('timeout')); });
      });
      break;
    } catch (_) {
      await new Promise(r => setTimeout(r, 400));
    }
  }
  return proc;
}

async function stopTestServer(proc) {
  proc.kill('SIGTERM');
  await new Promise(r => setTimeout(r, 600));
  // Clean up the temp dir created for this run.
  if (proc._pipeTestDir) {
    try { fs.rmSync(proc._pipeTestDir, { recursive: true, force: true }); } catch (_) {}
  }
}

module.exports = { startTestServer, stopTestServer };
