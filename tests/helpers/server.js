const { spawn } = require('child_process');
const path = require('path');

const PIPE_DIR = path.join(__dirname, '..', '..');
const SERVER_SCRIPT = path.join(PIPE_DIR, 'code', 'http', 'test_server.py');

async function startTestServer() {
  const proc = spawn('python3', [SERVER_SCRIPT, '--test-mode', '--port', '19999'], {
    cwd: PIPE_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
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
}

module.exports = { startTestServer, stopTestServer };
