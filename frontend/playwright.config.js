import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const frontend = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(frontend, '..');
const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const absent = number => path.join(root, 'output/playwright', `missing-checkpoint-${number}.pt`);

export default defineConfig({
  testDir: './e2e',
  timeout: 60000,
  workers: 1,
  outputDir: '../output/playwright/test-results',
  reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:5175', channel: 'chrome', reducedMotion: 'reduce', viewport: { width: 1440, height: 1000 }, screenshot: 'only-on-failure' },
  webServer: [
    { command: `"${python}" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8766`, cwd: root,
      url: 'http://127.0.0.1:8766/api/health', reuseExistingServer: false,
      env: { MODEL_MODE: process.env.SETQUERY_TEST_MODE || 'demo', SETQUERY_CORS_ORIGINS: 'http://127.0.0.1:5175',
        SETQUERY_MODEL1_CHECKPOINT: absent(1), SETQUERY_MODEL2_CHECKPOINT: absent(2), SETQUERY_MODEL3_CHECKPOINT: absent(3) } },
    { command: 'npm run dev -- --host 127.0.0.1 --port 5175 --strictPort', cwd: frontend,
      url: 'http://127.0.0.1:5175', reuseExistingServer: false,
      env: { VITE_API_BASE_URL: 'http://127.0.0.1:8766/api' } },
  ],
});
