import { defineConfig, devices } from '@playwright/test';

const PORT = 5302;
const BASE_URL = `http://localhost:${PORT}/SilentSpeak/`;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env['CI'],
  retries: process.env['CI'] ? 1 : 0,
  reporter: process.env['CI'] ? 'github' : 'list',
  timeout: 60_000,
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // A production build in test mode: the CSP meta tag is in place, and the
  // landmark-injection hook (compiled out of the real build) is available.
  webServer: {
    command: `npm run build:test && npx vite preview --outDir dist-test --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env['CI'],
    timeout: 120_000,
  },
});
