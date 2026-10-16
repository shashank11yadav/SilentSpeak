import { expect, type Page } from '@playwright/test';

// Installed by src/lib/test-hook.ts in `--mode test` builds.
declare global {
  interface Window {
    __silentspeak?: {
      captureClip: (id: string, as?: 'camera' | 'replay', limit?: number) => Promise<void>;
    };
  }
}

export async function ready(page: Page) {
  await page.goto('./');
  await expect(page.getByTestId('backend')).not.toHaveText('loading…', { timeout: 30_000 });
}

export async function captureClip(
  page: Page,
  id: string,
  as: 'camera' | 'replay' = 'camera',
  limit?: number,
) {
  await page.waitForFunction(() => window.__silentspeak !== undefined);
  await page.evaluate(
    ([clip, mode, n]) => window.__silentspeak?.captureClip(clip, mode as 'camera', n),
    [id, as, limit] as const,
  );
}

/** Make getUserMedia fail the way a browser does for the given reason. */
export async function failCamera(page: Page, name: string) {
  await page.addInitScript((errorName) => {
    navigator.mediaDevices.getUserMedia = () =>
      Promise.reject(new DOMException('blocked for the test', errorName));
  }, name);
}
