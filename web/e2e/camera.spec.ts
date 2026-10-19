import { expect, test } from '@playwright/test';
import { ready } from './helpers';

// Chromium's fake camera shows a test pattern with no face in it.
test.use({
  permissions: ['camera'],
  launchOptions: {
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  },
});

test('says it is looking for a face and refuses to guess', async ({ page }) => {
  await ready(page);
  await page.getByRole('button', { name: 'Start camera' }).click();
  await expect(page.getByTestId('tracking-status')).toHaveText('Looking for a face', {
    timeout: 45_000,
  });
  const hold = page.getByRole('button', { name: /Hold to read/ });
  await hold.hover();
  await page.mouse.down();
  await page.waitForTimeout(800);
  await page.mouse.up();
  await expect(page.getByRole('alert')).toContainText('could not see your face');
  await expect(page.getByTestId('grid-result')).toContainText('will appear here');
});

test('releases a camera that starts after the tab was hidden', async ({ page }) => {
  // Hold getUserMedia until the test lets it finish, and keep the stream it returns.
  await page.addInitScript(() => {
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    const w = window as unknown as {
      __release: () => void;
      __stream: MediaStream | null;
    };
    w.__stream = null;
    const gate = new Promise<void>((resolve) => {
      w.__release = resolve;
    });
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      await gate;
      w.__stream = await real(constraints);
      return w.__stream;
    };
  });
  await ready(page);
  await page.getByRole('button', { name: 'Start camera' }).click();
  await expect(page.getByText('Starting the camera and face tracker…')).toBeVisible();
  await page.evaluate(() => {
    Object.defineProperty(document, 'visibilityState', { value: 'hidden', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));
    (window as unknown as { __release: () => void }).__release();
  });
  await page.waitForFunction(
    () => (window as unknown as { __stream: MediaStream | null }).__stream !== null,
  );
  await expect
    .poll(() =>
      page.evaluate(() =>
        (window as unknown as { __stream: MediaStream }).__stream
          .getTracks()
          .every((t) => t.readyState === 'ended'),
      ),
    )
    .toBe(true);
  await expect(page.getByText('Camera paused')).toBeVisible();
  await expect(page.getByTestId('tracking-status')).toHaveCount(0);
});
