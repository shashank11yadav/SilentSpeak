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
