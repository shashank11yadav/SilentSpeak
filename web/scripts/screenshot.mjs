// Captures screenshots of the production build for review and the README.
// Usage: npm run build && npm run preview (in another shell) &&
//        node scripts/screenshot.mjs <outdir> [width ...]
import { chromium } from '@playwright/test';

const url = process.env.SCREENSHOT_URL ?? 'http://localhost:5301/SilentSpeak/';
const outDir = process.argv[2] ?? '../docs';
const widths = process.argv.slice(3).map(Number);
const scheme = process.env.SCREENSHOT_SCHEME === 'dark' ? 'dark' : 'light';
const replay = process.env.SCREENSHOT_REPLAY === '1';

const browser = await chromium.launch({
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});
for (const width of widths.length ? widths : [1280]) {
  const page = await browser.newPage({
    viewport: { width, height: 900 },
    deviceScaleFactor: 1,
    colorScheme: scheme,
    reducedMotion: 'reduce',
  });
  await page.goto(url);
  await page.getByText('Pick what kind of word').waitFor({ timeout: 30_000 });
  if (replay) {
    await page.getByRole('radio', { name: 'Replay' }).click();
    await page.getByRole('button', { name: 'Read this clip' }).click();
    await page.getByTestId('grid-result').getByRole('listitem').first().waitFor();
  }
  await page.screenshot({
    path: `${outDir}/screenshot-${width}${replay ? '-replay' : ''}-${scheme}.png`,
    fullPage: process.env.SCREENSHOT_FULL === '1',
  });
  await page.close();
}
await browser.close();
