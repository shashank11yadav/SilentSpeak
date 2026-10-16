import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { captureClip, failCamera, ready } from './helpers';

test.describe('first load', () => {
  test('shows the untrained preview notice and opens on GRID words', async ({ page }) => {
    await ready(page);
    await expect(page.getByText('Untrained preview model.')).toBeVisible();
    await expect(page.getByRole('tab', { name: 'GRID words' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    await expect(page.getByText('GRID corpus, Cooke et al., CC BY 4.0')).toBeVisible();
  });

  test('ships a strict CSP and makes no third-party requests', async ({ page }) => {
    const external: string[] = [];
    page.on('request', (req) => {
      const url = new URL(req.url());
      if (!['localhost', '127.0.0.1'].includes(url.hostname) && url.protocol.startsWith('http')) {
        external.push(req.url());
      }
    });
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    await ready(page);
    const csp = await page
      .locator('meta[http-equiv="Content-Security-Policy"]')
      .getAttribute('content');
    expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
    expect(csp).toContain("connect-src 'self'");
    expect(external).toEqual([]);
    expect(errors.filter((e) => /Content Security Policy/i.test(e))).toEqual([]);
  });

  for (const mode of ['GRID words', 'Teach words', 'Silent remote']) {
    test(`has no axe violations: ${mode}`, async ({ page }) => {
      await ready(page);
      await page.getByRole('tab', { name: mode }).click();
      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
    });
  }

  test('has no axe violations in Replay', async ({ page }) => {
    await ready(page);
    await page.getByRole('radio', { name: 'Replay' }).click();
    await expect(page.getByRole('button', { name: 'Read this clip' })).toBeEnabled();
    const results = await new AxeBuilder({ page }).analyze();
    expect(results.violations).toEqual([]);
  });
});

test.describe('camera problems never leave a blank page', () => {
  for (const [name, message] of [
    ['NotAllowedError', 'Camera access is blocked'],
    ['NotFoundError', 'No camera found'],
    ['NotReadableError', 'The camera is busy'],
  ] as const) {
    test(`${name} explains itself and offers Replay`, async ({ page }) => {
      await failCamera(page, name);
      await ready(page);
      await page.getByRole('button', { name: 'Start camera' }).click();
      const alert = page.getByRole('alert').filter({ hasText: message });
      await expect(alert).toBeVisible();
      await alert.getByRole('button', { name: 'Use replay clips' }).click();
      await expect(page.getByRole('button', { name: 'Read this clip' })).toBeVisible();
    });
  }
});

test.describe('model load failure', () => {
  test('a missing model shows an error and Retry recovers', async ({ page }) => {
    await page.route('**/models/model.onnx', (route) =>
      route.fulfill({ status: 404, body: 'gone' }),
    );
    await page.goto('./');
    const alert = page.getByRole('alert').filter({ hasText: 'could not load' });
    await expect(alert).toBeVisible({ timeout: 30_000 });
    await expect(alert).toContainText('HTTP 404');
    await page.unroute('**/models/model.onnx');
    await alert.getByRole('button', { name: 'Retry' }).click();
    await expect(page.getByTestId('backend')).toHaveText(/WebAssembly|WebGPU/, { timeout: 30_000 });
    await expect(alert).toBeHidden();
  });
});

test.describe('replay', () => {
  test('reads a bundled clip through the model and shows the truth', async ({ page }) => {
    await ready(page);
    await page.getByRole('radio', { name: 'Replay' }).click();
    await page.getByRole('radio', { name: /blue/ }).click();
    await page.getByRole('button', { name: 'Read this clip' }).click();
    const result = page.getByTestId('grid-result');
    await expect(result).toContainText('The clip says blue');
    await expect(result.getByRole('listitem')).toHaveCount(3);
    await expect(page.getByRole('radio', { name: /^Colours/ })).toBeChecked();
  });
});

test.describe('GRID words', () => {
  test('a capture gives three guesses within the chosen category', async ({ page }) => {
    await ready(page);
    await page.getByText('Digits').click();
    await captureClip(page, 'clip-5');
    const result = page.getByTestId('grid-result');
    await expect(result.getByRole('listitem')).toHaveCount(3);
    await expect(result).toContainText(/You said|Not sure/);
  });

  test('a too-short capture asks to say it again', async ({ page }) => {
    await ready(page);
    await captureClip(page, 'clip-1', 'camera', 3);
    await expect(page.getByRole('alert')).toContainText('too short');
    await expect(page.getByTestId('grid-result')).toContainText('will appear here');
  });
});

test.describe('teach your own words and the silent remote', () => {
  test('quick setup, recognition and remote control', async ({ page }) => {
    // The untrained stub embeds every clip almost alike, so loosen its "not
    // sure" margin; the flow, not the stub's accuracy, is under test.
    await page.route('**/models/model-meta.json', async (route) => {
      const res = await route.fetch();
      const meta = (await res.json()) as { thresholds: Record<string, number> };
      meta.thresholds = { protoMargin: 1e-6, protoMinSimilarity: 0 };
      await route.fulfill({ response: res, json: meta });
    });
    await ready(page);
    await page.getByRole('tab', { name: 'Teach words' }).click();
    await page.getByRole('button', { name: /Start quick setup/ }).click();
    await expect(page.getByTestId('setup')).toContainText('take 1 of 9');
    const clipFor: Record<string, string> = { play: 'clip-1', pause: 'clip-2', next: 'clip-4' };
    for (const word of ['play', 'pause', 'next']) {
      for (let take = 0; take < 3; take++) {
        await expect(page.getByTestId('setup')).toContainText(word);
        const before = await page.getByTestId('setup').textContent();
        await captureClip(page, clipFor[word]!);
        if (!(word === 'next' && take === 2)) {
          await expect(page.getByTestId('setup')).not.toHaveText(before ?? '');
        }
      }
    }
    await expect(page.getByTestId('setup')).toBeHidden();
    await expect(page.getByText('3 takes')).toHaveCount(3);

    await captureClip(page, 'clip-4');
    await expect(page.getByTestId('teach-result')).toContainText(/You said\s*next/);

    await page.getByRole('tab', { name: 'Silent remote' }).click();
    await expect(page.getByLabel('Next', { exact: true })).toHaveValue('next');
    const title = await page.getByTestId('scene-title').textContent();
    await captureClip(page, 'clip-4');
    await expect(page.getByTestId('remote-result')).toContainText('Next');
    await expect(page.getByTestId('scene-title')).not.toHaveText(title ?? '');

    // Taught words survive a reload (IndexedDB), and Clear all removes them.
    await ready(page);
    await page.getByRole('tab', { name: 'Teach words' }).click();
    await expect(page.getByText('3 takes')).toHaveCount(3);
    await page.getByRole('button', { name: 'Clear all' }).click();
    await expect(page.getByRole('button', { name: /Start quick setup/ })).toBeVisible();
  });
});
