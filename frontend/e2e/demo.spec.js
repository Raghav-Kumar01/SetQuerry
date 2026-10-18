import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = name => path.join(root, 'output/playwright/fixtures', name);

test.beforeAll(() => {
  const python = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  execFileSync(python, ['integration/create_demo_inputs.py'], { cwd: root });
});

const architecture = process.env.SETQUERY_TEST_MODE === 'architecture';
const notice = item => architecture || item.label === 'UNTRAINED' ? 'Untrained model' : 'Baseline';
const cases = [
  { model: 'single_image', route: 'single', files: ['after.png'], label: 'UNTRAINED' },
  { model: 'change_analysis', route: 'change', files: ['before.png', 'after.png'], label: 'BASELINE' },
  { model: 'optical_sar', route: 'optical-sar', files: ['optical.png', 'sar.png'], label: 'BASELINE' },
];

for (const item of cases) {
  test(`${item.model}: upload -> compute -> rendered result`, async ({ page }) => {
    const failures = [];
    page.on('pageerror', error => failures.push(error.message));
    await page.goto('/');
    await page.getByLabel('Analysis model').selectOption(item.model);
    if (item.model === 'single_image') await page.getByLabel('Question', { exact: true }).fill('Is there a road?');
    await page.getByLabel('Upload images').setInputFiles(item.files.map(fixture));
    // Hold the actual request briefly to verify loading UI; no response is mocked.
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    await page.route(`**/api/analyze/${item.route}`, async route => { await gate; await route.continue(); });
    const responsePromise = page.waitForResponse(response => response.url().endsWith(`/api/analyze/${item.route}`));
    await page.getByRole('button', { name: 'Analyze Imagery' }).click();
    await expect(page.getByText('Running Analysis', { exact: true })).toBeVisible();
    release();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    const data = await response.json();
    expect(data.is_trained).toBe(false);
    await expect(page.getByTestId('inference-notice')).toContainText(notice(item));
    await expect(page.locator('.inference-notice')).toHaveCount(0);
    await expect(page.getByTestId('inference-notice')).toHaveCSS('color', 'rgb(156, 163, 175)');
    await expect(page.getByText('Running Analysis', { exact: true })).toHaveCount(0);
    if (item.route === 'single') {
      await expect(page.getByText(data.answer, { exact: true }).first()).toBeVisible();
      expect(data.query).toBe('Is there a road?');
      const followup = page.waitForResponse(response => response.url().endsWith('/api/analyze/single'));
      await page.getByPlaceholder('Ask about this satellite scene...').fill('How many buildings are in this image?');
      await page.getByRole('button', { name: 'Submit Query' }).click();
      expect((await (await followup).json()).query).toBe('How many buildings are in this image?');
      await expect(page.getByTestId('inference-notice')).toContainText('Untrained model');
    } else if (item.route === 'change') {
      const mask = page.getByAltText('Model-predicted binary change mask');
      await expect(mask).toBeVisible();
      await expect.poll(() => mask.evaluate(image => image.naturalWidth)).toBe(256);
      await expect(page.getByText(`${data.change_percentage.toFixed(2)}%`, { exact: true }).first()).toBeVisible();
    } else if (architecture) {
      expect(data.logits).toHaveLength(2);
      await expect(page.getByText('Uncalibrated class score', { exact: true })).toBeVisible();
      await expect(page.getByText(data.answer, { exact: true }).first()).toBeVisible();
    } else {
      await expect(page.getByText(data.similarity_score.toFixed(4), { exact: true })).toBeVisible();
      await expect(page.getByText('Structural similarity (0–1), not probability', { exact: true })).toBeVisible();
    }
    expect(failures).toEqual([]);
    await page.screenshot({ path: path.join(root, 'output/playwright', `${architecture ? "architecture-" : ""}${item.route}.png`), fullPage: true });
  });
}

test('invalid image displays error and supports retry', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Upload images').setInputFiles(fixture('invalid.png'));
  await page.getByRole('button', { name: 'Analyze Imagery' }).click();
  await expect(page.getByText("Invalid image: 'invalid.png'", { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Try Again' }).click();
  await expect(page.getByRole('button', { name: 'Analyze Imagery' })).toBeEnabled();
});

test('direct route without uploads shows recovery action', async ({ page }) => {
  await page.goto('/analyze/change');
  await expect(page.getByText(/Two images required/)).toBeVisible();
  await page.getByRole('button', { name: 'Back to Upload' }).click();
  await expect(page.getByLabel('Analysis model')).toBeVisible();
});
