import { test, expect } from '@playwright/test'

test('Navigate to root, click run, wait for results', async ({ page }) => {
  await page.goto('/')

  const fileInputElement = page.locator('[type="file"]')

  // NOTE: Playwright reads the entire file so in practice only use MB size files...
  await fileInputElement.setInputFiles('public/sample_maxwell_raw.h5')

  // Click the run button.
  // In your App.vue the run button is a v-app-bar-nav-icon with icon="mdi-play".
  // You might want to add a data attribute (e.g., data-cy="run-button") to ease selection.
  // For now, we'll target it by its color property if possible.
  await page.click('[data-cy="run-button"]')

  // Wait for the status element to display the expected value.
  // Replace '[data-cy=status-label]' and 'Expected Value' with your actual selector and value.
  await expect(page.locator('[data-cy="status"]')).toHaveText(/100%/, {
    timeout: 60000, // Wait up to 60 seconds for the status to change
  })
})
