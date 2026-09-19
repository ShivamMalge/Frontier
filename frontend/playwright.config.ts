import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against a *running stack*, not a mocked one.
 *
 * The jsdom tests in `src/` mount pages against a stubbed API; they cannot tell
 * you that nginx proxies correctly, that the worker picks a job up, or that a
 * chart actually paints. These can. Point `E2E_BASE_URL` at whatever is
 * serving the app -- CI points it at `docker compose up`.
 */
export default defineConfig({
  testDir: "./e2e",
  // A pipeline run goes through Redis and a worker process; be patient.
  timeout: 180_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8080",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
