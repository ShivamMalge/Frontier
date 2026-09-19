import { expect, test, type Page } from "@playwright/test";

/**
 * The path nothing else covers: browser → nginx → API → Redis → worker → back.
 *
 * Unit tests stub the API; the API tests never open a browser. Everything
 * between the two is what breaks in deployment, and this is the only test that
 * exercises it.
 */

/** Fails the test on any console error, which is how a broken render shows up. */
function failOnConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

test.describe("the served app", () => {
  test("reports the API it is talking to", async ({ page }) => {
    const errors = failOnConsoleErrors(page);
    await page.goto("/run");

    // The sidebar reads /health, so this proves the proxy as well as the render.
    await expect(page.getByText(/Frontier API/)).toBeVisible();
    await expect(page.getByText(/jobs: (redis|memory)/)).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("serves a client-side route directly, not just the index", async ({ page }) => {
    // A deep link is a request nginx has to rewrite to index.html; getting this
    // wrong shows up as a 404 only on reload, which is easy to miss by hand.
    const response = await page.goto("/portfolio");
    expect(response?.status()).toBe(200);
    await expect(page.getByRole("heading", { name: "Portfolio" })).toBeVisible();
  });

  test("runs a pipeline end to end and renders the result", async ({ page }) => {
    const errors = failOnConsoleErrors(page);
    await page.goto("/run");

    await page.getByRole("combobox").first().selectOption("lightgbm");
    await page.getByRole("button", { name: "Run pipeline" }).click();

    // Queued -> running -> succeeded, executed by the worker container.
    await expect(page.getByText(/finished in/)).toBeVisible({ timeout: 150_000 });

    // The numbers, not just the chrome: a MASE value, a strategy, a weight.
    await expect(page.getByRole("heading", { name: "Forecast quality" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Strategy performance" })).toBeVisible();
    await expect(page.locator("td", { hasText: /^\d\.\d{3}$/ }).first()).toBeVisible();
    await expect(page.locator("td", { hasText: /%$/ }).first()).toBeVisible();

    // Charts are canvases: assert one was painted, not merely mounted.
    const painted = await page.locator("canvas").first().evaluate((node) => {
      const canvas = node as HTMLCanvasElement;
      return canvas.width > 0 && canvas.height > 0;
    });
    expect(painted).toBe(true);

    expect(errors).toEqual([]);
  });

  test("explains why it will not run, rather than going quietly dead", async ({ page }) => {
    await page.goto("/portfolio");
    // A covariance needs a pair. Strip the selection down to one ticker: the
    // button must both disable *and* say why -- a dead control with no
    // explanation is the failure this guards against.
    const chips = page.locator(".chip[aria-pressed='true']");
    const extra = (await chips.count()) - 1;
    for (let i = 0; i < extra; i += 1) await chips.first().click();

    await expect(page.getByRole("button", { name: "Optimize" })).toBeDisabled();
    await expect(page.getByText(/Select at least two tickers/)).toBeVisible();
  });
});
