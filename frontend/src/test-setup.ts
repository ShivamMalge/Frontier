/**
 * What jsdom does not implement but the app legitimately uses.
 *
 * `matchMedia` drives the light/dark palette swap. Stubbing it here keeps the
 * shim out of the component, where it would be dead weight in every browser.
 */
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

// ECharts is mocked in the tests that need it; ResizeObserver is stubbed here
// so anything that slips through mounts rather than throwing.
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

import { cleanup } from "@testing-library/react";
// With `globals: false`, testing-library does not register its own cleanup, so
// a second render would otherwise find the first one's DOM still mounted.
import { afterEach } from "vitest";

afterEach(cleanup);
