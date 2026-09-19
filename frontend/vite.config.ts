/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies to the API rather than calling it cross-origin, so the
// browser sees one origin in development and in the nginx image alike. That
// keeps `SO_CORS_ORIGINS` out of the critical path: if CORS is misconfigured,
// only a direct-to-8000 build notices, and there isn't one.
const API = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API, changeOrigin: true },
      "/health": { target: API, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // ECharts is most of the bundle and changes only when it is upgraded.
        // Splitting it keeps app edits from invalidating it in a browser cache.
        manualChunks: (id: string) => (id.includes("node_modules/echarts") || id.includes("node_modules/zrender") ? "echarts" : undefined),
      },
    },
  },
  test: {
    // jsdom, not a real browser: these tests check that pages mount, fetch and
    // render their tables -- not that a canvas painted. The chart component is
    // stubbed where it appears, since jsdom has no canvas.
    environment: "jsdom",
    globals: false,
    setupFiles: ["src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
