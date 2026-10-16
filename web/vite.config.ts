import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vitest/config';
import { buildCsp } from './src/csp.ts';

// GitHub Pages serves the site from /<repo>/.
const BASE = '/SilentSpeak/';

/**
 * Adds the CSP meta tag to production builds only. The dev server injects
 * inline scripts for hot reload, which a strict policy would block.
 */
function contentSecurityPolicy(): Plugin {
  return {
    name: 'silentspeak:csp-meta',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler: () => [
        {
          tag: 'meta',
          attrs: { 'http-equiv': 'Content-Security-Policy', content: buildCsp() },
          injectTo: 'head-prepend',
        },
      ],
    },
  };
}

export default defineConfig({
  base: BASE,
  plugins: [react(), tailwindcss(), contentSecurityPolicy()],
  build: {
    target: 'es2022',
    sourcemap: false,
    emptyOutDir: true,
    // Keep every asset a file; inlined data: URIs would need a looser CSP.
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 700,
  },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
