const LAST_RELOAD_KEY = 'silentspeak:stale-chunk-reload-at';
const MIN_RELOAD_INTERVAL_MS = 10_000;

/**
 * A tab opened before a redeploy still references the old chunk file names,
 * which GitHub Pages no longer serves, so loading a lazy chunk would leave a
 * blank page. Reloading picks up the new build. The interval check stops a
 * reload loop when a chunk is missing for some other reason.
 */
export function reloadOnStaleChunk(win: Window = window): void {
  win.addEventListener('vite:preloadError', (event) => {
    const now = Date.now();
    let lastReload: number;
    try {
      lastReload = Number(win.sessionStorage.getItem(LAST_RELOAD_KEY)) || 0;
      win.sessionStorage.setItem(LAST_RELOAD_KEY, String(now));
    } catch {
      // Without storage there is no way to tell a first failure from a loop.
      return;
    }
    if (now - lastReload < MIN_RELOAD_INTERVAL_MS) return;
    event.preventDefault();
    win.location.reload();
  });
}
