import { REPO_URL } from '../links';

export function SiteHeader() {
  return (
    <header className="border-b border-line bg-paper/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-3 px-4">
        <a href="#top" className="flex shrink-0 items-center gap-2 font-semibold tracking-tight">
          <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width={26} height={26} />
          <span>SilentSpeak</span>
          <span className="rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink-muted">
            Live demo
          </span>
        </a>
        <nav aria-label="Main" className="flex items-center gap-1 text-sm whitespace-nowrap">
          <a
            href="#how-it-works"
            className="hidden rounded-md px-2.5 py-2 hover:bg-surface-muted sm:inline-block"
          >
            How it works
          </a>
          <a href={REPO_URL} className="rounded-md px-2 py-2 hover:bg-surface-muted sm:px-2.5">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
