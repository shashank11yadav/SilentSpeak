# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/shashank11yadav/SilentSpeak/security/advisories/new)
rather than a public issue. I will acknowledge a report within a week.

## Scope

The deployed site is a static bundle (HTML, JS, CSS, WASM, an ONNX model and a
few images) on GitHub Pages. It has no server, no accounts, no cookies, no
analytics and no third-party requests. The camera stream is processed in the
page and never uploaded; taught words are stored as numeric embeddings in the
browser's IndexedDB and can be cleared from the page.

The realistic risks are in the build and supply chain:

- Dependencies are pinned (`web/package-lock.json`, `pipeline/uv.lock`) and
  updated by Dependabot. CI runs `npm audit --omit=dev` and `pip-audit`.
- GitHub Actions are pinned to commit SHAs and run with `contents: read` unless
  a job needs more (Pages deploy, CodeQL upload).
- The MediaPipe face landmarker model is downloaded over HTTPS and checked
  against a pinned SHA-256, both by the pipeline and by the web build; a
  mismatch fails the build. MediaPipe and ONNX Runtime WASM files are copied
  from the locked npm packages and served from the same origin.
- The page ships a Content-Security-Policy that allows only same-origin
  scripts, styles, workers and fetches, plus `'wasm-unsafe-eval'`, which
  WebAssembly needs.
- The pipeline only opens GRID zips whose size matches the download manifest,
  reads frames from the zip in memory, and never extracts archive paths to disk.
