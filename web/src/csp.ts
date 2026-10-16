/**
 * GitHub Pages cannot send response headers, so the policy ships as a <meta>
 * tag that Vite adds at build time. frame-ancestors, report-uri and sandbox
 * are ignored in a meta policy, which is why they are not here.
 *
 * Everything is same-origin. 'wasm-unsafe-eval' lets MediaPipe and ONNX
 * Runtime compile their WebAssembly; it does not allow JavaScript eval.
 * blob: workers and media are for the camera stream and MediaPipe's loader.
 */
export const CSP_DIRECTIVES: Readonly<Record<string, readonly string[]>> = {
  'default-src': ["'self'"],
  'script-src': ["'self'", "'wasm-unsafe-eval'"],
  'style-src': ["'self'"],
  'img-src': ["'self'", 'blob:'],
  'media-src': ["'self'", 'blob:'],
  'font-src': ["'self'"],
  'connect-src': ["'self'"],
  'worker-src': ["'self'", 'blob:'],
  'manifest-src': ["'self'"],
  'frame-src': ["'none'"],
  'object-src': ["'none'"],
  'base-uri': ["'self'"],
  'form-action': ["'none'"],
};

export function buildCsp(directives = CSP_DIRECTIVES): string {
  return Object.entries(directives)
    .map(([name, values]) => `${name} ${values.join(' ')}`)
    .join('; ');
}
