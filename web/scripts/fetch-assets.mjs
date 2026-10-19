// Puts every runtime file the app needs under public/, so the site never loads
// anything from a third-party origin:
// - MediaPipe's WASM, copied from node_modules (version pinned in package-lock);
// - the face landmarker model, downloaded once and checked against a pinned
//   SHA-256 before it is written (see download.mjs). A mismatch fails the build.
// The pipeline pins the same model hash (pipeline/src/silentspeak/assets.py).
// ONNX Runtime's WASM needs no copy: Vite emits it as a hashed asset because
// the runtime refers to it with new URL(..., import.meta.url).
import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetchVerified } from './download.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const modules = join(root, 'node_modules');
const pub = join(root, 'public');

const MODEL = {
  url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  dest: join(pub, 'mediapipe', 'face_landmarker.task'),
  maxBytes: 16 * 1024 * 1024, // the pinned file is 3.7 MB
};

const COPIES = [
  [
    '@mediapipe/tasks-vision/wasm/vision_wasm_internal.js',
    'mediapipe/wasm/vision_wasm_internal.js',
  ],
  [
    '@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm',
    'mediapipe/wasm/vision_wasm_internal.wasm',
  ],
  [
    '@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js',
    'mediapipe/wasm/vision_wasm_nosimd_internal.js',
  ],
  [
    '@mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm',
    'mediapipe/wasm/vision_wasm_nosimd_internal.wasm',
  ],
];

for (const [from, to] of COPIES) {
  const dest = join(pub, to);
  await mkdir(dirname(dest), { recursive: true });
  await copyFile(join(modules, from), dest);
}
const status = await fetchVerified(MODEL);
process.stdout.write(`assets ready (${COPIES.length} files copied, landmarker model ${status})\n`);
