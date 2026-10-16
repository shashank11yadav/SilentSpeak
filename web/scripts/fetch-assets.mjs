// Puts every runtime file the app needs under public/, so the site never loads
// anything from a third-party origin:
// - MediaPipe's WASM, copied from node_modules (version pinned in package-lock);
// - the face landmarker model, downloaded once and checked against a pinned
//   SHA-256. A mismatch fails the build.
// The pipeline pins the same model hash (pipeline/src/silentspeak/assets.py).
// ONNX Runtime's WASM needs no copy: Vite emits it as a hashed asset because
// the runtime refers to it with new URL(..., import.meta.url).
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const modules = join(root, 'node_modules');
const pub = join(root, 'public');

const MODEL = {
  url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  dest: join(pub, 'mediapipe', 'face_landmarker.task'),
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

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchModel() {
  if (await exists(MODEL.dest)) {
    if (sha256(await readFile(MODEL.dest)) === MODEL.sha256) return 'cached';
  }
  const res = await fetch(MODEL.url);
  if (!res.ok) throw new Error(`download failed: ${res.status} ${MODEL.url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const got = sha256(buf);
  if (got !== MODEL.sha256) {
    throw new Error(`face_landmarker.task SHA-256 mismatch: expected ${MODEL.sha256}, got ${got}`);
  }
  await mkdir(dirname(MODEL.dest), { recursive: true });
  await writeFile(`${MODEL.dest}.part`, buf);
  await rename(`${MODEL.dest}.part`, MODEL.dest);
  return 'downloaded';
}

for (const [from, to] of COPIES) {
  const dest = join(pub, to);
  await mkdir(dirname(dest), { recursive: true });
  await copyFile(join(modules, from), dest);
}
const status = await fetchModel();
process.stdout.write(`assets ready (${COPIES.length} files copied, landmarker model ${status})\n`);
