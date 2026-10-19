// Downloads a build-time asset and keeps it only if it matches a pinned
// SHA-256. The bytes are checked in memory before anything touches the disk,
// then written to a temporary file next to the destination and renamed into
// place, so a failed or tampered download never leaves a file behind.
import { createHash, randomBytes } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export const sha256Hex = (buf) => createHash('sha256').update(buf).digest('hex');

async function cachedHash(path) {
  try {
    return sha256Hex(await readFile(path));
  } catch {
    return null;
  }
}

/**
 * @param {{ url: string, sha256: string, dest: string, maxBytes: number,
 *   fetchImpl?: typeof fetch, timeoutMs?: number }} opts
 * @returns {Promise<'cached' | 'downloaded'>}
 */
export async function fetchVerified({
  url,
  sha256,
  dest,
  maxBytes,
  fetchImpl = fetch,
  timeoutMs = 120_000,
}) {
  if ((await cachedHash(dest)) === sha256) return 'cached';
  if (new URL(url).protocol !== 'https:')
    throw new Error(`only HTTPS downloads are allowed: ${url}`);
  const res = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status} ${url}`);
  const declared = Number(res.headers.get('content-length') ?? 0);
  if (declared > maxBytes) throw new Error(`${url} is larger than ${maxBytes} bytes`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length > maxBytes) throw new Error(`${url} is larger than ${maxBytes} bytes`);
  const got = sha256Hex(buf);
  if (got !== sha256) {
    throw new Error(`${url}: SHA-256 mismatch, expected ${sha256}, got ${got}`);
  }
  await mkdir(dirname(dest), { recursive: true });
  const tmp = `${dest}.${randomBytes(6).toString('hex')}.part`;
  try {
    await writeFile(tmp, buf, { flag: 'wx' });
    await rename(tmp, dest);
  } finally {
    await rm(tmp, { force: true });
  }
  return 'downloaded';
}
