import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchVerified } from './download.mjs';

const body = Buffer.from('face landmarker bytes');
const sha256 = createHash('sha256').update(body).digest('hex');
const url = 'https://example.test/model.task';

let dir;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'silentspeak-download-'));
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

const serve = (bytes, init) => vi.fn(async () => new Response(bytes, init));

describe('fetchVerified', () => {
  it('writes the file only after its hash matches', async () => {
    const dest = join(dir, 'sub', 'model.task');
    const fetchImpl = serve(body);
    await expect(fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl })).resolves.toBe(
      'downloaded',
    );
    expect(await readFile(dest)).toEqual(body);
    expect(await readdir(join(dir, 'sub'))).toEqual(['model.task']);
  });

  it('writes nothing when the hash does not match', async () => {
    const dest = join(dir, 'model.task');
    const fetchImpl = serve(Buffer.from('tampered'));
    await expect(fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl })).rejects.toThrow(
      /SHA-256 mismatch/,
    );
    expect(await readdir(dir)).toEqual([]);
  });

  it('leaves a good cached copy alone and does not download', async () => {
    const dest = join(dir, 'model.task');
    await writeFile(dest, body);
    const fetchImpl = serve(body);
    await expect(fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl })).resolves.toBe(
      'cached',
    );
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('replaces a corrupt cached copy', async () => {
    const dest = join(dir, 'model.task');
    await writeFile(dest, 'corrupt');
    await fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl: serve(body) });
    expect(await readFile(dest)).toEqual(body);
  });

  it('keeps a corrupt cached copy untouched when the new download is bad too', async () => {
    const dest = join(dir, 'model.task');
    await writeFile(dest, 'corrupt');
    await expect(
      fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl: serve('bad') }),
    ).rejects.toThrow(/mismatch/);
    expect(await readdir(dir)).toEqual(['model.task']);
  });

  it('refuses a response larger than the limit', async () => {
    const dest = join(dir, 'model.task');
    await expect(
      fetchVerified({ url, sha256, dest, maxBytes: 4, fetchImpl: serve(body) }),
    ).rejects.toThrow(/larger than/);
    expect(await readdir(dir)).toEqual([]);
  });

  it('fails on an HTTP error', async () => {
    const dest = join(dir, 'model.task');
    await expect(
      fetchVerified({ url, sha256, dest, maxBytes: 1024, fetchImpl: serve('', { status: 404 }) }),
    ).rejects.toThrow(/404/);
  });

  it('only downloads over HTTPS', async () => {
    const dest = join(dir, 'model.task');
    const fetchImpl = serve(body);
    await expect(
      fetchVerified({ url: 'http://example.test/m', sha256, dest, maxBytes: 1024, fetchImpl }),
    ).rejects.toThrow(/HTTPS/);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
