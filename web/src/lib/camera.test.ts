import { describe, expect, it } from 'vitest';
import { cameraProblem, classifyCameraError } from './camera';

const err = (name: string) => Object.assign(new Error('x'), { name });

describe('classifyCameraError', () => {
  it.each([
    ['NotAllowedError', 'denied'],
    ['NotFoundError', 'not-found'],
    ['OverconstrainedError', 'not-found'],
    ['NotReadableError', 'busy'],
    ['AbortError', 'busy'],
    ['TypeError', 'unknown'],
  ])('%s -> %s', (name, kind) => {
    expect(classifyCameraError(err(name))).toBe(kind);
  });

  it('flags insecure pages', () => {
    expect(cameraProblem(new DOMException('x', 'InsecureContextError'))).toBe('insecure');
    expect(cameraProblem('weird')).toBe('unknown');
  });
});
