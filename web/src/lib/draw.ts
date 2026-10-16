import {
  INNER_LOWER_CONTOUR,
  INNER_UPPER_CONTOUR,
  OUTER_LOWER,
  OUTER_UPPER,
  storedRow,
} from '../ml/landmarks';

export type PointFn = (meshIndex: number) => [number, number] | null;

/** Stored points (x, y, z triples) mapped through `scale` into canvas pixels. */
export function pointsMapper(
  points: Float32Array,
  scale: (x: number, y: number) => [number, number],
): PointFn {
  return (idx) => {
    const r = storedRow(idx);
    const x = points[3 * r];
    const y = points[3 * r + 1];
    if (x === undefined || y === undefined || Number.isNaN(x) || Number.isNaN(y)) return null;
    return scale(x, y);
  };
}

function ring(
  ctx: CanvasRenderingContext2D,
  at: PointFn,
  upper: readonly number[],
  lower: readonly number[],
) {
  const pts: [number, number][] = [];
  for (const idx of [...upper, ...[...lower].reverse().slice(1, -1)]) {
    const p = at(idx);
    if (!p) return false;
    pts.push(p);
  }
  ctx.beginPath();
  pts.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  return true;
}

/** Lip contour overlay: outer lips filled softly, inner lips outlined. */
export function drawLips(ctx: CanvasRenderingContext2D, at: PointFn, lineWidth = 2) {
  ctx.save();
  ctx.lineJoin = 'round';
  ctx.lineWidth = lineWidth;
  if (ring(ctx, at, OUTER_UPPER, OUTER_LOWER)) {
    ctx.fillStyle = 'rgba(242, 122, 160, 0.18)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 140, 175, 0.95)';
    ctx.stroke();
  }
  if (ring(ctx, at, INNER_UPPER_CONTOUR, INNER_LOWER_CONTOUR)) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.stroke();
  }
  ctx.fillStyle = '#ffffff';
  for (const idx of [...OUTER_UPPER, ...OUTER_LOWER]) {
    const p = at(idx);
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], lineWidth * 0.9, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
