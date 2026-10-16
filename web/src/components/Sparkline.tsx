import { useEffect, useRef } from 'react';

const WIDTH = 480;
const HEIGHT = 56;

/**
 * Canvas sparkline of the live mouth-opening signal. `values` is read every
 * animation frame from a ref the tracker loop writes to, so updates do not
 * re-render React.
 */
export function Sparkline({ values }: { values: { current: number[] } }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    let raf = 0;
    let drawn = -1;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    const line = getComputedStyle(document.documentElement).getPropertyValue('--line').trim();
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const xs = values.current;
      const key = xs.length > 0 ? xs.length * 1000 + (xs[xs.length - 1] ?? 0) : 0;
      if (key === drawn) return;
      drawn = key;
      ctx.clearRect(0, 0, WIDTH, HEIGHT);
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, HEIGHT - 4);
      ctx.lineTo(WIDTH, HEIGHT - 4);
      ctx.stroke();
      if (xs.length < 2) return;
      const max = Math.max(0.35, ...xs);
      const step = WIDTH / (Math.max(xs.length, 120) - 1);
      ctx.strokeStyle = accent || '#a3204f';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      xs.forEach((v, i) => {
        const x = WIDTH - (xs.length - 1 - i) * step;
        const y = HEIGHT - 4 - (v / max) * (HEIGHT - 10);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [values]);

  return (
    <canvas
      ref={canvasRef}
      width={WIDTH}
      height={HEIGHT}
      className="h-14 w-full"
      role="img"
      aria-label="Mouth opening over the last few seconds"
    />
  );
}
