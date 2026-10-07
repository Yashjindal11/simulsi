import { useMemo } from "react";
import { fmt } from "./format";

const COLORS = ["#4f46e5", "#0891b2", "#d97706", "#db2777", "#16a34a", "#7c3aed", "#dc2626", "#0d9488"];
export const color = (i: number) => COLORS[i % COLORS.length];

const W = 640;
const PAD = { l: 52, r: 12, t: 10, b: 30 };

function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function scale(d0: number, d1: number, r0: number, r1: number) {
  const span = d1 - d0 || 1;
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

function Axes({ x0, x1, y0, y1, h, xLabel, yLabel }: { x0: number; x1: number; y0: number; y1: number; h: number; xLabel?: string; yLabel?: string }) {
  const sx = scale(x0, x1, PAD.l, W - PAD.r);
  const sy = scale(y0, y1, h - PAD.b, PAD.t);
  return (
    <g fontSize={10} fill="#64748b">
      {niceTicks(y0, y1).map((t) => (
        <g key={`y${t}`}>
          <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} stroke="#e2e8f0" />
          <text x={PAD.l - 4} y={sy(t) + 3} textAnchor="end">{fmt(t, 3)}</text>
        </g>
      ))}
      {niceTicks(x0, x1, 6).map((t) => (
        <text key={`x${t}`} x={sx(t)} y={h - PAD.b + 14} textAnchor="middle">{fmt(t, 3)}</text>
      ))}
      <line x1={PAD.l} x2={W - PAD.r} y1={h - PAD.b} y2={h - PAD.b} stroke="#94a3b8" />
      {xLabel && <text x={(PAD.l + W - PAD.r) / 2} y={h - 2} textAnchor="middle">{xLabel}</text>}
      {yLabel && <text x={10} y={PAD.t + 4} textAnchor="start">{yLabel}</text>}
    </g>
  );
}

export interface Series {
  name: string;
  points: [number, number][];
  dashed?: boolean;
}

/** Step chart for piecewise-constant series (queue length, busy units). */
export function StepChart({ series, endTime, height = 220, yLabel }: { series: Series[]; endTime: number; height?: number; yLabel?: string }) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return <p className="text-sm text-slate-500">No series recorded.</p>;
  const x0 = Math.min(...all.map((p) => p[0]));
  const x1 = Math.max(endTime, ...all.map((p) => p[0]));
  const y1 = Math.max(1, ...all.map((p) => p[1])) * 1.05;
  const sx = scale(x0, x1, PAD.l, W - PAD.r);
  const sy = scale(0, y1, height - PAD.b, PAD.t);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
        <Axes x0={x0} x1={x1} y0={0} y1={y1} h={height} xLabel="time" yLabel={yLabel} />
        {series.map((s, i) => {
          let d = "";
          s.points.forEach(([t, v], j) => {
            d += j === 0 ? `M${sx(t)},${sy(v)}` : `H${sx(t)}V${sy(v)}`;
          });
          if (s.points.length) d += `H${sx(x1)}`;
          return <path key={s.name} d={d} fill="none" stroke={color(i)} strokeWidth={1.4} strokeDasharray={s.dashed ? "4 3" : undefined} />;
        })}
      </svg>
      <Legend names={series.map((s) => s.name)} />
    </div>
  );
}

export function Legend({ names }: { names: string[] }) {
  return (
    <div className="mt-1 flex flex-wrap gap-3 text-xs text-slate-600">
      {names.map((n, i) => (
        <span key={n} className="inline-flex items-center gap-1">
          <span className="inline-block h-2 w-3 rounded" style={{ background: color(i) }} />
          {n}
        </span>
      ))}
    </div>
  );
}

/** Histogram with the mean (solid) and CI bounds (dashed). */
export function Histogram({ values, mean, ciLow, ciHigh, height = 220, bins = 20 }: { values: number[]; mean?: number | null; ciLow?: number | null; ciHigh?: number | null; height?: number; bins?: number }) {
  const data = useMemo(() => {
    if (!values.length) return null;
    let lo = Math.min(...values);
    let hi = Math.max(...values);
    if (lo === hi) {
      lo -= 0.5;
      hi += 0.5;
    }
    const k = Math.max(3, Math.min(bins, Math.ceil(Math.sqrt(values.length)) + 2));
    const width = (hi - lo) / k;
    const counts = new Array<number>(k).fill(0);
    values.forEach((v) => (counts[Math.min(k - 1, Math.floor((v - lo) / width))] += 1));
    return { lo, hi, width, counts };
  }, [values, bins]);
  if (!data) return <p className="text-sm text-slate-500">No values.</p>;
  const yMax = Math.max(...data.counts) * 1.1;
  const sx = scale(data.lo, data.hi, PAD.l, W - PAD.r);
  const sy = scale(0, yMax, height - PAD.b, PAD.t);
  const marks: [number | null | undefined, boolean][] = [[mean, false], [ciLow, true], [ciHigh, true]];
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
      <Axes x0={data.lo} x1={data.hi} y0={0} y1={yMax} h={height} yLabel="count" />
      {data.counts.map((c, i) => (
        <rect key={i} x={sx(data.lo + i * data.width) + 1} y={sy(c)} width={Math.max(1, sx(data.lo + (i + 1) * data.width) - sx(data.lo + i * data.width) - 2)} height={height - PAD.b - sy(c)} fill="#818cf8" />
      ))}
      {marks.map(([v, dashed], i) =>
        v === null || v === undefined || !Number.isFinite(v) ? null : (
          <line key={i} x1={sx(v)} x2={sx(v)} y1={PAD.t} y2={height - PAD.b} stroke="#0f172a" strokeDasharray={dashed ? "4 3" : undefined} />
        ),
      )}
    </svg>
  );
}

/** Running mean with a confidence band. */
export function BandChart({ x, y, lo, hi, height = 200, xLabel }: { x: number[]; y: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; height?: number; xLabel?: string }) {
  const vals = [...y, ...lo, ...hi].filter((v): v is number => v !== null && Number.isFinite(v));
  if (!vals.length) return <p className="text-sm text-slate-500">Not enough data.</p>;
  const y0 = Math.min(...vals);
  const y1 = Math.max(...vals);
  const pad = (y1 - y0) * 0.05 || 1;
  const sx = scale(Math.min(...x), Math.max(...x), PAD.l, W - PAD.r);
  const sy = scale(y0 - pad, y1 + pad, height - PAD.b, PAD.t);
  const idx = x.map((_, i) => i).filter((i) => lo[i] !== null && hi[i] !== null);
  const band = idx.length
    ? `M${idx.map((i) => `${sx(x[i])},${sy(hi[i] as number)}`).join("L")}L${idx
        .slice()
        .reverse()
        .map((i) => `${sx(x[i])},${sy(lo[i] as number)}`)
        .join("L")}Z`
    : "";
  const line = x
    .map((xi, i) => (y[i] === null ? null : `${sx(xi)},${sy(y[i] as number)}`))
    .filter(Boolean)
    .join("L");
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
      <Axes x0={Math.min(...x)} x1={Math.max(...x)} y0={y0 - pad} y1={y1 + pad} h={height} xLabel={xLabel} />
      {band && <path d={band} fill="#c7d2fe" opacity={0.6} />}
      <path d={`M${line}`} fill="none" stroke="#4f46e5" strokeWidth={1.6} />
    </svg>
  );
}

export interface Bar {
  label: string;
  value: number | null;
  lo: number | null;
  hi: number | null;
  highlight?: boolean;
}

/** Horizontal bars with error bars, centred on zero (differences). */
export function DiffBars({ bars, unit = "" }: { bars: Bar[]; unit?: string }) {
  const rowH = 26;
  const height = PAD.t + PAD.b + bars.length * rowH;
  const vals = bars.flatMap((b) => [b.value, b.lo, b.hi]).filter((v): v is number => v !== null && Number.isFinite(v));
  const m = Math.max(1e-9, ...vals.map(Math.abs)) * 1.1;
  const left = 220;
  const sx = scale(-m, m, left, W - PAD.r);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
      <g fontSize={10} fill="#64748b">
        {niceTicks(-m, m, 6).map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={PAD.t} y2={height - PAD.b} stroke="#f1f5f9" />
            <text x={sx(t)} y={height - PAD.b + 14} textAnchor="middle">{fmt(t, 3)}{unit}</text>
          </g>
        ))}
      </g>
      <line x1={sx(0)} x2={sx(0)} y1={PAD.t} y2={height - PAD.b} stroke="#334155" />
      {bars.map((b, i) => {
        const y = PAD.t + i * rowH + 4;
        const v = b.value ?? 0;
        return (
          <g key={b.label}>
            <text x={left - 6} y={y + 12} fontSize={11} textAnchor="end" fill="#334155">{b.label}</text>
            <rect x={Math.min(sx(0), sx(v))} y={y} width={Math.abs(sx(v) - sx(0))} height={rowH - 10} fill={b.highlight ? "#4f46e5" : "#a5b4fc"} rx={2} />
            {b.lo !== null && b.hi !== null && (
              <g stroke="#0f172a">
                <line x1={sx(b.lo)} x2={sx(b.hi)} y1={y + 8} y2={y + 8} />
                <line x1={sx(b.lo)} x2={sx(b.lo)} y1={y + 3} y2={y + 13} />
                <line x1={sx(b.hi)} x2={sx(b.hi)} y1={y + 3} y2={y + 13} />
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Events as dots: x = time, one row per event type. */
export function Timeline({ points, endTime }: { points: { t: number; kind: string }[]; endTime: number }) {
  const kinds = Array.from(new Set(points.map((p) => p.kind))).sort();
  const rowH = 22;
  const left = 140;
  const height = PAD.t + PAD.b + Math.max(1, kinds.length) * rowH;
  const x0 = Math.min(0, ...points.map((p) => p.t));
  const sx = scale(x0, endTime, left, W - PAD.r);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
      <g fontSize={10} fill="#64748b">
        {niceTicks(x0, endTime, 6).map((t) => (
          <text key={t} x={sx(t)} y={height - PAD.b + 14} textAnchor="middle">{fmt(t, 3)}</text>
        ))}
      </g>
      {kinds.map((k, i) => (
        <g key={k}>
          <text x={left - 6} y={PAD.t + i * rowH + 14} fontSize={10} textAnchor="end" fill="#334155">{k}</text>
          <line x1={left} x2={W - PAD.r} y1={PAD.t + i * rowH + 10} y2={PAD.t + i * rowH + 10} stroke="#f1f5f9" />
        </g>
      ))}
      {points.map((p, j) => (
        <circle key={j} cx={sx(p.t)} cy={PAD.t + kinds.indexOf(p.kind) * rowH + 10} r={2} fill={color(kinds.indexOf(p.kind))} opacity={0.7} />
      ))}
    </svg>
  );
}

export interface CurvePoint {
  x: number;
  y: number | null;
  lo: number | null;
  hi: number | null;
}

/** Lines of means against a numeric x, with confidence-interval whiskers; one line per group. */
export function LineCI({ curves, height = 260, xLabel, yLabel }: { curves: { name: string; points: CurvePoint[] }[]; height?: number; xLabel?: string; yLabel?: string }) {
  const pts = curves.flatMap((c) => c.points);
  const ys = pts.flatMap((p) => [p.y, p.lo, p.hi]).filter((v): v is number => v !== null && Number.isFinite(v));
  if (!pts.length || !ys.length) return <p className="text-sm text-slate-500">Nothing to plot.</p>;
  const x0 = Math.min(...pts.map((p) => p.x));
  const x1 = Math.max(...pts.map((p) => p.x));
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.08 || 1;
  const sx = scale(x0, x1 === x0 ? x0 + 1 : x1, PAD.l + 8, W - PAD.r - 8);
  const sy = scale(yMin - pad, yMax + pad, height - PAD.b, PAD.t);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full">
        <Axes x0={x0} x1={x1 === x0 ? x0 + 1 : x1} y0={yMin - pad} y1={yMax + pad} h={height} xLabel={xLabel} yLabel={yLabel} />
        {curves.map((c, i) => {
          const sorted = [...c.points].sort((a, b) => a.x - b.x).filter((p) => p.y !== null);
          const d = sorted.map((p, j) => `${j ? "L" : "M"}${sx(p.x)},${sy(p.y as number)}`).join("");
          return (
            <g key={c.name} stroke={color(i)}>
              <path d={d} fill="none" strokeWidth={1.8} />
              {sorted.map((p) => (
                <g key={p.x}>
                  <circle cx={sx(p.x)} cy={sy(p.y as number)} r={3} fill={color(i)} />
                  {p.lo !== null && p.hi !== null && (
                    <path d={`M${sx(p.x) - 4},${sy(p.lo)}h8M${sx(p.x)},${sy(p.lo)}V${sy(p.hi)}M${sx(p.x) - 4},${sy(p.hi)}h8`} strokeWidth={1} />
                  )}
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      {curves.length > 1 && <Legend names={curves.map((c) => c.name)} />}
    </div>
  );
}
