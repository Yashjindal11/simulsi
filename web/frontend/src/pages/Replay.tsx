import { useEffect, useMemo, useState } from "react";
import { fmt } from "../format";
import type { Trace } from "../types";

/** Value of a step series at time t (the last change at or before t). */
function valueAt(points: [number, number][] | undefined, t: number): number | null {
  if (!points || !points.length || points[0][0] > t) return null;
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (points[mid][0] <= t) lo = mid;
    else hi = mid - 1;
  }
  return points[lo][1];
}

const MAX_UNITS = 40;

/** Resources, queues, containers and other recorded levels of one run at time t. */
function ReplayState({ trace, t, label }: { trace: Trace; t: number; label?: string }) {
  const keys = Object.keys(trace.series);
  const resources = keys.filter((k) => k.startsWith("resource.") && k.endsWith(".busy")).map((k) => k.slice(9, -5));
  const queues = keys.filter((k) => k.startsWith("queue.") && k.endsWith(".length"));
  const containers = keys.filter((k) => k.startsWith("container.") && k.endsWith(".level"));
  const gauges = keys.filter((k) => !k.startsWith("resource.") && !k.startsWith("queue.") && !k.startsWith("container."));
  const recent = trace.log.filter((r) => r.timestamp <= t).slice(-10).reverse();
  return (
    <div className="space-y-3">
      {label && <p className="text-sm font-bold text-indigo-700">{label}</p>}
      {resources.map((name) => {
        const busy = valueAt(trace.series[`resource.${name}.busy`], t) ?? 0;
        const cap = valueAt(trace.series[`resource.${name}.capacity`], t) ?? busy;
        const queue = valueAt(trace.series[`resource.${name}.queue_length`], t) ?? 0;
        const units = Math.min(MAX_UNITS, Math.max(cap, busy));
        return (
          <div key={name}>
            <p className="text-sm font-semibold">{name} <span className="font-normal text-slate-500">{fmt(busy)} busy of {fmt(cap)} · {fmt(queue)} waiting</span></p>
            <div className="mt-1 flex flex-wrap items-center gap-1">
              {Array.from({ length: units }, (_, i) => (
                <span key={i} className={`inline-block h-5 w-5 rounded ${i < busy ? "bg-indigo-600" : "border border-slate-300 bg-white"}`} title={i < busy ? "busy" : "idle"} />
              ))}
              <span className="mx-2 text-slate-400">|</span>
              {Array.from({ length: Math.min(MAX_UNITS, queue) }, (_, i) => <span key={i} className="inline-block h-3 w-3 rounded-full bg-amber-500" />)}
              {queue > MAX_UNITS && <span className="text-xs text-slate-500">+{fmt(queue - MAX_UNITS)}</span>}
            </div>
          </div>
        );
      })}
      {queues.map((k) => {
        const n = valueAt(trace.series[k], t) ?? 0;
        return (
          <div key={k}>
            <p className="text-sm font-semibold">{k.slice(6, -7)} <span className="font-normal text-slate-500">{fmt(n)} items</span></p>
            <div className="mt-1 flex flex-wrap gap-1">{Array.from({ length: Math.min(MAX_UNITS, n) }, (_, i) => <span key={i} className="inline-block h-3 w-3 rounded-sm bg-cyan-600" />)}</div>
          </div>
        );
      })}
      {[...containers, ...gauges].map((k) => {
        const v = valueAt(trace.series[k], t) ?? 0;
        const max = Math.max(1e-9, ...trace.series[k].map((p) => p[1]));
        return (
          <div key={k}>
            <p className="text-sm font-semibold">{k.replace(/^container\./, "").replace(/\.level$/, "")} <span className="font-normal text-slate-500">{fmt(v)}</span></p>
            <div className="mt-1 h-2.5 w-full overflow-hidden rounded bg-slate-200"><div className="h-full bg-emerald-600" style={{ width: `${Math.max(0, (100 * v) / max)}%` }} /></div>
          </div>
        );
      })}
      <div>
        <p className="mb-1 text-sm font-semibold">Latest events</p>
        <table className="table">
          <tbody>
            {recent.map((r, i) => (
              <tr key={i}><td className="w-20">{fmt(r.timestamp)}</td><td className="font-mono text-xs">{r.event_type}</td><td>{r.entity ?? ""}</td><td>{r.resource ?? ""}</td></tr>
            ))}
          </tbody>
        </table>
        {!recent.length && <p className="text-sm text-slate-500">No events yet.</p>}
      </div>
    </div>
  );
}

/** Step through one or more traced runs on a shared clock. */
export function Replay({ runs }: { runs: { label?: string; trace: Trace }[] }) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const end = useMemo(() => Math.max(...runs.map((r) => r.trace.end_time)), [runs]);

  useEffect(() => {
    if (!playing) return;
    // A timer rather than requestAnimationFrame, so playback also advances in background tabs.
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const dt = ((now - last) / 1000) * speed * (end / 30); // the whole run in 30 s at 1x
      last = now;
      setT((prev) => {
        const next = Math.min(end, prev + dt);
        if (next >= end) setPlaying(false);
        return next;
      });
    }, 50);
    return () => window.clearInterval(timer);
  }, [playing, speed, end]);

  return (
    <section className="card space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="font-semibold">Replay{runs.length > 1 ? " side by side" : ""}</h3>
        <button className="btn" onClick={() => { if (t >= end) setT(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play"}</button>
        <select className="input w-24" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          {[0.25, 0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}x</option>)}
        </select>
        <input type="range" className="min-w-64 flex-1" min={0} max={end} step={end / 1000 || 1} value={t} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
        <span className="w-28 text-right font-mono text-sm">t = {fmt(t)}</span>
      </div>
      <div className={`grid gap-6 ${runs.length > 1 ? "lg:grid-cols-2" : ""}`}>
        {runs.map((r, i) => <ReplayState key={i} trace={r.trace} t={t} label={r.label} />)}
      </div>
    </section>
  );
}
