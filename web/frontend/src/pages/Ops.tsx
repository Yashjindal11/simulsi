import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { OpsForecast, GanttBar } from "../types";
import { OpsPlanning } from "./OpsPlanning";

const DAY_START = 5 * 60;

function hhmm(m: number): string {
  const t = Math.round(m);
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
}

function color(bar: GanttBar): string {
  if (bar.status === "cancelled" || bar.p_cancel >= 0.5) return "#94a3b8";
  if (bar.p_on_time >= 0.8) return "#16a34a";
  if (bar.p_on_time >= 0.5) return "#f59e0b";
  return "#dc2626";
}

/** Tails down the side, time across: the planned slot outlined, the predicted (median) slot filled. */
export function Gantt({ bars, now }: { bars: GanttBar[]; now?: number | null }) {
  const tails = useMemo(
    () => [...new Set(bars.map((b) => b.tail))].sort((a, c) => a.localeCompare(c, undefined, { numeric: true })),
    [bars],
  );
  const end = Math.max(24 * 60, ...bars.map((b) => b.eta + 15));
  const start = Math.min(DAY_START, ...bars.map((b) => b.std - 15));
  const width = 1100;
  const left = 56;
  const rowH = 26;
  const x = (t: number) => left + ((t - start) / (end - start)) * (width - left - 8);
  const hours: number[] = [];
  for (let h = Math.ceil(start / 60); h * 60 <= end; h++) hours.push(h * 60);
  return (
    <div className="overflow-x-auto">
      <svg width={width} height={tails.length * rowH + 28} role="img" aria-label="aircraft rotations Gantt chart">
        {hours.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={18} y2={tails.length * rowH + 22} stroke="#e2e8f0" />
            <text x={x(t)} y={12} fontSize={10} textAnchor="middle" fill="#64748b">{hhmm(t)}</text>
          </g>
        ))}
        {tails.map((tail, i) => (
          <text key={tail} x={4} y={30 + i * rowH} fontSize={11} fill="#334155">{tail}</text>
        ))}
        {bars.map((b) => {
          const row = tails.indexOf(b.tail);
          const y = 20 + row * rowH;
          const tip = `${b.flight} ${b.origin}-${b.dest}  STD ${hhmm(b.std)}  predicted ${hhmm(b.etd)}\n` +
            `on time ${(b.p_on_time * 100).toFixed(0)}%  cancel ${(b.p_cancel * 100).toFixed(0)}%` +
            (b.p80 != null ? `  p80 delay ${b.p80.toFixed(0)} min` : "") + `  cause: ${b.cause}  (${b.status})`;
          return (
            <g key={b.flight}>
              <title>{tip}</title>
              <rect x={x(b.std)} y={y} width={Math.max(2, x(b.sta) - x(b.std))} height={rowH - 8} fill="none" stroke="#cbd5e1" strokeDasharray="3 2" />
              <rect x={x(b.etd)} y={y + 3} width={Math.max(2, x(b.eta) - x(b.etd))} height={rowH - 14} rx={3} fill={color(b)} opacity={b.status === "planned" ? 0.9 : 0.5} />
              {x(b.eta) - x(b.etd) > 34 && (
                <text x={x(b.etd) + 3} y={y + rowH / 2 + 1} fontSize={9} fill="white">{b.flight}</text>
              )}
            </g>
          );
        })}
        {now != null && (
          <line x1={x(now)} x2={x(now)} y1={16} y2={tails.length * rowH + 22} stroke="#4f46e5" strokeWidth={2} />
        )}
      </svg>
      <p className="mt-1 text-xs text-slate-500">
        Dashed: planned. Filled: predicted median times, coloured by chance of an on-time arrival (green ≥ 80%, amber ≥ 50%, red below); grey: likely cancelled. Hover a flight for details.
      </p>
    </div>
  );
}

function Kpi({ label, value, range }: { label: string; value: string; range?: string }) {
  return (
    <div className="card">
      <p className="label">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {range && <p className="text-xs text-slate-500">80% range {range}</p>}
    </div>
  );
}

export function OpsView() {
  const [schedule, setSchedule] = useState("");
  const [connections, setConnections] = useState("");
  const [ops, setOps] = useState("");
  const [weather, setWeather] = useState("");
  const [now, setNow] = useState("");
  const [status, setStatus] = useState("");
  const [actions, setActions] = useState("");
  const [reps, setReps] = useState(200);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fc, setFc] = useState<OpsForecast | null>(null);
  const [baseline, setBaseline] = useState<OpsForecast | null>(null);
  const [example, setExample] = useState<{ status_csv: string; status_now: string } | null>(null);

  useEffect(() => {
    api.aviationExample().then((ex) => {
      setSchedule(ex.schedule_csv);
      setConnections(ex.connections_csv);
      setOps(ex.ops_yaml);
      setExample(ex);
    }).catch((e: Error) => setError(e.message));
  }, []);

  const body = (withActions: boolean) => ({
    schedule_csv: schedule,
    connections_csv: connections,
    ops_yaml: ops,
    weather: weather.split("\n"),
    now: now.trim() || undefined,
    status_csv: status,
    actions: withActions ? actions.split("\n") : [],
    replications: reps,
  });

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const hasActions = actions.trim().length > 0;
      const [main, base] = await Promise.all([
        api.aviationForecast(body(true)),
        hasActions ? api.aviationForecast(body(false)) : Promise.resolve(null),
      ]);
      setFc(main);
      setBaseline(base);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const file = (set: (s: string) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) f.text().then(set);
  };

  const s = fc?.summary;
  const b = baseline?.summary;
  const pctOf = (v?: number | null) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);
  const num = (v?: number | null, d = 1) => (v == null ? "-" : v.toLocaleString(undefined, { maximumFractionDigits: d }));
  const nowMin = now.trim() ? (() => { const [h, m] = now.split(":").map(Number); return h * 60 + (m || 0); })() : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Airline operations twin</h1>
        <p className="text-sm text-slate-600">
          Forecast a day of flying: per-flight on-time and cancellation chances, fragile rotations, connections at risk.
          Start from the plan, or from the live state at a time of day; try controller actions against the same disturbances.
        </p>
      </div>
      <section className="card grid gap-3 md:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className="label">Schedule CSV (flight, tail, origin, dest, std, sta, pax, crew)</span>
          <textarea className="input h-36 font-mono text-xs" value={schedule} onChange={(e) => setSchedule(e.target.value)} />
          <input type="file" accept=".csv" onChange={file(setSchedule)} className="text-xs" />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Connections CSV (inbound, outbound, pax)</span>
          <textarea className="input h-36 font-mono text-xs" value={connections} onChange={(e) => setConnections(e.target.value)} />
          <input type="file" accept=".csv" onChange={file(setConnections)} className="text-xs" />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Operating rules (YAML)</span>
          <textarea className="input h-44 font-mono text-xs" value={ops} onChange={(e) => setOps(e.target.value)} />
        </label>
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className="label">Extra weather, one per line (e.g. HUB 15:00-18:00 0.4 p=0.6)</span>
            <textarea className="input h-14 font-mono text-xs" value={weather} onChange={(e) => setWeather(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="label">Actions to test, one per line (cancel F1 F2 · retime F1 30 · swap T1 T2 12:00)</span>
            <textarea className="input h-14 font-mono text-xs" value={actions} onChange={(e) => setActions(e.target.value)} />
          </label>
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="label">Now (HH:MM, optional)</span>
              <input className="input w-28" value={now} placeholder="12:00" onChange={(e) => setNow(e.target.value)} />
            </label>
            <label className="flex flex-col gap-1">
              <span className="label">Live status CSV {status ? "(loaded)" : ""}</span>
              <input type="file" accept=".csv" onChange={file(setStatus)} className="text-xs" />
            </label>
            {example && (
              <button className="btn-ghost" onClick={() => { setStatus(example.status_csv); setNow(example.status_now); }}>
                Example live day at {example.status_now}
              </button>
            )}
            {(status || now) && (
              <button className="btn-ghost" onClick={() => { setStatus(""); setNow(""); }}>Back to the plan</button>
            )}
            <label className="flex flex-col gap-1">
              <span className="label">Simulated days</span>
              <input className="input w-24" type="number" min={10} max={1000} value={reps} onChange={(e) => setReps(Number(e.target.value))} />
            </label>
            <button className="btn" disabled={busy || !schedule.trim()} onClick={run}>{busy ? "Simulating…" : "Forecast"}</button>
          </div>
        </div>
      </section>
      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      <OpsPlanning body={() => body(false)} />
      {fc && s && (
        <>
          <div className="grid gap-3 md:grid-cols-5">
            <Kpi label="On-time arrivals" value={pctOf(s.otp?.mean)} range={`${pctOf(s.otp?.p10)} – ${pctOf(s.otp?.p90)}`} />
            <Kpi label="Cancellations" value={num(s.cancelled?.mean)} range={`${num(s.cancelled?.p10)} – ${num(s.cancelled?.p90)}`} />
            <Kpi label="Misconnected pax" value={num(s.misconnected_pax?.mean, 0)} range={`${num(s.misconnected_pax?.p10, 0)} – ${num(s.misconnected_pax?.p90, 0)}`} />
            <Kpi label="Mean arrival delay" value={`${num(s["arrival_delay.mean"]?.mean)} min`} />
            <Kpi label="Disruption cost" value={num(s.cost?.mean, 0)} range={`${num(s.cost?.p10, 0)} – ${num(s.cost?.p90, 0)}`} />
          </div>
          {b && (
            <div className="card text-sm">
              <p className="font-semibold">With your actions vs as planned (same disturbances)</p>
              <p>
                OTP {pctOf(b.otp?.mean)} → {pctOf(s.otp?.mean)} · cancellations {num(b.cancelled?.mean)} → {num(s.cancelled?.mean)} ·
                misconnected pax {num(b.misconnected_pax?.mean, 0)} → {num(s.misconnected_pax?.mean, 0)} · cost {num(b.cost?.mean, 0)} → {num(s.cost?.mean, 0)}
              </p>
            </div>
          )}
          <section className="card">
            <h3 className="mb-2 font-semibold">Aircraft rotations</h3>
            <Gantt bars={fc.gantt} now={nowMin} />
          </section>
          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card">
              <h3 className="mb-2 font-semibold">Alerts ({fc.alerts.length})</h3>
              {fc.alerts.length === 0 && <p className="text-sm text-slate-500">Nothing above the thresholds.</p>}
              <ul className="space-y-1 text-sm">
                {fc.alerts.slice(0, 30).map((a, i) => (
                  <li key={i} className={a.level === "high" ? "text-red-700" : a.level === "medium" ? "text-amber-700" : "text-slate-600"}>
                    <b>{a.kind}</b>: {a.message}
                  </li>
                ))}
              </ul>
            </section>
            <section className="card overflow-x-auto">
              <h3 className="mb-2 font-semibold">Most fragile rotations</h3>
              <table className="table">
                <thead><tr><th>tail</th><th>legs</th><th>exp. delay min</th><th>P(any cancel)</th><th>worst leg</th><th>first at risk</th></tr></thead>
                <tbody>
                  {fc.rotations.slice(0, 8).map((r) => (
                    <tr key={r.tail}><td>{r.tail}</td><td>{r.legs}</td><td>{r.expected_delay_minutes}</td><td>{pctOf(r.p_any_cancel)}</td><td>{r.worst_leg}</td><td>{r.first_at_risk || "-"}</td></tr>
                  ))}
                </tbody>
              </table>
            </section>
          </div>
          <section className="card overflow-x-auto">
            <h3 className="mb-2 font-semibold">Flights most at risk</h3>
            <table className="table">
              <thead><tr><th>flight</th><th>route</th><th>STD</th><th>P(on time)</th><th>P(cancel)</th><th>delay p50 / p80 / p95</th><th>cause</th><th>misconnect pax</th><th>status</th></tr></thead>
              <tbody>
                {[...fc.flights].filter((f) => f.status === "planned").sort((a, c) => (a.p_on_time ?? 1) - (c.p_on_time ?? 1)).slice(0, 20).map((f) => (
                  <tr key={f.id}>
                    <td>{f.id}</td><td>{f.origin}-{f.dest}</td><td>{f.std}</td><td>{pctOf(f.p_on_time)}</td><td>{pctOf(f.p_cancel)}</td>
                    <td>{[f.dep_delay_p50, f.dep_delay_p80, f.dep_delay_p95].map((v) => (v == null ? "-" : v.toFixed(0))).join(" / ")}</td>
                    <td>{f.main_cause}</td><td>{num(f.misconnect_pax, 1)}</td><td>{f.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
