import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { StepChart, Timeline } from "../charts";
import { fmt } from "../format";
import type { Trace } from "../types";
import { Loading, Select } from "./common";
import { Replay } from "./Replay";
import { ParamForm, toParams, useModels } from "./Run";

export function TraceView() {
  const { models, error } = useModels();
  const [modelId, setModelId] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [seed, setSeed] = useState("1");
  const [duration, setDuration] = useState("200");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (models && !modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  const model = models?.find((m) => m.id === modelId);

  const run = async () => {
    if (!model) return;
    setBusy(true);
    setRunError(null);
    try {
      setTrace(await api.trace({ model: model.id, parameters: toParams(model, values), seed: Number(seed), duration: duration.trim() ? Number(duration) : undefined }));
    } catch (e) {
      setRunError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const groups = useMemo(() => {
    if (!trace) return { queues: [], busy: [] as string[] };
    const keys = Object.keys(trace.series);
    return {
      queues: keys.filter((k) => k.endsWith(".queue_length") || (k.startsWith("queue.") && k.endsWith(".length"))),
      busy: keys.filter((k) => k.endsWith(".busy")),
    };
  }, [trace]);

  if (!models) return <Loading error={error} />;

  const log = trace ? trace.log.filter((r) => !filter || JSON.stringify(r).toLowerCase().includes(filter.toLowerCase())) : [];

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-bold">Trace a single run</h1>
        <p className="text-sm text-slate-600">Runs one replication with the event log on and shows what happened over time.</p>
      </header>
      <section className="card space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <Select label="Model" value={modelId} options={models.map((m) => m.id)} onChange={(v) => { setModelId(v); setValues({}); }} />
          <label className="flex flex-col gap-1"><span className="label">Seed</span><input className="input w-24" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
          <label className="flex flex-col gap-1"><span className="label">Duration</span><input className="input w-28" value={duration} onChange={(e) => setDuration(e.target.value)} /></label>
          <button className="btn" disabled={busy || !model} onClick={() => void run()}>{busy ? "Running…" : "Run with trace"}</button>
        </div>
        {model && <ParamForm model={model} values={values} onChange={setValues} />}
      </section>
      {runError && <Loading error={runError} />}
      {trace && (
        <>
          <p className="text-sm text-slate-600">
            seed {trace.seed} · {trace.events.toLocaleString()} events · ended at t = {fmt(trace.end_time)}
            {trace.log_dropped > 0 && ` · log capped (${trace.log_dropped.toLocaleString()} records dropped)`}
          </p>
          {trace.warnings.map((w) => <div key={w} className="rounded-lg bg-amber-50 p-2 text-sm text-amber-800">{w}</div>)}
          <Replay trace={trace} />
          <div className="grid gap-4 xl:grid-cols-2">
            <section className="card">
              <h3 className="mb-2 font-semibold">Queue lengths</h3>
              <StepChart series={groups.queues.map((k) => ({ name: k, points: trace.series[k] }))} endTime={trace.end_time} yLabel="waiting" />
            </section>
            <section className="card">
              <h3 className="mb-2 font-semibold">Resource utilization (busy vs available units)</h3>
              <StepChart
                series={groups.busy.flatMap((k) => {
                  const cap = k.replace(/\.busy$/, ".capacity");
                  return [{ name: k, points: trace.series[k] }, ...(trace.series[cap] ? [{ name: cap, points: trace.series[cap], dashed: true }] : [])];
                })}
                endTime={trace.end_time}
                yLabel="units"
              />
            </section>
          </div>
          <section className="card">
            <h3 className="mb-2 font-semibold">Event timeline</h3>
            <Timeline points={trace.log.slice(0, 5000).map((r) => ({ t: r.timestamp, kind: r.event_type }))} endTime={trace.end_time} />
          </section>
          <section className="card overflow-x-auto">
            <div className="mb-2 flex items-center gap-3">
              <h3 className="font-semibold">Event log</h3>
              <input className="input ml-auto w-64" placeholder="filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
            </div>
            <table className="table">
              <thead><tr><th>Time</th><th>Event</th><th>Entity</th><th>Resource</th><th>Old → new</th><th>Metadata</th></tr></thead>
              <tbody>
                {log.slice(0, 300).map((r, i) => (
                  <tr key={i}>
                    <td>{fmt(r.timestamp)}</td>
                    <td className="font-mono text-xs">{r.event_type}</td>
                    <td>{r.entity ?? ""}</td>
                    <td>{r.resource ?? ""}</td>
                    <td>{r.old_state || r.new_state ? `${r.old_state ?? ""} → ${r.new_state ?? ""}` : ""}</td>
                    <td className="max-w-md truncate text-xs text-slate-500">{Object.keys(r.metadata).length ? JSON.stringify(r.metadata) : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {log.length > 300 && <p className="mt-2 text-xs text-slate-500">Showing 300 of {log.length.toLocaleString()} records.</p>}
          </section>
          <section className="card overflow-x-auto">
            <h3 className="mb-2 font-semibold">Metrics of this run</h3>
            <table className="table">
              <tbody>
                {Object.entries(trace.metrics).sort().map(([k, v]) => (
                  <tr key={k}><td className="font-mono text-xs">{k}</td><td>{fmt(v)}</td></tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
