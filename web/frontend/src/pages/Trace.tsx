import { useEffect, useState } from "react";
import { api } from "../api";
import { StepChart, Timeline } from "../charts";
import { fmt } from "../format";
import type { Trace } from "../types";
import { Loading, Select } from "./common";
import { Replay } from "./Replay";
import { ParamForm, toParams, useModels } from "./Run";

/** Group series keys for charts: queues, busy units, and each model-recorded family (people.*, inventory.*, ...). */
function seriesGroups(trace: Trace): { title: string; keys: string[]; dashed?: string[] }[] {
  const keys = Object.keys(trace.series);
  const out: { title: string; keys: string[]; dashed?: string[] }[] = [];
  const queues = keys.filter((k) => k.endsWith(".queue_length") || (k.startsWith("queue.") && k.endsWith(".length")));
  if (queues.length) out.push({ title: "Queue lengths", keys: queues });
  const busy = keys.filter((k) => k.endsWith(".busy"));
  if (busy.length) {
    out.push({
      title: "Busy vs available units",
      keys: busy,
      dashed: busy.map((k) => k.replace(/\.busy$/, ".capacity")).filter((k) => trace.series[k]),
    });
  }
  const containers = keys.filter((k) => k.startsWith("container."));
  if (containers.length) out.push({ title: "Container levels", keys: containers });
  const families = new Map<string, string[]>();
  keys
    .filter((k) => !k.startsWith("resource.") && !k.startsWith("queue.") && !k.startsWith("container."))
    .forEach((k) => {
      const fam = k.includes(".") ? k.slice(0, k.indexOf(".")) : k;
      families.set(fam, [...(families.get(fam) ?? []), k]);
    });
  families.forEach((ks, fam) => out.push({ title: `Recorded by the model: ${fam}`, keys: ks }));
  return out;
}

function TraceCharts({ trace, label }: { trace: Trace; label?: string }) {
  return (
    <div className="space-y-4">
      {label && <p className="text-sm font-bold text-indigo-700">{label}</p>}
      <p className="text-sm text-slate-600">
        seed {trace.seed} · {trace.events.toLocaleString()} events · ended at t = {fmt(trace.end_time)}
        {trace.log_dropped > 0 && ` · log capped (${trace.log_dropped.toLocaleString()} records dropped)`}
      </p>
      {trace.warnings.map((w) => <div key={w} className="rounded-lg bg-amber-50 p-2 text-sm text-amber-800">{w}</div>)}
      {seriesGroups(trace).map((g) => (
        <section key={g.title} className="card">
          <h3 className="mb-2 font-semibold">{g.title}</h3>
          <StepChart
            series={[...g.keys.map((k) => ({ name: k, points: trace.series[k] })), ...(g.dashed ?? []).map((k) => ({ name: k, points: trace.series[k], dashed: true }))]}
            endTime={trace.end_time}
          />
        </section>
      ))}
      <section className="card">
        <h3 className="mb-2 font-semibold">Event timeline</h3>
        <Timeline points={trace.log.slice(0, 5000).map((r) => ({ t: r.timestamp, kind: r.event_type }))} endTime={trace.end_time} />
      </section>
    </div>
  );
}

export function TraceView() {
  const { models, error } = useModels();
  const [modelId, setModelId] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [compare, setCompare] = useState(false);
  const [valuesB, setValuesB] = useState<Record<string, string>>({});
  const [seed, setSeed] = useState("1");
  const [duration, setDuration] = useState("200");
  const [traces, setTraces] = useState<Trace[]>([]);
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
      const body = (v: Record<string, string>) => ({
        model: model.id, parameters: toParams(model, v), seed: Number(seed),
        duration: duration.trim() ? Number(duration) : undefined,
      });
      const sets = compare ? [values, valuesB] : [values];
      setTraces(await Promise.all(sets.map((v) => api.trace(body(v)))));
    } catch (e) {
      setRunError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const loadPreset = (name: string, target: "A" | "B") => {
    if (!model) return;
    const v = Object.fromEntries(Object.entries(model.presets[name] ?? {}).map(([k, x]) => [k, String(x)]));
    if (target === "A") setValues(v);
    else setValuesB(v);
  };

  if (!models) return <Loading error={error} />;
  const presetNames = model ? Object.keys(model.presets ?? {}) : [];
  const labels = compare ? ["A", "B"] : [undefined];
  const first = traces[0];
  const log = first ? first.log.filter((r) => !filter || JSON.stringify(r).toLowerCase().includes(filter.toLowerCase())) : [];

  const presetPicker = (target: "A" | "B") =>
    presetNames.length > 0 && (
      <label className="flex items-center gap-2 text-sm">
        <span className="label">Preset</span>
        <select className="input w-56" defaultValue="" onChange={(e) => e.target.value && loadPreset(e.target.value, target)}>
          <option value="">(model defaults)</option>
          {presetNames.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </label>
    );

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-bold">Trace and replay</h1>
        <p className="text-sm text-slate-600">
          Runs one replication with the event log on, then replays it. Turn on comparison to run two parameter sets with
          the same seed and watch them side by side.
        </p>
      </header>
      <section className="card space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <Select label="Model" value={modelId} options={models.map((m) => m.id)} onChange={(v) => { setModelId(v); setValues({}); setValuesB({}); setTraces([]); }} />
          <label className="flex flex-col gap-1"><span className="label">Seed</span><input className="input w-24" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
          <label className="flex flex-col gap-1"><span className="label">Duration</span><input className="input w-28" value={duration} onChange={(e) => setDuration(e.target.value)} /></label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} /> Compare two parameter sets</label>
          <button className="btn" disabled={busy || !model} onClick={() => void run()}>{busy ? "Running…" : compare ? "Run both" : "Run with trace"}</button>
        </div>
        {model && (
          <div className={`grid gap-4 ${compare ? "xl:grid-cols-2" : ""}`}>
            <div className="space-y-2">
              <div className="flex items-center gap-3">{compare && <span className="font-bold text-indigo-700">A</span>}{presetPicker("A")}</div>
              <ParamForm model={model} values={values} onChange={setValues} />
            </div>
            {compare && (
              <div className="space-y-2">
                <div className="flex items-center gap-3"><span className="font-bold text-indigo-700">B</span>{presetPicker("B")}</div>
                <ParamForm model={model} values={valuesB} onChange={setValuesB} />
              </div>
            )}
          </div>
        )}
      </section>
      {runError && <Loading error={runError} />}
      {traces.length > 0 && (
        <>
          <Replay runs={traces.map((trace, i) => ({ trace, label: labels[i] }))} />
          <div className={`grid gap-6 ${traces.length > 1 ? "xl:grid-cols-2" : ""}`}>
            {traces.map((trace, i) => <TraceCharts key={i} trace={trace} label={labels[i]} />)}
          </div>
          {traces.length > 1 && (
            <section className="card overflow-x-auto">
              <h3 className="mb-2 font-semibold">Metrics: A vs B</h3>
              <table className="table">
                <thead><tr><th>Metric</th><th>A</th><th>B</th><th>B − A</th></tr></thead>
                <tbody>
                  {Object.keys(traces[0].metrics).sort().map((k) => {
                    const a = traces[0].metrics[k];
                    const b = traces[1].metrics[k];
                    const d = a !== null && b !== null && b !== undefined ? b - a : null;
                    return (
                      <tr key={k} className={d ? "" : "text-slate-400"}>
                        <td className="font-mono text-xs">{k}</td><td>{fmt(a)}</td><td>{fmt(b)}</td><td>{fmt(d)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          )}
          <section className="card overflow-x-auto">
            <div className="mb-2 flex items-center gap-3">
              <h3 className="font-semibold">Event log{traces.length > 1 ? " (A)" : ""}</h3>
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
          {traces.length === 1 && (
            <section className="card overflow-x-auto">
              <h3 className="mb-2 font-semibold">Metrics of this run</h3>
              <table className="table">
                <tbody>
                  {Object.entries(first.metrics).sort().map(([k, v]) => (
                    <tr key={k}><td className="font-mono text-xs">{k}</td><td>{fmt(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  );
}
