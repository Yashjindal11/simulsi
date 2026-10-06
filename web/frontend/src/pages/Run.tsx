import { useEffect, useState } from "react";
import { api } from "../api";
import { fmt } from "../format";
import type { ModelInfo, ParamInfo } from "../types";
import { Loading, Select } from "./common";

/** Parse a form value: numbers and booleans become JSON values, distribution specs may be JSON objects. */
export function parseValue(raw: string, p: ParamInfo | undefined): unknown {
  const s = raw.trim();
  if (p?.kind === "str") return raw;
  if (s === "true" || s === "false") return s === "true";
  if (s !== "" && !Number.isNaN(Number(s))) return Number(s);
  if (s.startsWith("{")) {
    try {
      return JSON.parse(s);
    } catch {
      return raw;
    }
  }
  return raw;
}

export function useModels() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.models().then(setModels).catch((e: Error) => setError(e.message));
  }, []);
  return { models, error };
}

export function ParamForm({ model, values, onChange }: { model: ModelInfo; values: Record<string, string>; onChange: (v: Record<string, string>) => void }) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      {model.parameters.map((p) => (
        <label key={p.name} className="flex flex-col gap-1" title={p.description}>
          <span className="label">
            {p.name} {p.unit && <span className="normal-case text-slate-400">({p.unit})</span>}
          </span>
          <input
            className="input"
            value={values[p.name] ?? ""}
            placeholder={fmt(p.default)}
            onChange={(e) => onChange({ ...values, [p.name]: e.target.value })}
          />
          <span className="text-xs text-slate-400">
            {p.kind}
            {p.low !== null ? ` ≥ ${p.low}` : ""}
            {p.high !== null ? ` ≤ ${p.high}` : ""}
          </span>
        </label>
      ))}
    </div>
  );
}

export function toParams(model: ModelInfo, values: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(values)) {
    if (v.trim() === "") continue;
    out[k] = parseValue(v, model.parameters.find((p) => p.name === k));
  }
  return out;
}

interface ScenarioDraft {
  name: string;
  values: Record<string, string>;
}

export function RunView({ onDone }: { onDone: (id: string) => void }) {
  const { models, error } = useModels();
  const [modelId, setModelId] = useState("");
  const [scenarios, setScenarios] = useState<ScenarioDraft[]>([{ name: "baseline", values: {} }]);
  const [replications, setReplications] = useState("10");
  const [seed, setSeed] = useState("42");
  const [duration, setDuration] = useState("");
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [progress, setProgress] = useState<[number, number] | null>(null);

  useEffect(() => {
    if (models && !modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  if (!models) return <Loading error={error} />;
  const model = models.find((m) => m.id === modelId);

  const submit = async () => {
    if (!model) return;
    setBusy(true);
    setRunError(null);
    try {
      let job = await api.run({
        model: model.id,
        replications: Number(replications),
        seed: Number(seed),
        duration: duration.trim() ? Number(duration) : undefined,
        scenarios: scenarios.map((s) => ({ name: s.name, parameters: toParams(model, s.values) })),
      });
      setProgress([job.done, job.total]);
      while (job.status === "running") {
        await new Promise((r) => setTimeout(r, 400));
        job = await api.job(job.id);
        setProgress([job.done, job.total]);
      }
      if (job.status === "error") throw new Error(job.error ?? "run failed");
      if (job.result_id) onDone(job.result_id);
    } catch (e) {
      setRunError((e as Error).message);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <div className="max-w-5xl space-y-4">
      <header>
        <h1 className="text-xl font-bold">Run an experiment</h1>
        <p className="text-sm text-slate-600">
          Built-in models and models started with <code>simulsi ui --model file.py</code>. Each scenario overrides the
          model defaults; empty fields keep the default. All scenarios share replication seeds (common random numbers).
        </p>
      </header>
      <section className="card space-y-3">
        <Select label="Model" value={modelId} options={models.map((m) => m.id)} onChange={setModelId} />
        {model && (
          <p className="text-sm text-slate-600">
            {model.description} · duration {fmt(model.duration)} · warm-up {fmt(model.warmup)} · version {model.version}
          </p>
        )}
      </section>
      {model &&
        scenarios.map((s, i) => (
          <section key={i} className="card space-y-3">
            <div className="flex items-center gap-2">
              <input className="input font-semibold" value={s.name} onChange={(e) => setScenarios(scenarios.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} />
              {i === 0 && <span className="text-xs text-slate-500">baseline for comparisons</span>}
              <span className="flex-1" />
              {i > 0 && (
                <button className="btn-ghost" onClick={() => setScenarios(scenarios.filter((_, j) => j !== i))}>Remove</button>
              )}
            </div>
            <ParamForm model={model} values={s.values} onChange={(v) => setScenarios(scenarios.map((x, j) => (j === i ? { ...x, values: v } : x)))} />
          </section>
        ))}
      <div className="flex flex-wrap items-end gap-3">
        <button className="btn-ghost" onClick={() => setScenarios([...scenarios, { name: `scenario_${scenarios.length}`, values: { ...scenarios[0].values } }])}>
          + Add scenario
        </button>
        <label className="flex flex-col gap-1"><span className="label">Replications</span><input className="input w-28" value={replications} onChange={(e) => setReplications(e.target.value)} /></label>
        <label className="flex flex-col gap-1"><span className="label">Seed</span><input className="input w-28" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
        <label className="flex flex-col gap-1"><span className="label">Duration (optional)</span><input className="input w-36" value={duration} placeholder={fmt(model?.duration)} onChange={(e) => setDuration(e.target.value)} /></label>
        <button className="btn" disabled={busy || !model} onClick={() => void submit()}>{busy ? "Running…" : "Run experiment"}</button>
      </div>
      {progress && (
        <div className="max-w-md space-y-1">
          <div className="h-2 overflow-hidden rounded-full bg-slate-200">
            <div className="h-full bg-indigo-600 transition-all" style={{ width: `${progress[1] ? (100 * progress[0]) / progress[1] : 0}%` }} />
          </div>
          <p className="text-xs text-slate-500">{progress[0]} of {progress[1]} runs</p>
        </div>
      )}
      {runError && <Loading error={runError} />}
    </div>
  );
}
