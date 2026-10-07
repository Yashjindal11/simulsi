import { useEffect, useState } from "react";
import { api, waitForJob } from "../api";
import { DiffBars, Histogram } from "../charts";
import { fmt } from "../format";
import type { ModelInfo, MonteCarloOutput, MorrisOutput } from "../types";
import { Loading, Select } from "./common";
import { ParamForm, parseValue, toParams, useModels } from "./Run";

type Mode = "grid" | "morris" | "montecarlo";

const MODES: [Mode, string][] = [
  ["grid", "Grid sweep"],
  ["morris", "Sensitivity (Morris)"],
  ["montecarlo", "Monte Carlo"],
];

/** Numeric parameters, the ones that can be swept or sampled. */
function numericParams(model: ModelInfo) {
  return model.parameters.filter((p) => ["int", "float", "probability", "any"].includes(p.kind) && typeof p.default === "number");
}

export function ExploreView({ onDone }: { onDone: (id: string) => void }) {
  const { models, error } = useModels();
  const [mode, setMode] = useState<Mode>("grid");
  const [modelId, setModelId] = useState("");
  const [base, setBase] = useState<Record<string, string>>({});
  const [spec, setSpec] = useState<Record<string, string>>({});
  const [outputs, setOutputs] = useState("");
  const [count, setCount] = useState("5");
  const [seed, setSeed] = useState("1");
  const [duration, setDuration] = useState("");
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [morris, setMorris] = useState<MorrisOutput | null>(null);
  const [mc, setMc] = useState<MonteCarloOutput | null>(null);

  useEffect(() => {
    if (models && !modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  if (!models) return <Loading error={error} />;
  const model = models.find((m) => m.id === modelId);

  const outputList = () => outputs.split(",").map((s) => s.trim()).filter(Boolean);

  const submit = async () => {
    if (!model) return;
    setBusy(true);
    setRunError(null);
    setMorris(null);
    setMc(null);
    try {
      const common = {
        model: model.id,
        seed: Number(seed),
        duration: duration.trim() ? Number(duration) : undefined,
        base: toParams(model, Object.fromEntries(Object.entries(base).filter(([k]) => !(k in spec) || !spec[k].trim()))),
      };
      const filled = Object.entries(spec).filter(([, v]) => v.trim());
      if (!filled.length) throw new Error("fill in at least one parameter to vary");
      if (mode === "grid") {
        const factors = Object.fromEntries(
          filled.map(([k, v]) => [k, v.split(",").map((s) => parseValue(s, model.parameters.find((p) => p.name === k)))]),
        );
        const job = await waitForJob(await api.grid({ ...common, factors, replications: Number(count) }), (j) => setProgress([j.done, j.total]));
        if (job.result_id) onDone(job.result_id);
      } else if (mode === "morris") {
        const ranges = Object.fromEntries(
          filled.map(([k, v]) => {
            const parts = v.split(",").map(Number);
            if (parts.length !== 2 || parts.some(Number.isNaN)) throw new Error(`${k}: give "low, high"`);
            return [k, parts];
          }),
        );
        const job = await waitForJob(await api.sensitivity({ ...common, ranges, r: Number(count), outputs: outputList() }));
        setMorris(job.output as MorrisOutput);
      } else {
        const inputs = Object.fromEntries(
          filled.map(([k, v]) => {
            try {
              return [k, JSON.parse(v)];
            } catch {
              throw new Error(`${k}: not valid JSON`);
            }
          }),
        );
        const job = await waitForJob(await api.montecarlo({ ...common, inputs, iterations: Number(count), outputs: outputList() }));
        setMc(job.output as MonteCarloOutput);
      }
    } catch (e) {
      setRunError((e as Error).message);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const hint: Record<Mode, { field: string; placeholder: string; count: string; help: string }> = {
    grid: { field: "values (comma-separated)", placeholder: "1, 2, 3", count: "Replications", help: "Every combination of the listed values becomes a scenario (at most 200). The result opens with a Response tab." },
    morris: { field: "range: low, high", placeholder: "0.5, 1.5", count: "Trajectories (r)", help: "Morris screening: r × (parameters + 1) runs. mu* ranks importance; a large sigma means non-linear effects or interactions." },
    montecarlo: { field: "distribution (JSON spec)", placeholder: '{"distribution": "uniform", "low": 1, "high": 2}', count: "Iterations", help: "Samples the uncertain inputs (Latin hypercube), one run per draw, and shows the output distribution and which inputs drive it." },
  };
  const h = hint[mode];

  return (
    <div className="max-w-5xl space-y-4">
      <header>
        <h1 className="text-xl font-bold">Explore a model</h1>
        <p className="text-sm text-slate-600">{h.help}</p>
      </header>
      <div className="flex gap-2">
        {MODES.map(([m, label]) => (
          <button key={m} className={m === mode ? "btn" : "btn-ghost"} onClick={() => { setMode(m); setSpec({}); setMorris(null); setMc(null); }}>{label}</button>
        ))}
      </div>
      <section className="card space-y-3">
        <Select label="Model" value={modelId} options={models.map((m) => m.id)} onChange={(v) => { setModelId(v); setSpec({}); setBase({}); }} />
        {model && (
          <div className="grid gap-3 md:grid-cols-2">
            {numericParams(model).map((p) => (
              <label key={p.name} className="flex flex-col gap-1">
                <span className="label">{p.name} - {h.field}</span>
                <input className="input" value={spec[p.name] ?? ""} placeholder={`leave empty to keep ${fmt(p.default)}`} onChange={(e) => setSpec({ ...spec, [p.name]: e.target.value })} />
              </label>
            ))}
          </div>
        )}
      </section>
      {model && (
        <details className="card">
          <summary className="cursor-pointer text-sm font-semibold">Fixed values for the other parameters</summary>
          <div className="mt-3"><ParamForm model={model} values={base} onChange={setBase} /></div>
        </details>
      )}
      <div className="flex flex-wrap items-end gap-3">
        {mode !== "grid" && (
          <label className="flex flex-col gap-1">
            <span className="label">Output metrics (comma-separated)</span>
            <input className="input w-96" value={outputs} placeholder="resource.server.wait.mean" onChange={(e) => setOutputs(e.target.value)} list="known-outputs" />
            <datalist id="known-outputs">{model?.outputs.map((o) => <option key={o} value={o} />)}</datalist>
          </label>
        )}
        <label className="flex flex-col gap-1"><span className="label">{h.count}</span><input className="input w-28" value={count} onChange={(e) => setCount(e.target.value)} /></label>
        <label className="flex flex-col gap-1"><span className="label">Seed</span><input className="input w-24" value={seed} onChange={(e) => setSeed(e.target.value)} /></label>
        <label className="flex flex-col gap-1"><span className="label">Duration (optional)</span><input className="input w-32" value={duration} placeholder={fmt(model?.duration)} onChange={(e) => setDuration(e.target.value)} /></label>
        <button className="btn" disabled={busy || !model} onClick={() => void submit()}>{busy ? "Running…" : "Run"}</button>
      </div>
      {progress && <p className="text-xs text-slate-500">{progress[0]} of {progress[1]} runs</p>}
      {busy && !progress && <p className="text-xs text-slate-500">Running in the background…</p>}
      {runError && <Loading error={runError} />}
      {morris && <MorrisPanel out={morris} />}
      {mc && <MonteCarloPanel out={mc} />}
    </div>
  );
}

function MorrisPanel({ out }: { out: MorrisOutput }) {
  const byOutput = new Map<string, typeof out.rows>();
  out.rows.forEach((r) => byOutput.set(r.output, [...(byOutput.get(r.output) ?? []), r]));
  return (
    <>
      {[...byOutput.entries()].map(([name, rows]) => {
        const star = rows.filter((r) => r.method === "morris-mu_star").sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
        const sigma = (p: string) => rows.find((r) => r.parameter === p && r.method === "morris-sigma")?.value ?? null;
        const mu = (p: string) => rows.find((r) => r.parameter === p && r.method === "morris-mu")?.value ?? null;
        return (
          <section key={name} className="card space-y-3">
            <h3 className="font-semibold">{name} <span className="text-xs font-normal text-slate-500">({out.info.evaluations} runs)</span></h3>
            <DiffBars bars={star.map((r, i) => ({ label: r.parameter, value: r.value, lo: r.ci_low, hi: r.ci_high, highlight: i === 0 }))} />
            <table className="table">
              <thead><tr><th>Parameter</th><th>mu* (importance)</th><th>95% CI</th><th>mu (direction)</th><th>sigma</th></tr></thead>
              <tbody>
                {star.map((r) => (
                  <tr key={r.parameter}>
                    <td>{r.parameter}</td><td>{fmt(r.value)}</td><td>{fmt(r.ci_low)} – {fmt(r.ci_high)}</td>
                    <td>{fmt(mu(r.parameter))}</td><td>{fmt(sigma(r.parameter))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        );
      })}
    </>
  );
}

function MonteCarloPanel({ out }: { out: MonteCarloOutput }) {
  return (
    <>
      {Object.entries(out.outputs).map(([name, o]) => {
        const sens = out.sensitivity.filter((r) => r.output === name && r.method === "spearman");
        const values = o.values.filter((v): v is number => v !== null);
        return (
          <section key={name} className="card space-y-3">
            <h3 className="font-semibold">{name} <span className="text-xs font-normal text-slate-500">({out.iterations} iterations, {out.sampling})</span></h3>
            <div className="grid gap-4 xl:grid-cols-2">
              <div>
                <Histogram values={values} mean={o.summary.mean} ciLow={o.quantiles.p5} ciHigh={o.quantiles.p95} />
                <p className="text-xs text-slate-500">Solid line: mean. Dashed: 5th and 95th percentiles.</p>
              </div>
              <div>
                <p className="mb-1 text-sm font-semibold">Spearman correlation with each input</p>
                <DiffBars bars={sens.map((r) => ({ label: r.parameter, value: r.value, lo: null, hi: null }))} />
              </div>
            </div>
            <table className="table">
              <tbody>
                <tr><td>mean (95% CI)</td><td>{fmt(o.summary.mean)} ({fmt(o.summary.ci_low)} – {fmt(o.summary.ci_high)})</td></tr>
                {Object.entries(o.quantiles).map(([q, v]) => <tr key={q}><td>{q}</td><td>{fmt(v)}</td></tr>)}
              </tbody>
            </table>
          </section>
        );
      })}
    </>
  );
}
