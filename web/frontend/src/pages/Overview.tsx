import { useMemo, useState } from "react";
import { fmt } from "../format";
import { Loading, useResult } from "./common";

export function OverviewView({ id }: { id: string }) {
  const { data, error } = useResult(id);
  const [filter, setFilter] = useState("");
  const rows = useMemo(() => {
    if (!data) return [];
    const f = filter.trim().toLowerCase();
    return data.summary.filter((r) => !f || r.metric.toLowerCase().includes(f) || r.scenario.toLowerCase().includes(f));
  }, [data, filter]);
  if (!data) return <Loading error={error} />;
  const md = data.metadata;
  const paramNames = Array.from(new Set(Object.values(data.parameters).flatMap((p) => Object.keys(p))));

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-bold text-slate-900">{md.name}</h1>
        <p className="text-sm text-slate-500">{md.experiment_id}</p>
      </header>

      <section className="grid gap-4 md:grid-cols-4">
        {[
          ["Model", `${md.model_name} v${md.model_version}`],
          ["Seed", md.seed],
          ["Replications", `${md.replications} × ${data.scenarios.length} scenario(s)`],
          ["Common random numbers", md.common_random_numbers ? "yes (paired comparisons)" : "no"],
          ["Run length", md.duration === null ? "until no events" : `${fmt(md.duration)} (warm-up ${fmt(md.warmup)})`],
          ["Created", md.timestamp],
          ["Git commit", md.git?.commit ? `${md.git.commit.slice(0, 12)}${md.git.dirty ? " (dirty)" : ""}` : "–"],
          ["Runtime", `${fmt(md.runtime)} s on ${md.workers} worker(s)`],
        ].map(([k, v]) => (
          <div key={String(k)} className="card">
            <p className="label">{k}</p>
            <p className="mt-1 text-sm font-medium text-slate-800">{String(v)}</p>
          </div>
        ))}
      </section>

      {data.errors.length > 0 && (
        <div className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">
          {data.errors.length} replication(s) failed. First: {data.errors[0].scenario} #{data.errors[0].replication}: {data.errors[0].error}
        </div>
      )}

      <section className="card overflow-x-auto">
        <h2 className="mb-2 font-semibold">Parameters</h2>
        <table className="table">
          <thead>
            <tr><th>Scenario</th>{paramNames.map((p) => <th key={p}>{p}</th>)}</tr>
          </thead>
          <tbody>
            {data.scenarios.map((s) => (
              <tr key={s}>
                <td className="font-medium">{s}</td>
                {paramNames.map((p) => <td key={p}>{fmt(data.parameters[s]?.[p])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card overflow-x-auto">
        <div className="mb-2 flex items-center gap-3">
          <h2 className="font-semibold">Metrics (mean and 95% CI over replications)</h2>
          <input className="input ml-auto w-64" placeholder="filter metrics or scenarios…" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </div>
        <table className="table">
          <thead>
            <tr><th>Scenario</th><th>Metric</th><th>n</th><th>Mean</th><th>95% CI</th><th>Std</th><th>Min</th><th>Max</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.scenario}|${r.metric}`}>
                <td>{r.scenario}</td>
                <td className="font-mono text-xs">{r.metric}</td>
                <td>{r.n}</td>
                <td className="font-medium">{fmt(r.mean)}</td>
                <td>{r.ci_low === null ? "–" : `${fmt(r.ci_low)} … ${fmt(r.ci_high)}`}</td>
                <td>{fmt(r.std)}</td>
                <td>{fmt(r.min)}</td>
                <td>{fmt(r.max)}</td>
                <td>
                  <a className="text-xs text-indigo-600 hover:underline" href={`#/result/${id}/distributions?metric=${encodeURIComponent(r.metric)}&scenario=${encodeURIComponent(r.scenario)}`}>
                    distribution
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
