import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { DiffBars } from "../charts";
import { fmt, pct } from "../format";
import type { CompareRow } from "../types";
import { Loading, Select, useResult } from "./common";

export function CompareView({ id }: { id: string }) {
  const { data, error } = useResult(id);
  const [baseline, setBaseline] = useState("");
  const [filter, setFilter] = useState("wait");
  const [rows, setRows] = useState<CompareRow[] | null>(null);
  const [cmpError, setCmpError] = useState<string | null>(null);

  useEffect(() => {
    if (data && !baseline) setBaseline(data.scenarios.includes("baseline") ? "baseline" : data.scenarios[0]);
  }, [data, baseline]);

  const metrics = useMemo(() => {
    if (!data) return [];
    const f = filter.trim().toLowerCase();
    const m = data.metrics.filter((x) => !f || x.toLowerCase().includes(f));
    return m.slice(0, 40);
  }, [data, filter]);

  useEffect(() => {
    if (!baseline || !data || data.scenarios.length < 2 || metrics.length === 0) {
      setRows([]);
      return;
    }
    setCmpError(null);
    api.compare(id, baseline, metrics).then(setRows).catch((e: Error) => setCmpError(e.message));
  }, [id, baseline, metrics, data]);

  if (!data) return <Loading error={error} />;
  if (data.scenarios.length < 2) return <p className="text-sm text-slate-500">This experiment has a single scenario; nothing to compare.</p>;

  const byMetric = new Map<string, CompareRow[]>();
  (rows ?? []).forEach((r) => byMetric.set(r.metric, [...(byMetric.get(r.metric) ?? []), r]));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <Select label="Baseline" value={baseline} options={data.scenarios} onChange={setBaseline} />
        <label className="flex flex-col gap-1">
          <span className="label">Metrics containing</span>
          <input className="input w-64" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="e.g. wait, utilization" />
        </label>
        <p className="max-w-xl text-xs text-slate-500">
          Differences are scenario − baseline with 95% confidence intervals ({data.metadata.common_random_numbers ? "paired t, common random numbers" : "Welch t"}).
          They quantify simulation sampling error only; no multiple-comparison correction is applied.
        </p>
      </div>
      {cmpError && <Loading error={cmpError} />}
      {rows === null && <Loading />}
      {Array.from(byMetric.entries()).map(([metric, rs]) => (
        <section key={metric} className="card">
          <h3 className="mb-2 font-mono text-sm font-semibold">{metric}</h3>
          <div className="grid gap-4 lg:grid-cols-2">
            <table className="table">
              <thead>
                <tr><th>Scenario</th><th>Baseline</th><th>Value</th><th>Diff</th><th>Diff %</th><th>95% CI</th><th /></tr>
              </thead>
              <tbody>
                {rs.map((r) => (
                  <tr key={r.scenario}>
                    <td className="font-medium">{r.scenario}</td>
                    <td>{fmt(r.baseline_mean)}</td>
                    <td>{fmt(r.scenario_mean)}</td>
                    <td>{fmt(r.absolute_difference)}</td>
                    <td>{pct(r.percentage_difference)}</td>
                    <td>{r.ci_low === null ? "–" : `${fmt(r.ci_low)} … ${fmt(r.ci_high)}`}</td>
                    <td>{r.significant ? <span className="rounded bg-indigo-100 px-1.5 text-xs text-indigo-800">CI excludes 0</span> : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <DiffBars bars={rs.map((r) => ({ label: r.scenario, value: r.absolute_difference, lo: r.ci_low, hi: r.ci_high, highlight: r.significant }))} />
          </div>
        </section>
      ))}
      {rows && rows.length === 0 && <p className="text-sm text-slate-500">No metrics match the filter.</p>}
    </div>
  );
}
