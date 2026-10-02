import { useEffect, useState } from "react";
import { api } from "../api";
import { BandChart, Histogram } from "../charts";
import { finite, fmt } from "../format";
import type { MetricDetail } from "../types";
import { Loading, Select, useResult } from "./common";

function initial(key: string): string {
  const q = window.location.hash.split("?")[1] ?? "";
  return new URLSearchParams(q).get(key) ?? "";
}

export function DistributionsView({ id }: { id: string }) {
  const { data, error } = useResult(id);
  const [metric, setMetric] = useState(() => initial("metric"));
  const [scenario, setScenario] = useState(() => initial("scenario"));
  const [detail, setDetail] = useState<MetricDetail | null>(null);
  const [mError, setMError] = useState<string | null>(null);

  useEffect(() => {
    if (!data) return;
    if (!metric || !data.metrics.includes(metric)) {
      setMetric(data.metrics.find((m) => m.endsWith("wait.mean")) ?? data.metrics[0] ?? "");
    }
    if (!scenario || !data.scenarios.includes(scenario)) setScenario(data.scenarios[0]);
  }, [data, metric, scenario]);

  useEffect(() => {
    if (!metric || !scenario) return;
    setMError(null);
    api.metric(id, metric, scenario).then(setDetail).catch((e: Error) => setMError(e.message));
  }, [id, metric, scenario]);

  if (!data) return <Loading error={error} />;
  const values = detail ? finite(detail.values) : [];
  const s = detail?.summary;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <Select label="Metric" value={metric} options={data.metrics} onChange={setMetric} />
        <Select label="Scenario" value={scenario} options={data.scenarios} onChange={setScenario} />
      </div>
      {mError && <Loading error={mError} />}
      {detail && s && (
        <>
          <section className="grid gap-4 md:grid-cols-5">
            {[["n", s.n], ["mean", s.mean], ["95% CI", s.ci_low === null ? "–" : `${fmt(s.ci_low)} … ${fmt(s.ci_high)}`], ["std", s.std], ["median", s.median]].map(([k, v]) => (
              <div key={String(k)} className="card">
                <p className="label">{k}</p>
                <p className="mt-1 text-sm font-semibold">{typeof v === "string" ? v : fmt(v)}</p>
              </div>
            ))}
          </section>
          <div className="grid gap-4 xl:grid-cols-2">
            <section className="card">
              <h3 className="mb-1 font-semibold">Per-replication values</h3>
              <p className="mb-2 text-xs text-slate-500">Each replication contributes one value; solid line = mean, dashed = 95% CI of the mean.</p>
              <Histogram values={values} mean={s.mean} ciLow={s.ci_low} ciHigh={s.ci_high} />
            </section>
            <section className="card">
              <h3 className="mb-1 font-semibold">Convergence</h3>
              <p className="mb-2 text-xs text-slate-500">Running mean with its 95% confidence band as replications accumulate.</p>
              <BandChart
                x={detail.convergence.n}
                y={detail.convergence.running_mean}
                lo={detail.convergence.running_mean.map((m, i) => (m === null || detail.convergence.running_half_width[i] === null ? null : m - (detail.convergence.running_half_width[i] as number)))}
                hi={detail.convergence.running_mean.map((m, i) => (m === null || detail.convergence.running_half_width[i] === null ? null : m + (detail.convergence.running_half_width[i] as number)))}
                xLabel="replications"
              />
            </section>
          </div>
        </>
      )}
    </div>
  );
}
