import { useEffect, useMemo, useState } from "react";
import { LineCI } from "../charts";
import { Loading, Select, useResult } from "./common";

/** Metric means (with CIs) against one numeric parameter, one line per value of another. */
export function ResponseView({ id }: { id: string }) {
  const { data, error } = useResult(id);
  const [metric, setMetric] = useState("");
  const [xParam, setXParam] = useState("");
  const [group, setGroup] = useState(""); // "" = automatic: the first other varying parameter

  const varying = useMemo(() => {
    if (!data) return [] as string[];
    const keys = new Set(Object.values(data.parameters).flatMap((p) => Object.keys(p)));
    return [...keys].filter((k) => {
      const vals = new Set(Object.values(data.parameters).map((p) => JSON.stringify(p[k])));
      return vals.size > 1;
    });
  }, [data]);
  const numeric = useMemo(
    () => (data ? varying.filter((k) => Object.values(data.parameters).every((p) => typeof p[k] === "number")) : []),
    [data, varying],
  );

  useEffect(() => {
    if (!data) return;
    if (!metric) setMetric(data.metrics.find((m) => m.endsWith("wait.mean")) ?? data.metrics[0] ?? "");
    if (!xParam && numeric.length) setXParam(numeric[0]);
  }, [data, metric, xParam, numeric]);

  if (!data) return <Loading error={error} />;
  if (!numeric.length) {
    return (
      <p className="text-sm text-slate-600">
        No numeric parameter varies across scenarios. Run a grid sweep (Explore → Grid sweep) to see response curves.
      </p>
    );
  }
  const groups = varying.filter((k) => k !== xParam);
  const groupKey = group === "" ? (groups[0] ?? null) : groups.includes(group) ? group : null;
  const curves = new Map<string, { x: number; y: number | null; lo: number | null; hi: number | null }[]>();
  for (const sc of data.scenarios) {
    const params = data.parameters[sc] ?? {};
    const row = data.summary.find((r) => r.scenario === sc && r.metric === metric);
    if (!row) continue;
    const name = groupKey ? `${groupKey}=${JSON.stringify(params[groupKey])}` : metric;
    const list = curves.get(name) ?? [];
    list.push({ x: params[xParam] as number, y: row.mean, lo: row.ci_low, hi: row.ci_high });
    curves.set(name, list);
  }
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <Select label="Metric" value={metric} options={data.metrics} onChange={setMetric} />
        <Select label="x axis" value={xParam} options={numeric} onChange={setXParam} />
        <Select label="One line per" value={groupKey ?? "(none)"} options={["(none)", ...groups]} onChange={setGroup} />
      </div>
      <section className="card">
        <h3 className="mb-2 font-semibold">{metric} vs {xParam}</h3>
        <LineCI curves={[...curves.entries()].map(([name, points]) => ({ name, points }))} xLabel={xParam} yLabel={metric} />
        <p className="mt-2 text-xs text-slate-500">
          Points are scenario means; whiskers are {Math.round(100 * ((data.summary[0] as { confidence?: number } | undefined)?.confidence ?? 0.95))}%
          confidence intervals over replications. Other varying parameters are not held fixed unless chosen as the line grouping.
        </p>
      </section>
    </div>
  );
}
