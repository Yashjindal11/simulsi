import { useState } from "react";
import { api, waitForJob } from "../api";

type Row = Record<string, string | number | boolean | null>;

interface RecoverOut {
  actions: string[];
  steps: Row[];
  candidates_tried: number;
}

interface ReservesOut {
  airport: string;
  rows: Row[];
  front: number[];
  recommended: Row;
}

function Table({ rows, mark }: { rows: Row[]; mark?: (i: number) => boolean }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    <table className="table">
      <thead><tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className={mark?.(i) ? "bg-indigo-50" : ""}>
            {cols.map((c) => <td key={c}>{typeof r[c] === "number" ? (r[c] as number).toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(r[c] ?? "-")}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Background searches on the same inputs as the forecast: recovery actions and reserve sizing. */
export function OpsPlanning({ body }: { body: () => Record<string, unknown> }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recover, setRecover] = useState<RecoverOut | null>(null);
  const [reserves, setReserves] = useState<ReservesOut | null>(null);
  const [reps, setReps] = useState(30);
  const [spares, setSpares] = useState("0,1,2");
  const [standby, setStandby] = useState("0,1,2");
  const [maxActions, setMaxActions] = useState(3);

  const start = async (kind: "recover" | "reserves") => {
    setBusy(kind);
    setError(null);
    try {
      const payload = { ...body(), replications: reps, max_actions: maxActions, spares, standby, actions: [] };
      const job = await (kind === "recover" ? api.aviationRecover(payload) : api.aviationReserves(payload));
      const done = await waitForJob(job);
      if (kind === "recover") setRecover(done.output as RecoverOut);
      else setReserves(done.output as ReservesOut);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="card space-y-3">
      <h3 className="font-semibold">Decisions: recovery actions and reserves</h3>
      <p className="text-sm text-slate-600">
        Both run on the same schedule, rules, weather and live status as the forecast, with every option simulated on the same disturbances.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="label">Simulated days per option</span>
          <input className="input w-24" type="number" min={5} max={200} value={reps} onChange={(e) => setReps(Number(e.target.value))} />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Max actions</span>
          <input className="input w-20" type="number" min={1} max={5} value={maxActions} onChange={(e) => setMaxActions(Number(e.target.value))} />
        </label>
        <button className="btn" disabled={busy !== null} onClick={() => start("recover")}>{busy === "recover" ? "Searching…" : "Find recovery actions"}</button>
        <label className="flex flex-col gap-1">
          <span className="label">Spares to try</span>
          <input className="input w-24" value={spares} onChange={(e) => setSpares(e.target.value)} />
        </label>
        <label className="flex flex-col gap-1">
          <span className="label">Standby crews to try</span>
          <input className="input w-24" value={standby} onChange={(e) => setStandby(e.target.value)} />
        </label>
        <button className="btn" disabled={busy !== null} onClick={() => start("reserves")}>{busy === "reserves" ? "Simulating…" : "Plan reserves"}</button>
      </div>
      {error && <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      {recover && (
        <div className="space-y-2 overflow-x-auto">
          <p className="text-sm font-semibold">
            {recover.actions.length ? `Recommended (${recover.candidates_tried} candidates tried): ${recover.actions.join(" · ")}` : `No action clearly beats the plan (${recover.candidates_tried} candidates tried).`}
          </p>
          <Table rows={recover.steps} />
        </div>
      )}
      {reserves && (
        <div className="space-y-2 overflow-x-auto">
          <p className="text-sm font-semibold">
            Lowest total cost at {reserves.airport}: {String(reserves.recommended.spares)} spare aircraft, {String(reserves.recommended.standby_crews)} standby crews. Highlighted rows are on the Pareto front.
          </p>
          <Table rows={reserves.rows} mark={(i) => reserves.front.includes(i)} />
        </div>
      )}
    </section>
  );
}
