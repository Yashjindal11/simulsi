import { useRef, useState } from "react";
import { api } from "../api";
import type { ResultIndex } from "../types";

export function Home({ results, onOpened }: { results: ResultIndex[]; onOpened: (id: string) => void }) {
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const load = async (file: File) => {
    setMessage(null);
    try {
      const data: unknown = JSON.parse(await file.text());
      const { id } = await api.upload(data);
      onOpened(id);
    } catch (e) {
      setMessage(`Could not load ${file.name}: ${(e as Error).message}`);
    }
  };

  return (
    <div className="max-w-4xl space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">SimulSI dashboard</h1>
        <p className="text-slate-600">Inspect saved experiments, run models, compare scenarios and look inside individual runs. Everything stays on this machine.</p>
      </header>

      <div
        className={`card flex flex-col items-center gap-2 border-2 border-dashed py-10 text-center ${dragging ? "border-indigo-500 bg-indigo-50" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) void load(f);
        }}
      >
        <p className="font-medium text-slate-800">Drop an <code>experiment.json</code> here</p>
        <p className="text-sm text-slate-500">It is read in the browser and sent only to the local SimulSI server.</p>
        <button className="btn-ghost" onClick={() => input.current?.click()}>Choose file…</button>
        <input ref={input} type="file" accept=".json,application/json" className="hidden" onChange={(e) => e.target.files?.[0] && void load(e.target.files[0])} />
        {message && <p className="text-sm text-red-600">{message}</p>}
      </div>

      <section className="card">
        <h2 className="mb-2 font-semibold">Loaded experiments</h2>
        {results.length === 0 ? (
          <p className="text-sm text-slate-500">
            None yet. Start the dashboard with <code>simulsi ui results/my-study</code>, drop a file above, or{" "}
            <a className="text-indigo-600 underline" href="#/run">run an experiment</a>.
          </p>
        ) : (
          <table className="table">
            <thead>
              <tr><th>Name</th><th>Model</th><th>Scenarios</th><th>Reps</th><th>Created</th><th>Source</th></tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.id} className="cursor-pointer hover:bg-slate-50" onClick={() => onOpened(r.id)}>
                  <td className="font-medium text-indigo-700">{r.name}</td>
                  <td>{r.model} v{r.model_version}</td>
                  <td>{r.scenarios.join(", ")}</td>
                  <td>{r.replications}</td>
                  <td>{r.timestamp}</td>
                  <td className="max-w-48 truncate text-xs text-slate-500">{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
