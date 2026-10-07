import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { ResultIndex } from "./types";
import { CompareView } from "./pages/Compare";
import { DistributionsView } from "./pages/Distributions";
import { ExploreView } from "./pages/Explore";
import { Home } from "./pages/Home";
import { OverviewView } from "./pages/Overview";
import { ResponseView } from "./pages/Response";
import { RunView } from "./pages/Run";
import { TraceView } from "./pages/Trace";

type Tab = "overview" | "compare" | "distributions" | "response";

type Route =
  | { page: "home" }
  | { page: "run" }
  | { page: "explore" }
  | { page: "trace" }
  | { page: "result"; id: string; tab: Tab };

const TABS: Tab[] = ["overview", "compare", "distributions", "response"];

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean);
  if (parts[0] === "run") return { page: "run" };
  if (parts[0] === "explore") return { page: "explore" };
  if (parts[0] === "trace") return { page: "trace" };
  if (parts[0] === "result" && parts[1]) {
    const tab = (TABS as string[]).includes(parts[2]) ? (parts[2] as Tab) : "overview";
    return { page: "result", id: parts[1], tab };
  }
  return { page: "home" };
}

export function go(path: string) {
  window.location.hash = path;
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  const [results, setResults] = useState<ResultIndex[]>([]);
  const [version, setVersion] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.results().then(setResults).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const onHash = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onHash);
    api.info().then((i) => setVersion(i.version)).catch((e: Error) => setError(e.message));
    refresh();
    return () => window.removeEventListener("hashchange", onHash);
  }, [refresh]);

  const opened = (id: string) => {
    refresh();
    go(`/result/${id}/overview`);
  };

  const navItem = (label: string, href: string, active: boolean) => (
    <a href={`#${href}`} className={`block rounded-lg px-3 py-1.5 text-sm ${active ? "bg-indigo-600 text-white" : "text-slate-300 hover:bg-slate-800"}`}>
      {label}
    </a>
  );

  return (
    <div className="flex min-h-screen">
      <aside className="w-64 shrink-0 space-y-6 bg-slate-900 p-4 text-slate-100">
        <div>
          <a href="#/" className="text-lg font-bold tracking-tight">SimulSI</a>
          <p className="text-xs text-slate-400">Model the system. Simulate the future.</p>
        </div>
        <nav className="space-y-1">
          {navItem("Home & load", "/", route.page === "home")}
          {navItem("Run experiment", "/run", route.page === "run")}
          {navItem("Explore (grid, sensitivity, MC)", "/explore", route.page === "explore")}
          {navItem("Trace & replay a run", "/trace", route.page === "trace")}
        </nav>
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Experiments</p>
          <div className="space-y-1">
            {results.length === 0 && <p className="text-xs text-slate-500">None loaded yet.</p>}
            {results.map((r) => (
              <a key={r.id} href={`#/result/${r.id}/overview`} className={`block rounded-lg px-3 py-1.5 text-sm ${route.page === "result" && route.id === r.id ? "bg-slate-700" : "hover:bg-slate-800"}`}>
                <span className="block truncate font-medium">{r.name}</span>
                <span className="block truncate text-xs text-slate-400">{r.scenarios.length} scenario(s) · {r.replications} reps</span>
              </a>
            ))}
          </div>
        </div>
        <p className="text-xs text-slate-500">v{version} · local only, no telemetry</p>
      </aside>
      <main className="min-w-0 flex-1 p-6">
        {error && <div className="mb-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>}
        {route.page === "home" && <Home results={results} onOpened={opened} />}
        {route.page === "run" && <RunView onDone={opened} />}
        {route.page === "explore" && <ExploreView onDone={(id) => { refresh(); go(`/result/${id}/response`); }} />}
        {route.page === "trace" && <TraceView />}
        {route.page === "result" && (
          <ResultShell id={route.id} tab={route.tab} />
        )}
      </main>
    </div>
  );
}

function ResultShell({ id, tab }: { id: string; tab: Tab }) {
  const tabs: [Tab, string][] = [["overview", "Overview"], ["compare", "Compare scenarios"], ["distributions", "Distributions"], ["response", "Response"]];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 pb-2">
        {tabs.map(([t, label]) => (
          <a key={t} href={`#/result/${id}/${t}`} className={`rounded-lg px-3 py-1.5 text-sm ${t === tab ? "bg-indigo-100 font-semibold text-indigo-800" : "text-slate-600 hover:bg-slate-100"}`}>
            {label}
          </a>
        ))}
        <span className="flex-1" />
        <a className="btn-ghost" href={api.reportUrl(id)} download>HTML report</a>
        <a className="btn-ghost" href={api.exportUrl(id, "json")} download>Export JSON</a>
        <a className="btn-ghost" href={api.exportUrl(id, "csv")} download>Export CSV</a>
      </div>
      {tab === "overview" && <OverviewView id={id} />}
      {tab === "compare" && <CompareView id={id} />}
      {tab === "distributions" && <DistributionsView id={id} />}
      {tab === "response" && <ResponseView id={id} />}
    </div>
  );
}
