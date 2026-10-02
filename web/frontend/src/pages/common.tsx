import { useEffect, useState } from "react";
import { api } from "../api";
import type { ResultDetail } from "../types";

/** Fetch a result's detail; shared by the result tabs. */
export function useResult(id: string) {
  const [data, setData] = useState<ResultDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    api
      .result(id)
      .then((d) => live && setData(d))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [id]);
  return { data, error };
}

export function Loading({ error }: { error?: string | null }) {
  if (error) return <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</div>;
  return <p className="text-sm text-slate-500">Loading…</p>;
}

export function Select({ value, options, onChange, label }: { value: string; options: string[]; onChange: (v: string) => void; label: string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}
