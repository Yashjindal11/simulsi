import type { CompareRow, Job, MetricDetail, ModelInfo, ResultDetail, ResultIndex, Trace } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw new Error((body as { error?: string }).error ?? res.statusText);
  return body as T;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  info: () => request<{ version: string; models: string[] }>("/api/info"),
  models: () => request<ModelInfo[]>("/api/models"),
  results: () => request<ResultIndex[]>("/api/results"),
  result: (id: string) => request<ResultDetail>(`/api/results/${encodeURIComponent(id)}`),
  metric: (id: string, metric: string, scenario: string) =>
    request<MetricDetail>(
      `/api/results/${encodeURIComponent(id)}/metric?metric=${encodeURIComponent(metric)}&scenario=${encodeURIComponent(scenario)}`,
    ),
  compare: (id: string, baseline: string, metrics: string[], adjust = "none", confidence = 0.95) =>
    request<CompareRow[]>(
      `/api/results/${encodeURIComponent(id)}/compare?baseline=${encodeURIComponent(baseline)}&metrics=${encodeURIComponent(metrics.join(","))}&confidence=${confidence}&adjust=${adjust}`,
    ),
  run: (body: unknown) => post<Job>("/api/run", body),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  trace: (body: unknown) => post<Trace>("/api/trace", body),
  upload: (experiment: unknown) => post<{ id: string }>("/api/results", experiment),
  exportUrl: (id: string, kind: "json" | "csv") => `/api/results/${encodeURIComponent(id)}/export.${kind}`,
};
