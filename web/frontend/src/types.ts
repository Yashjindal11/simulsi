export type Num = number | null;

export interface Job {
  id: string;
  status: "running" | "done" | "error";
  done: number;
  total: number;
  result_id: string | null;
  error: string | null;
  kind?: string;
  output?: unknown;
}

export interface SensitivityRowT {
  parameter: string;
  output: string;
  method: string;
  value: Num;
  ci_low: Num;
  ci_high: Num;
  setting: string;
  detail: string;
}

export interface MorrisOutput {
  method: "morris";
  rows: SensitivityRowT[];
  info: { r: number; levels: number; evaluations: number };
}

export interface MonteCarloOutput {
  iterations: number;
  sampling: string;
  outputs: Record<string, { summary: { n: number; mean: Num; std: Num; ci_low: Num; ci_high: Num }; quantiles: Record<string, Num>; values: Num[] }>;
  sensitivity: SensitivityRowT[];
}

export interface ResultIndex {
  id: string;
  source: string;
  experiment_id: string;
  name: string;
  model: string;
  model_version: string;
  timestamp: string;
  scenarios: string[];
  replications: number;
}

export interface Metadata {
  experiment_id: string;
  name: string;
  model_name: string;
  model_version: string;
  seed: number;
  replications: number;
  common_random_numbers: boolean;
  scenarios: Record<string, Record<string, unknown>>;
  timestamp: string;
  git: { commit: string | null; dirty: boolean | null } | null;
  environment: Record<string, unknown>;
  runtime: number;
  workers: number;
  duration: Num;
  warmup: number;
}

export interface SummaryRow {
  scenario: string;
  metric: string;
  n: number;
  mean: Num;
  std: Num;
  ci_low: Num;
  ci_high: Num;
  half_width: Num;
  min: Num;
  median: Num;
  max: Num;
}

export interface ResultDetail {
  metadata: Metadata;
  scenarios: string[];
  metrics: string[];
  summary: SummaryRow[];
  errors: { scenario: string; replication: number; error: string | null }[];
  parameters: Record<string, Record<string, unknown>>;
}

export interface MetricDetail {
  metric: string;
  scenario: string;
  values: Num[];
  summary: { n: number; mean: Num; std: Num; ci_low: Num; ci_high: Num; median: Num; min: Num; max: Num };
  convergence: { n: number[]; running_mean: Num[]; running_half_width: Num[] };
}

export interface CompareRow {
  metric: string;
  baseline: string;
  scenario: string;
  baseline_mean: Num;
  scenario_mean: Num;
  absolute_difference: Num;
  percentage_difference: Num;
  ci_low: Num;
  ci_high: Num;
  p_value: Num;
  method: string;
  n: number;
  significant: boolean;
  p_adjusted: Num;
}

export interface ParamInfo {
  name: string;
  default: unknown;
  required: boolean;
  kind: string;
  low: Num;
  high: Num;
  choices: unknown[] | null;
  description: string;
  unit: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  description: string;
  duration: Num;
  warmup: number;
  parameters: ParamInfo[];
  outputs: string[];
  presets: Record<string, Record<string, unknown>>;
}

export interface LogRecord {
  timestamp: number;
  event_type: string;
  entity: string | null;
  resource: string | null;
  old_state: string | null;
  new_state: string | null;
  metadata: Record<string, unknown>;
}

export interface Trace {
  seed: number;
  end_time: number;
  events: number;
  metrics: Record<string, Num>;
  warnings: string[];
  series: Record<string, [number, number][]>;
  log: LogRecord[];
  log_dropped: number;
}

export interface GanttBar {
  tail: string;
  flight: string;
  origin: string;
  dest: string;
  std: number;
  sta: number;
  etd: number;
  eta: number;
  p_on_time: number;
  p_cancel: number;
  p80: number | null;
  cause: string;
  status: string;
}

export interface OpsFlight {
  id: string;
  tail: string;
  origin: string;
  dest: string;
  std: string;
  sta: string;
  p_on_time: number | null;
  p_cancel: number | null;
  dep_delay_p50: number | null;
  dep_delay_p80: number | null;
  dep_delay_p95: number | null;
  main_cause: string;
  misconnect_pax: number | null;
  status: string;
}

export interface OpsForecast {
  replications: number;
  now: number | null;
  summary: Record<string, { mean: number; p10: number; p90: number }>;
  flights: OpsFlight[];
  rotations: { tail: string; legs: number; expected_delay_minutes: number; p_any_cancel: number; worst_leg: string; first_at_risk: string }[];
  alerts: { level: string; kind: string; message: string; flight: string }[];
  gantt: GanttBar[];
}
