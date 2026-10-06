export type Num = number | null;

export interface Job {
  id: string;
  status: "running" | "done" | "error";
  done: number;
  total: number;
  result_id: string | null;
  error: string | null;
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
