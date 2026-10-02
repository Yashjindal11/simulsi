import type { Num } from "./types";

export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
    if (Number.isInteger(v)) return v.toLocaleString();
    return a >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : Number(v.toPrecision(digits)).toString();
  }
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function pct(v: Num): string {
  return v === null || !Number.isFinite(v) ? "–" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

export function finite(values: Num[]): number[] {
  return values.filter((v): v is number => v !== null && Number.isFinite(v));
}
