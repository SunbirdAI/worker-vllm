"use client";

import { Thermometer } from "lucide-react";

interface Props {
  value: number;
  onChange: (value: number) => void;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
}

export function TemperatureSlider({
  value,
  onChange,
  hint,
  min = 0,
  max = 1,
  step = 0.05,
}: Props) {
  return (
    <div className="flex flex-col gap-2 rounded-2xl border border-border bg-card px-4 py-3 shadow-sm sm:flex-row sm:items-center sm:gap-4">
      <div className="flex items-center gap-2">
        <Thermometer className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-foreground">Temperature</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label="Sampling temperature"
        className="flex-1 accent-accent"
      />
      <div className="flex items-center justify-between gap-3 sm:justify-end">
        <span className="font-mono text-xs tabular-nums text-foreground">
          {value.toFixed(2)}
        </span>
        {hint && (
          <span className="text-xs text-muted-foreground">{hint}</span>
        )}
      </div>
    </div>
  );
}
