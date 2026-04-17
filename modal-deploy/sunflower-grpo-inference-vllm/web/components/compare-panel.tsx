"use client";

import { AlertCircle, Loader2, RotateCw, Sparkles, Cloud } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { friendlyError } from "@/lib/errors";

export type PanelStatus = "idle" | "loading" | "done" | "error";

export interface PanelState {
  text: string;
  status: PanelStatus;
  error?: string;
  latencyMs?: number;
}

interface PanelProps {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  state: PanelState;
  onRetry?: () => void;
  retryDisabled?: boolean;
  headerExtra?: React.ReactNode;
}

function Panel({
  title,
  subtitle,
  icon,
  state,
  onRetry,
  retryDisabled,
  headerExtra,
}: PanelProps) {
  return (
    <div className="flex min-h-[240px] flex-col rounded-2xl border border-border bg-card shadow-sm">
      <div className="flex items-center gap-3 border-b border-border px-4 py-3">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-muted text-foreground">
          {icon}
        </span>
        <div className="flex-1">
          <div className="text-sm font-semibold text-foreground">{title}</div>
          <div className="text-xs text-muted-foreground">{subtitle}</div>
        </div>
        {headerExtra}
        {state.status === "loading" && (
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        )}
        {state.status === "done" && typeof state.latencyMs === "number" && (
          <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {Math.round(state.latencyMs)} ms
          </span>
        )}
      </div>

      <div className="flex-1 p-4 text-sm leading-relaxed text-foreground">
        {state.status === "error" ? (
          <div className="flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <span>{friendlyError(state.error)}</span>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                disabled={retryDisabled}
                className="inline-flex items-center gap-2 rounded-full border border-border bg-muted px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted/80 disabled:opacity-40"
              >
                <RotateCw className="h-3.5 w-3.5" />
                Try again
              </button>
            )}
          </div>
        ) : state.text ? (
          <div className="prose prose-sm max-w-none break-words dark:prose-invert prose-pre:bg-muted prose-code:text-accent">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {state.text}
            </ReactMarkdown>
          </div>
        ) : (
          <span className="text-muted-foreground">
            {state.status === "loading" ? "Waiting for first token…" : "Awaiting response…"}
          </span>
        )}
      </div>
    </div>
  );
}

export type ProdMode = "stream" | "block";

interface Props {
  grpo: PanelState;
  prod: PanelState;
  onRetryGrpo?: () => void;
  onRetryProd?: () => void;
  retryDisabled?: boolean;
  prodMode: ProdMode;
  onProdModeChange: (mode: ProdMode) => void;
  prodModeDisabled?: boolean;
}

function ProdModeToggle({
  value,
  onChange,
  disabled,
}: {
  value: ProdMode;
  onChange: (mode: ProdMode) => void;
  disabled?: boolean;
}) {
  const options: { key: ProdMode; label: string }[] = [
    { key: "stream", label: "Stream" },
    { key: "block", label: "Block" },
  ];
  return (
    <div
      role="radiogroup"
      aria-label="Production response mode"
      className="inline-flex items-center rounded-full border border-border bg-muted p-0.5 text-xs"
    >
      {options.map((opt) => {
        const selected = value === opt.key;
        return (
          <button
            key={opt.key}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(opt.key)}
            className={
              "rounded-full px-2.5 py-0.5 font-medium transition-colors disabled:opacity-40 " +
              (selected
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground")
            }
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function ComparePanel({
  grpo,
  prod,
  onRetryGrpo,
  onRetryProd,
  retryDisabled,
  prodMode,
  onProdModeChange,
  prodModeDisabled,
}: Props) {
  const prodSubtitle =
    prodMode === "stream"
      ? "streaming · /generate_openai_stream"
      : "blocking · api.sunbird.ai";

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Panel
        title="Sunflower GRPO (test)"
        subtitle="streaming · /generate_stream"
        icon={<Sparkles className="h-3.5 w-3.5" />}
        state={grpo}
        onRetry={onRetryGrpo}
        retryDisabled={retryDisabled}
      />
      <Panel
        title="Production"
        subtitle={prodSubtitle}
        icon={<Cloud className="h-3.5 w-3.5" />}
        state={prod}
        onRetry={onRetryProd}
        retryDisabled={retryDisabled}
        headerExtra={
          <ProdModeToggle
            value={prodMode}
            onChange={onProdModeChange}
            disabled={prodModeDisabled}
          />
        }
      />
    </div>
  );
}
