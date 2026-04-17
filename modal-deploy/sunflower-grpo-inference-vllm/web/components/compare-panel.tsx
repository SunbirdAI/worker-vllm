"use client";

import { AlertCircle, Loader2, Sparkles, Cloud } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

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
}

function Panel({ title, subtitle, icon, state }: PanelProps) {
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
          <div className="flex items-start gap-2 text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <span>{state.error ?? "Request failed"}</span>
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

interface Props {
  grpo: PanelState;
  prod: PanelState;
}

export function ComparePanel({ grpo, prod }: Props) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Panel
        title="Sunflower GRPO (test)"
        subtitle="streaming · /generate_stream"
        icon={<Sparkles className="h-3.5 w-3.5" />}
        state={grpo}
      />
      <Panel
        title="Production"
        subtitle="blocking · api.sunbird.ai"
        icon={<Cloud className="h-3.5 w-3.5" />}
        state={prod}
      />
    </div>
  );
}
