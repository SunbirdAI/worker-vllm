"use client";

import { useState } from "react";
import { ArrowLeftRight, MessageSquare, Send } from "lucide-react";
import { LanguagePicker } from "./language-picker";

interface Props {
  busy: boolean;
  onSubmit: (payload: { source: string; target: string; text: string }) => void;
  onExit: () => void;
}

export function TranslateForm({ busy, onSubmit, onExit }: Props) {
  const [source, setSource] = useState("English");
  const [target, setTarget] = useState("Luganda");
  const [text, setText] = useState("");

  const canSubmit =
    !busy &&
    source.length > 0 &&
    target.length > 0 &&
    source !== target &&
    text.trim().length > 0;

  return (
    <form
      className="rounded-2xl border border-border bg-card p-5 shadow-sm"
      onSubmit={(e) => {
        e.preventDefault();
        if (!canSubmit) return;
        onSubmit({ source, target, text: text.trim() });
      }}
    >
      <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-3">
        <div>
          <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
            From
          </label>
          <LanguagePicker
            value={source}
            onChange={setSource}
            excludeValue={target}
          />
        </div>

        <button
          type="button"
          onClick={() => {
            setSource(target);
            setTarget(source);
          }}
          disabled={busy}
          aria-label="Swap languages"
          className="mb-1 inline-flex h-10 w-10 items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:border-accent/50 hover:bg-muted disabled:opacity-40"
        >
          <ArrowLeftRight className="h-4 w-4" />
        </button>

        <div>
          <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
            To
          </label>
          <LanguagePicker
            value={target}
            onChange={setTarget}
            excludeValue={source}
          />
        </div>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text you want to translate"
        className="mt-4 min-h-[96px] w-full resize-none rounded-xl border border-border bg-input p-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-accent/60"
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (canSubmit) onSubmit({ source, target, text: text.trim() });
          }
        }}
      />

      <div className="mt-3 flex items-center justify-between">
        <button
          type="button"
          onClick={onExit}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-full border border-accent/50 px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/10 disabled:opacity-40"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          Exit translation
        </button>
        <button
          type="submit"
          disabled={!canSubmit}
          aria-label="Submit translation"
          className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-muted text-foreground transition-colors hover:bg-muted/80 disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
    </form>
  );
}
