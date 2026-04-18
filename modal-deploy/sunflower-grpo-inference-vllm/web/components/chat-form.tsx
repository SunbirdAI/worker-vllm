"use client";

import { useState } from "react";
import { Languages, Send } from "lucide-react";

interface Props {
  busy: boolean;
  onSubmit: (text: string) => void;
  onEnterTranslate: () => void;
}

export function ChatForm({ busy, onSubmit, onEnterTranslate }: Props) {
  const [text, setText] = useState("");

  const canSubmit = !busy && text.trim().length > 0;

  return (
    <form
      className="rounded-2xl border border-border bg-card p-5 shadow-sm"
      onSubmit={(e) => {
        e.preventDefault();
        if (!canSubmit) return;
        onSubmit(text.trim());
      }}
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type something in English, or any Ugandan language..."
        className="min-h-[96px] w-full resize-none rounded-xl bg-transparent p-2 text-sm text-foreground outline-none placeholder:text-muted-foreground"
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            if (canSubmit) onSubmit(text.trim());
          }
        }}
      />

      <div className="mt-2 flex items-center justify-between">
        <button
          type="button"
          onClick={onEnterTranslate}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-full border border-accent/50 px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/10 disabled:opacity-40"
        >
          <Languages className="h-3.5 w-3.5" />
          Translate
        </button>
        <button
          type="submit"
          disabled={!canSubmit}
          aria-label="Submit"
          className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-muted text-foreground transition-colors hover:bg-muted/80 disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
    </form>
  );
}
