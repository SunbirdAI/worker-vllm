"use client";

import Image from "next/image";
import { useRef, useState } from "react";
import logoIcon from "@/assets/logo_icon.png";
import { SunflowerHeader } from "@/components/sunflower-header";
import { ChatForm } from "@/components/chat-form";
import { TranslateForm } from "@/components/translate-form";
import {
  ComparePanel,
  type PanelState,
} from "@/components/compare-panel";
import { buildTranslationPrompt } from "@/lib/prompt";
import { generateProduction, streamGenerate } from "@/lib/api";

type Mode = "chat" | "translate";

const EMPTY: PanelState = { text: "", status: "idle" };

export default function Page() {
  const [mode, setMode] = useState<Mode>("chat");
  const [grpo, setGrpo] = useState<PanelState>(EMPTY);
  const [prod, setProd] = useState<PanelState>(EMPTY);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function runComparison(instruction: string) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setBusy(true);
    setGrpo({ text: "", status: "loading" });
    setProd({ text: "", status: "loading" });

    const startedGrpo = performance.now();
    const startedProd = performance.now();

    const grpoTask = (async () => {
      try {
        await streamGenerate(
          instruction,
          {
            signal: controller.signal,
            onDelta: (chunk) => {
              setGrpo((s) => ({
                ...s,
                text: s.text + chunk,
                status: "loading",
              }));
            },
            onDone: () => {
              setGrpo((s) => ({
                ...s,
                status: "done",
                latencyMs: performance.now() - startedGrpo,
              }));
            },
            onError: (err) => {
              setGrpo({ text: "", status: "error", error: err.message });
            },
          },
          { temperature: 0.2, maxTokens: 1024 },
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setGrpo({
          text: "",
          status: "error",
          error: (err as Error)?.message ?? "unknown error",
        });
      }
    })();

    const prodTask = (async () => {
      try {
        const data = await generateProduction({
          instruction,
          signal: controller.signal,
        });
        const text =
          typeof data?.response === "string"
            ? data.response
            : JSON.stringify(data, null, 2);
        setProd({
          text,
          status: "done",
          latencyMs: performance.now() - startedProd,
        });
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setProd({
          text: "",
          status: "error",
          error: (err as Error)?.message ?? "unknown error",
        });
      }
    })();

    await Promise.all([grpoTask, prodTask]);
    setBusy(false);
  }

  const hasResults = grpo.status !== "idle" || prod.status !== "idle";

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-8 sm:px-6 sm:py-12">
      <SunflowerHeader />

      <section className="mt-12 flex flex-col items-center text-center">
        <div className="flex items-center gap-3">
          <Image
            src={logoIcon}
            alt=""
            width={44}
            height={44}
            priority
            className="h-11 w-11"
          />
          <h1 className="text-3xl font-semibold tracking-tight">Sunflower</h1>
        </div>
        <p className="mt-3 max-w-xl text-sm text-muted-foreground">
          Compare the Sunflower GRPO (test) model side-by-side with the
          production Sunbird AI API across 30+ Ugandan languages.
        </p>
      </section>

      <section className="mt-8">
        {mode === "chat" ? (
          <ChatForm
            busy={busy}
            onSubmit={(text) => runComparison(text)}
            onEnterTranslate={() => setMode("translate")}
          />
        ) : (
          <TranslateForm
            busy={busy}
            onSubmit={({ source, target, text }) =>
              runComparison(buildTranslationPrompt({ source, target, text }))
            }
            onExit={() => setMode("chat")}
          />
        )}
      </section>

      {hasResults && (
        <section className="mt-10">
          <ComparePanel grpo={grpo} prod={prod} />
        </section>
      )}

      <footer className="mt-auto pt-12 text-center text-xs text-muted-foreground">
        Sunbird AI · Sunflower GRPO test harness
      </footer>
    </main>
  );
}
