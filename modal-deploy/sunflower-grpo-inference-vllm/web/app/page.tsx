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
  type ProdMode,
} from "@/components/compare-panel";
import { TemperatureSlider } from "@/components/temperature-slider";
import { buildTranslationPrompt } from "@/lib/prompt";
import { generateProduction, streamGenerate } from "@/lib/api";

type Mode = "chat" | "translate";

const EMPTY: PanelState = { text: "", status: "idle" };
const CHAT_TEMP = 0.6;
const TRANSLATE_TEMP = 0.1;

export default function Page() {
  const [mode, setMode] = useState<Mode>("chat");
  const [grpo, setGrpo] = useState<PanelState>(EMPTY);
  const [prod, setProd] = useState<PanelState>(EMPTY);
  const [grpoBusy, setGrpoBusy] = useState(false);
  const [prodBusy, setProdBusy] = useState(false);
  const [prodMode, setProdMode] = useState<ProdMode>("stream");
  const [temperature, setTemperature] = useState<number>(CHAT_TEMP);
  const [lastInstruction, setLastInstruction] = useState<string | null>(null);
  const grpoCtrlRef = useRef<AbortController | null>(null);
  const prodCtrlRef = useRef<AbortController | null>(null);

  const busy = grpoBusy || prodBusy;

  function handleModeChange(next: Mode) {
    if (next === mode) return;
    setMode(next);
    setTemperature(next === "translate" ? TRANSLATE_TEMP : CHAT_TEMP);
  }

  async function runGrpo(instruction: string, temp: number) {
    grpoCtrlRef.current?.abort();
    const controller = new AbortController();
    grpoCtrlRef.current = controller;

    setGrpoBusy(true);
    setGrpo({ text: "", status: "loading" });
    const started = performance.now();

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
              latencyMs: performance.now() - started,
            }));
          },
          onError: (err) => {
            setGrpo({ text: "", status: "error", error: err.message });
          },
        },
        { temperature: temp, maxTokens: 1024 },
      );
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setGrpo({
        text: "",
        status: "error",
        error: (err as Error)?.message ?? "unknown error",
      });
    } finally {
      setGrpoBusy(false);
    }
  }

  async function runProd(
    instruction: string,
    temp: number,
    forceMode?: ProdMode,
  ) {
    prodCtrlRef.current?.abort();
    const controller = new AbortController();
    prodCtrlRef.current = controller;

    const runMode = forceMode ?? prodMode;
    setProdBusy(true);
    setProd({ text: "", status: "loading" });
    const started = performance.now();

    try {
      if (runMode === "stream") {
        await streamGenerate(
          instruction,
          {
            signal: controller.signal,
            onDelta: (chunk) => {
              setProd((s) => ({
                ...s,
                text: s.text + chunk,
                status: "loading",
              }));
            },
            onDone: () => {
              setProd((s) => ({
                ...s,
                status: "done",
                latencyMs: performance.now() - started,
              }));
            },
            onError: (err) => {
              setProd({ text: "", status: "error", error: err.message });
            },
          },
          {
            temperature: temp,
            maxTokens: 1024,
            endpoint: "/generate_openai_stream",
          },
        );
      } else {
        const data = await generateProduction({
          instruction,
          temperature: temp,
          signal: controller.signal,
        });
        const text =
          typeof data?.response === "string"
            ? data.response
            : JSON.stringify(data, null, 2);
        setProd({
          text,
          status: "done",
          latencyMs: performance.now() - started,
        });
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setProd({
        text: "",
        status: "error",
        error: (err as Error)?.message ?? "unknown error",
      });
    } finally {
      setProdBusy(false);
    }
  }

  function handleProdModeChange(next: ProdMode) {
    if (next === prodMode) return;
    setProdMode(next);
    if (lastInstruction && !prodBusy) {
      runProd(lastInstruction, temperature, next);
    }
  }

  async function runComparison(instruction: string) {
    setLastInstruction(instruction);
    await Promise.all([
      runGrpo(instruction, temperature),
      runProd(instruction, temperature),
    ]);
  }

  const retryGrpo = lastInstruction
    ? () => runGrpo(lastInstruction, temperature)
    : undefined;
  const retryProd = lastInstruction
    ? () => runProd(lastInstruction, temperature)
    : undefined;

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
            onEnterTranslate={() => handleModeChange("translate")}
          />
        ) : (
          <TranslateForm
            busy={busy}
            onSubmit={({ source, target, text }) =>
              runComparison(buildTranslationPrompt({ source, target, text }))
            }
            onExit={() => handleModeChange("chat")}
          />
        )}
      </section>

      <section className="mt-3">
        <TemperatureSlider
          value={temperature}
          onChange={setTemperature}
          hint={
            mode === "translate"
              ? "0.1 recommended for translations"
              : "0.6 recommended for general prompts"
          }
        />
      </section>

      {hasResults && (
        <section className="mt-10">
          <ComparePanel
            grpo={grpo}
            prod={prod}
            onRetryGrpo={retryGrpo}
            onRetryProd={retryProd}
            retryDisabled={busy}
            prodMode={prodMode}
            onProdModeChange={handleProdModeChange}
            prodModeDisabled={prodBusy}
          />
        </section>
      )}

      <footer className="mt-auto pt-12 text-center text-xs text-muted-foreground">
        Sunbird AI · Sunflower GRPO test harness
      </footer>
    </main>
  );
}
