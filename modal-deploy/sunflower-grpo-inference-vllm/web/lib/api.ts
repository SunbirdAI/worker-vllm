export interface StreamHandlers {
  onDelta: (chunk: string) => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
  signal?: AbortSignal;
}

export async function streamGenerate(
  instruction: string,
  handlers: StreamHandlers,
  opts: {
    temperature?: number;
    maxTokens?: number;
    endpoint?: string;
  } = {},
): Promise<void> {
  const {
    temperature = 0.2,
    maxTokens = 1024,
    endpoint = "/generate_stream",
  } = opts;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      instruction,
      temperature,
      max_tokens: maxTokens,
    }),
    signal: handlers.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`stream request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLine = event
          .split("\n")
          .find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        const payload = dataLine.slice(5).trim();
        if (payload === "[DONE]") {
          handlers.onDone?.();
          return;
        }
        try {
          const parsed = JSON.parse(payload);
          if (typeof parsed?.delta === "string") {
            handlers.onDelta(parsed.delta);
          } else if (typeof parsed?.response === "string") {
            handlers.onDelta(parsed.response);
          }
        } catch {
          // non-JSON data line — skip
        }
      }
    }
    handlers.onDone?.();
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    handlers.onError?.(err as Error);
  } finally {
    reader.releaseLock();
  }
}

export interface ProductionResponse {
  response?: string;
  [key: string]: unknown;
}

export async function generateProduction({
  instruction,
  modelType = "qwen",
  temperature = 0.6,
  systemMessage = "",
  signal,
}: {
  instruction: string;
  modelType?: string;
  temperature?: number;
  systemMessage?: string;
  signal?: AbortSignal;
}): Promise<ProductionResponse> {
  const response = await fetch("/generate_production", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      instruction,
      model_type: modelType,
      temperature,
      system_message: systemMessage,
    }),
    signal,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `production request failed (${response.status})${body ? `: ${body}` : ""}`,
    );
  }

  return (await response.json()) as ProductionResponse;
}
