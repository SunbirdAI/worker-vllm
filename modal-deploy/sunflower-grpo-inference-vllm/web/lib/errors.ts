export function friendlyError(err: Error | string | undefined): string {
  const msg = typeof err === "string" ? err : err?.message ?? "";
  const name = typeof err === "string" ? "" : err?.name ?? "";

  if (name === "AbortError") return "Request was cancelled.";

  if (/failed to fetch|load failed|networkerror|network request failed|enotfound|econnrefused|econnreset/i.test(msg)) {
    return "Couldn't reach the server. Check your connection and try again.";
  }
  if (/timeout|timed out|etimedout|deadline exceeded/i.test(msg)) {
    return "The request took too long to respond. Please try again.";
  }
  if (/\b429\b|too many requests/i.test(msg)) {
    return "Too many requests right now. Wait a moment and try again.";
  }
  if (/\b5\d\d\b/.test(msg)) {
    return "The server hit an error. Please try again in a moment.";
  }
  if (/\b4\d\d\b/.test(msg)) {
    return "The request was rejected by the server. Please try again.";
  }
  return "Something went wrong. Please try again.";
}
