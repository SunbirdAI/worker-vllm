"""Synthesize a large chunk of text by splitting into sentences,
batching the calls, and stitching the WAVs together locally.

Why this approach:
- Each /tts call has a max_tokens cap; very long inputs risk truncated audio.
- vLLM does continuous batching on the server — N sentences in one
  /tts/batch call is much faster than N sequential /tts calls.
- Per-sentence chunks keep each generation safely within token bounds
  and give you natural points to insert breath silence.

Output: a single concatenated 24 kHz mono 16-bit PCM WAV.

Usage:
    # Inline text:
    python orpheus-3B/api/scripts/test_long_text.py \\
      --speaker salt_eng_0002 --language eng \\
      --text "Your long paragraph here. It can span many sentences."

    # From a file:
    python orpheus-3B/api/scripts/test_long_text.py \\
      --speaker salt_eng_0002 --language eng \\
      --file path/to/article.txt \\
      --out /tmp/long.wav

Env vars:
    API_BASE         (default http://localhost:8000)
    BATCH_SIZE       (default 8, must be <= server MAX_BATCH_SIZE)
    SILENCE_MS       (default 250) — silence inserted between sentences
"""

import argparse
import io
import os
import re
import sys
import time
import wave
from pathlib import Path

import httpx

BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
SILENCE_MS = int(os.environ.get("SILENCE_MS", "250"))

# Hard guard: the model itself caps text at 2000 chars per item.
# We further enforce a softer per-sentence limit so generation never
# pushes against max_tokens on a single chunk.
PER_ITEM_CHAR_LIMIT = 500


# ----- text chunking -----

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for prose, no NLTK dependency.

    Splits on `.`, `!`, `?` followed by whitespace and a capital letter.
    Long sentences are further split on commas/semicolons if they exceed
    PER_ITEM_CHAR_LIMIT, so no single chunk is too big for the model.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw = _SENTENCE_BOUNDARY.split(text)
    out: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        while len(s) > PER_ITEM_CHAR_LIMIT:
            # Try a clause boundary first.
            split_idx = s.rfind(", ", 0, PER_ITEM_CHAR_LIMIT)
            if split_idx < PER_ITEM_CHAR_LIMIT // 2:
                split_idx = s.rfind("; ", 0, PER_ITEM_CHAR_LIMIT)
            if split_idx < PER_ITEM_CHAR_LIMIT // 2:
                split_idx = s.rfind(" ", 0, PER_ITEM_CHAR_LIMIT)
            if split_idx <= 0:
                split_idx = PER_ITEM_CHAR_LIMIT
            out.append(s[: split_idx + 1].strip())
            s = s[split_idx + 1 :].strip()
        if s:
            out.append(s)
    return out


def chunked(items: list, n: int) -> list[list]:
    return [items[i : i + n] for i in range(0, len(items), n)]


# ----- WAV helpers (24 kHz mono 16-bit PCM, matches Orpheus output) -----

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes


def read_wav_pcm(data: bytes) -> bytes:
    """Read a WAV and return the raw PCM frames. Assert the expected format."""
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == CHANNELS, f"channels={w.getnchannels()}"
        assert w.getsampwidth() == SAMPLE_WIDTH, f"sampwidth={w.getsampwidth()}"
        assert w.getframerate() == SAMPLE_RATE, f"sr={w.getframerate()}"
        return w.readframes(w.getnframes())


def silence_pcm(ms: int) -> bytes:
    """Return `ms` milliseconds of silence as 16-bit PCM frames."""
    n_frames = SAMPLE_RATE * ms // 1000
    return b"\x00\x00" * n_frames


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


# ----- API calls -----

def call_batch(
    client: httpx.Client, items: list[dict], n_done: int, n_total: int
) -> list[dict]:
    print(f"  [{n_done + 1}-{n_done + len(items)} / {n_total}] POST /tts/batch...")
    t0 = time.monotonic()
    r = client.post("/tts/batch", json={"items": items})
    r.raise_for_status()
    body = r.json()
    elapsed = time.monotonic() - t0
    t = body["timings_ms"]
    print(
        f"      wall {elapsed:.1f}s  | server inference={t['inference_ms']:.0f}ms "
        f"upload={t['upload_ms']:.0f}ms total={t['total_ms']:.0f}ms"
    )
    return body["results"]


def download_audio(client: httpx.Client, url: str) -> bytes:
    r = client.get(url)
    r.raise_for_status()
    return r.content


# ----- main -----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Inline text to synthesize.")
    src.add_argument("--file", type=Path, help="Path to a text file.")
    ap.add_argument("--speaker", default="salt_eng_0002")
    ap.add_argument("--language", default="eng")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/orpheus-long.wav"),
        help="Output WAV path (default %(default)s).",
    )
    args = ap.parse_args()

    text = args.text if args.text else args.file.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    if not sentences:
        print("no sentences found", file=sys.stderr)
        return 2

    print(f"API base:        {BASE}")
    print(f"Speaker / lang:  {args.speaker} / {args.language}")
    print(f"Total chars:     {len(text):,}")
    print(f"Sentences:       {len(sentences)}")
    print(f"Batches:         {-(-len(sentences) // BATCH_SIZE)} of up to {BATCH_SIZE}")
    print(f"Silence between: {SILENCE_MS} ms")
    print(f"Output WAV:      {args.out}")
    print()

    base_item = {
        "speaker_id": args.speaker,
        "language": args.language,
    }
    items_all = [{**base_item, "text": s} for s in sentences]

    # 300s budget covers Modal cold start. Subsequent batches are fast.
    timeout = httpx.Timeout(300.0, connect=10.0)
    pcm_parts: list[bytes] = []
    failures = 0

    with httpx.Client(base_url=BASE, timeout=timeout) as client:
        # Liveness check up front.
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"server not reachable at {BASE}: {exc}", file=sys.stderr)
            return 1

        t_wall = time.monotonic()
        done = 0
        for batch in chunked(items_all, BATCH_SIZE):
            results = call_batch(client, batch, done, len(items_all))
            # results come back in original order; rely on that to stitch.
            for res in results:
                idx = done + res["index"]
                if res["status"] != "ok":
                    failures += 1
                    print(
                        f"      [{idx}] FAIL  code={res.get('error_code')}  "
                        f"detail={res.get('error_detail')}",
                        file=sys.stderr,
                    )
                    continue
                wav_bytes = download_audio(client, res["audio_url"])
                pcm_parts.append(read_wav_pcm(wav_bytes))
                if SILENCE_MS > 0:
                    pcm_parts.append(silence_pcm(SILENCE_MS))
            done += len(batch)

        wall = time.monotonic() - t_wall

    # Drop the trailing silence so the file doesn't end on a gap.
    if pcm_parts and SILENCE_MS > 0:
        pcm_parts.pop()

    if not pcm_parts:
        print("no audio produced", file=sys.stderr)
        return 1

    full_pcm = b"".join(pcm_parts)
    write_wav(args.out, full_pcm)
    duration_sec = len(full_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)

    print()
    print(f"OK: {len(sentences) - failures}/{len(sentences)} sentences synthesized")
    print(f"    wall time:    {wall:.1f}s")
    print(f"    audio length: {duration_sec:.2f}s")
    print(f"    output:       {args.out}  ({args.out.stat().st_size:,} bytes)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
