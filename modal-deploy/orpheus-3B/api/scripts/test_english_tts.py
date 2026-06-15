"""Hit the running FastAPI service against an English speaker.

Tests:
- POST /tts with one sentence
- POST /tts/batch with five sentences

Speaker: salt_eng_0002 (language: eng).

Usage:
    # API running locally:
    python orpheus-3B/api/scripts/test_english_tts.py

    # Different host:
    API_BASE=http://host:8000 python orpheus-3B/api/scripts/test_english_tts.py

    # Also download the WAVs to /tmp:
    DOWNLOAD=1 python orpheus-3B/api/scripts/test_english_tts.py
"""

import os
import sys
from pathlib import Path

import httpx

BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
DOWNLOAD = os.environ.get("DOWNLOAD") in ("1", "true", "yes")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/tmp/orpheus-english-tts"))

SPEAKER_ID = "salt_eng_0002"
LANGUAGE = "eng"

SINGLE_TEXT = "Good morning, my name is Patrick and I work on text to speech systems."

BATCH_TEXTS = [
    "The sun rose over the misty hills as the village began to wake.",
    "Please bring me a glass of water and the morning newspaper.",
    "Artificial intelligence has changed how we build software products.",
    "She laughed at the joke and then quickly returned to her work.",
    "Travel safely and remember to call us when you arrive home.",
]


def _short(url: str, n: int = 90) -> str:
    return url if len(url) <= n else url[: n - 3] + "..."


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    print(f"      saved → {dest}  ({dest.stat().st_size:,} bytes)")


def test_single(client: httpx.Client) -> dict:
    print(f"\n--> POST /tts  (speaker={SPEAKER_ID}, language={LANGUAGE})")
    print(f"    text: {SINGLE_TEXT!r}")
    r = client.post(
        "/tts",
        json={
            "text": SINGLE_TEXT,
            "speaker_id": SPEAKER_ID,
            "language": LANGUAGE,
        },
    )
    r.raise_for_status()
    body = r.json()
    t = body["timings_ms"]
    print(f"    request_id:      {body['request_id']}")
    print(f"    duration_sec:    {body['duration_seconds']:.2f}")
    print(f"    audio_size:      {body['audio_size_bytes']:,} bytes")
    print(f"    gcs_object:      {body['gcs_object']}")
    print(f"    audio_url:       {_short(body['audio_url'])}")
    print(f"    expires_at:      {body['audio_url_expires_at']}")
    print(
        f"    timings_ms:      inference={t['inference_ms']:.0f}  "
        f"upload={t['upload_ms']:.0f}  sign={t['signed_url_ms']:.0f}  "
        f"total={t['total_ms']:.0f}"
    )
    if DOWNLOAD:
        _download(client, body["audio_url"], OUT_DIR / "single.wav")
    return body


def test_batch(client: httpx.Client) -> dict:
    print(f"\n--> POST /tts/batch  ({len(BATCH_TEXTS)} items, speaker={SPEAKER_ID})")
    for i, text in enumerate(BATCH_TEXTS):
        print(f"    [{i}] {text!r}")
    r = client.post(
        "/tts/batch",
        json={
            "items": [
                {"text": text, "speaker_id": SPEAKER_ID, "language": LANGUAGE}
                for text in BATCH_TEXTS
            ]
        },
    )
    r.raise_for_status()
    body = r.json()
    t = body["timings_ms"]
    print(f"    request_id:      {body['request_id']}")
    print(
        f"    timings_ms:      inference={t['inference_ms']:.0f}  "
        f"upload={t['upload_ms']:.0f}  total={t['total_ms']:.0f}"
    )
    for item in body["results"]:
        idx = item["index"]
        if item["status"] != "ok":
            print(
                f"    [{idx}] ERROR  code={item.get('error_code')}  "
                f"detail={item.get('error_detail')}"
            )
            continue
        print(
            f"    [{idx}] ok  dur={item['duration_seconds']:.2f}s  "
            f"size={item['audio_size_bytes']:,}  "
            f"obj={item['gcs_object']}"
        )
        print(f"        url: {_short(item['audio_url'])}")
        if DOWNLOAD:
            _download(client, item["audio_url"], OUT_DIR / f"batch_{idx}.wav")
    return body


def main() -> int:
    print(f"API base: {BASE}")
    timeout = httpx.Timeout(300.0, connect=10.0)
    with httpx.Client(base_url=BASE, timeout=timeout) as client:
        # Quick liveness check up front so we fail fast on a misconfigured server.
        try:
            health = client.get("/health")
            health.raise_for_status()
            print(f"health: {health.json()}")
        except httpx.HTTPError as exc:
            print(f"server not reachable at {BASE}: {exc}")
            return 1

        test_single(client)
        test_batch(client)

    if DOWNLOAD:
        print(f"\nWAVs saved under {OUT_DIR}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
