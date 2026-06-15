"""Manual smoke test — hits a live API at $API_BASE (default http://localhost:8000).

Usage:
    API_BASE=http://localhost:8000 python orpheus-3B/api/scripts/smoke.py
"""

import os
import sys

import httpx


BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")


def main() -> int:
    with httpx.Client(base_url=BASE, timeout=httpx.Timeout(200.0, connect=10.0)) as c:
        print(f"--> GET /health")
        print(c.get("/health").json())

        print(f"--> GET /healthz")
        print(c.get("/healthz").json())

        print(f"--> GET /speakers")
        speakers = c.get("/speakers").json()
        print(f"  total: {speakers['total']}, languages: {speakers['languages']}")

        print(f"--> POST /tts")
        r = c.post(
            "/tts",
            json={"text": "Mwattu, oli otya?", "speaker_id": "salt_lug_0001", "language": "lug"},
        )
        r.raise_for_status()
        body = r.json()
        print(f"  audio_url: {body['audio_url'][:80]}...")
        print(f"  duration:  {body['duration_seconds']:.2f}s")
        print(f"  timings:   {body['timings_ms']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
