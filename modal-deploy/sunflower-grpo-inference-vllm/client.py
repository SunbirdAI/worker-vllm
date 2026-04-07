"""
Client for the deployed Sunflower GRPO vLLM Modal app.

Two transports are supported:

1. HTTP (FastAPI endpoints exposed by `web` ASGI app):
       python client.py http "Translate to luganda: Good morning"
       python client.py stream "Translate to luganda: Good morning"

2. Direct Modal class call (no HTTP, faster locally if you have modal CLI auth):
       python client.py modal "Translate to luganda: Good morning"

The HTTP base URL is auto-discovered via `modal app list` style URL convention,
or you can set SUNFLOWER_URL explicitly.
"""

from __future__ import annotations

import json
import os
import sys

import requests


def _base_url() -> str:
    url = os.environ.get("SUNFLOWER_URL")
    if not url:
        raise SystemExit(
            "Set SUNFLOWER_URL to the deployed `web` endpoint, e.g.\n"
            "  export SUNFLOWER_URL=https://<workspace>--sunflower-grpo-vllm-web.modal.run"
        )
    return url.rstrip("/")


def call_http(instruction: str, temperature: float = 0.6) -> None:
    r = requests.post(
        f"{_base_url()}/generate",
        json={"instruction": instruction, "temperature": temperature},
        timeout=300,
    )
    if not r.ok:
        print(f"HTTP {r.status_code}: {r.text}")
        r.raise_for_status()
    print(r.json()["response"])


def call_stream(instruction: str, temperature: float = 0.6) -> None:
    with requests.post(
        f"{_base_url()}/generate_stream",
        json={"instruction": instruction, "temperature": temperature},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                print(json.loads(payload)["delta"], end="", flush=True)
            except json.JSONDecodeError:
                pass
        print()


def call_modal(instruction: str, temperature: float = 0.6) -> None:
    """Invoke the deployed Modal class directly via the Modal SDK."""
    import modal

    cls = modal.Cls.from_name("sunflower-grpo-vllm", "SunflowerVLLM")
    print(cls().generate.remote(instruction, temperature))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python client.py [http|stream|modal] '<instruction>' [temperature]")
        sys.exit(1)

    mode = sys.argv[1]
    instruction = sys.argv[2]
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

    if mode == "http":
        call_http(instruction, temperature)
    elif mode == "stream":
        call_stream(instruction, temperature)
    elif mode == "modal":
        call_modal(instruction, temperature)
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
