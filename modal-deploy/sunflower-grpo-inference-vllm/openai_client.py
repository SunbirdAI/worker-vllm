"""Client for the Sunflower-14B OpenAI-compatible vLLM deployment on Modal.

Usage:
    # one-shot with streaming (default)
    uv run python openai_client.py --prompt "Translate to luganda: Good morning"

    # disable streaming
    uv run python openai_client.py --no-stream --prompt "Who is Sunbird AI?"

    # interactive chat
    uv run python openai_client.py --chat

    # override model / workspace / api key
    uv run python openai_client.py --model sunflower-14b --api-key "$VLLM_API_KEY"

The client auto-resolves the Modal URL:
    https://<workspace>[-<env>]--<app-name>-<function-name>.modal.run/v1
and auto-loads VLLM_API_KEY from `.env` (next to this file) if --api-key is not set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import modal
from openai import OpenAI


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"


DEFAULT_SYSTEM_PROMPT = (
    "You are Sunflower, a helpful assistant made by Sunbird AI who understands "
    "all Ugandan languages. You specialise in accurate translations, "
    "explanations, summaries and other language tasks."
)
DEFAULT_PROMPT = "Translate to luganda: I am watching an Arsenal game right now."


def load_dotenv_key(name: str) -> str | None:
    """Minimal .env reader: return the value for `name` if present in ./.env."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def get_completion(client: OpenAI, model_id: str, messages: list[dict], args):
    completion_args = {
        "model": model_id,
        "messages": messages,
        "frequency_penalty": args.frequency_penalty,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
        "stop": args.stop,
        "stream": args.stream,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    completion_args = {k: v for k, v in completion_args.items() if v is not None}
    try:
        return client.chat.completions.create(**completion_args)
    except Exception as e:
        print(Colors.RED, f"Error during API call: {e}", Colors.END, sep="")
        return None


def main():
    parser = argparse.ArgumentParser(description="Sunflower-14B OpenAI client")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id to use. Defaults to the first model advertised by /v1/models.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Modal workspace (defaults to current profile).",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Modal environment (defaults to current environment).",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="sunflower-14b-openai",
        help="Modal App name serving the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--function-name",
        type=str,
        default="serve",
        help="Modal Function name. Append '-dev' for a `modal serve`d function.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override the resolved base URL (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="VLLM_API_KEY. Defaults to env VLLM_API_KEY or the value in ./.env.",
    )

    # Completion parameters
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--frequency-penalty", type=float, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions. Streaming / chat mode require n=1.",
    )
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Prompting
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    # UI options
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming of response chunks.",
    )
    parser.add_argument(
        "--chat", action="store_true", help="Enable interactive chat mode."
    )

    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("VLLM_API_KEY")
        or load_dotenv_key("VLLM_API_KEY")
        or "EMPTY"
    )

    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        workspace = args.workspace or modal.config._profile
        environment = args.environment or modal.config.config.get("environment")
        prefix = workspace + (f"-{environment}" if environment else "")
        base_url = (
            f"https://{prefix}--{args.app_name}-{args.function_name}.modal.run/v1"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model:
        model_id = args.model
        print(
            Colors.BOLD,
            f"Using model {model_id}. First call may trigger a cold boot.",
            Colors.END,
            sep="",
        )
    else:
        print(
            Colors.BOLD,
            f"Listing models at {client.base_url} ...",
            Colors.END,
            sep="",
        )
        model = client.models.list().data[0]
        model_id = model.id
        print(Colors.BOLD, f"Using {model_id}", Colors.END, sep="")

    messages = [{"role": "system", "content": args.system_prompt}]
    print(Colors.GRAY + f"[system]: {args.system_prompt}" + Colors.END)

    if args.chat:
        print(
            Colors.GREEN
            + Colors.BOLD
            + "\nEntering chat mode. Type 'bye' to end the conversation."
            + Colors.END
        )
        MAX_HISTORY = 10
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in {"bye", "exit", "quit"}:
                break

            if len(messages) > MAX_HISTORY:
                messages = messages[:1] + messages[-MAX_HISTORY + 1 :]

            messages.append({"role": "user", "content": user_input})
            response = get_completion(client, model_id, messages, args)
            if not response:
                continue

            print(Colors.BLUE + "\n[bot]: ", end="")
            if args.stream:
                assistant_message = ""
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        assistant_message += delta
                print(Colors.END)
            else:
                assistant_message = response.choices[0].message.content
                print(assistant_message + Colors.END)

            messages.append({"role": "assistant", "content": assistant_message})
    else:
        messages.append({"role": "user", "content": args.prompt})
        print(Colors.GREEN + f"\nYou: {args.prompt}" + Colors.END)
        response = get_completion(client, model_id, messages, args)
        if not response:
            return

        if args.stream:
            print(Colors.BLUE + "\n[bot]: ", end="")
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
            print(Colors.END)
        else:
            for i, choice in enumerate(response.choices):
                label = f"[bot] Choice {i + 1}: " if len(response.choices) > 1 else "[bot]: "
                print(Colors.BLUE + f"\n{label}{choice.message.content}" + Colors.END)


if __name__ == "__main__":
    main()
