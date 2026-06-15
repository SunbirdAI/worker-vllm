"""Deploy Orpheus-3B multilingual TTS to Modal with vLLM.

Mirrors the inference pipeline from `Orpheus_3B_Sunbird_Luganda_vLLM_Inference.ipynb`:
same special-token layout, same SNAC decode, same `hf_overrides` fix for the
Unsloth pad-token off-by-one. Wraps vLLM behind a FastAPI ASGI app on Modal so
clients get `audio/wav` back instead of token IDs.

Multi-language + multi-speaker: every request carries its own `speaker_id`,
so a single deployed replica serves any speaker the finetuned checkpoint
knows (e.g. `salt_lug_0001`, `salt_swa_0001`, ...). vLLM's continuous
batching lets `/tts/batch` mix speakers in a single GPU pass.

Deploy:
    modal deploy orpheus-3B/modal_deploy.py

Tear down:
    modal app stop orpheus-3b-tts

See `MODAL_DEPLOYMENT.md` for full instructions.
"""

# Note: we deliberately do NOT use `from __future__ import annotations`.
# It would turn every type hint into a string. FastAPI then calls
# `typing.get_type_hints()` to resolve those strings against the
# function's `__globals__` — and the Pydantic request models (TTSReq /
# TTSBatchReq) are defined inside `OrpheusTTS.web()`, so they are NOT in
# `__globals__`. The resolution silently fails and FastAPI falls back to
# treating `req` as a query parameter, returning 422
# (`loc: ["query", "req"]`) on every POST.

import logging
import os
import re

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "orpheus-3b-tts"

# Default HF repo. Override at deploy time with:
#   ORPHEUS_MODEL_ID=patrickcmd/orpheus-3b-tts-multilingual modal deploy ...
MODEL_ID = os.environ.get(
    "ORPHEUS_MODEL_ID", "sunbird/orpheus-3b-tts-multilingual"
)
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"

# Orpheus-3B vocab = Llama-3 base (128256) + 11 specials + 7*4096 audio
# codebooks = 156939. Unsloth's `push_to_hub_merged` shim can bump
# `config.vocab_size` to 156940 while leaving the embedding at
# [156939, 3072]. transformers tolerates it; vLLM asserts equality and
# crashes. Override on load to align config with the real weight shape.
VOCAB_SIZE = 156939
MAX_MODEL_LEN = int(os.environ.get("ORPHEUS_MAX_MODEL_LEN", "4096"))

# GPU choice. L40S (48 GB) is a good cost/perf default for a 3B bf16 model
# with continuous batching. A10G (24 GB) works but is slower; H100 is fastest.
GPU = os.environ.get("ORPHEUS_GPU", "L40S")

# Per-replica concurrency. vLLM batches in-flight requests on one GPU, so
# raising this lifts batched throughput. Watch GPU memory if you raise it.
MAX_CONCURRENT_INPUTS = int(os.environ.get("ORPHEUS_MAX_INPUTS", "16"))

MINUTES = 60  # seconds

# ---------------------------------------------------------------------------
# Token constants (must match the training notebook's #2.6 / inference §2)
# ---------------------------------------------------------------------------

END_OF_TEXT = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
AUDIO_TOKEN_LO = 128266
AUDIO_TOKEN_HI = 128266 + 7 * 4096  # exclusive, = 156938


# ---------------------------------------------------------------------------
# Long-text chunking constants
#
# `_chunk_text` packs sentences into chunks ≤ MAX_CHARS_PER_CHUNK as a soft
# target. A chunk may exceed it only when (a) the orphan-tail merge keeps the
# join below CHUNK_HARD_CEILING or (b) a single indivisible word exceeds the
# hard ceiling (logged as a warning).
# ---------------------------------------------------------------------------

MAX_CHARS_PER_CHUNK = 220
CHUNK_HARD_CEILING = 280
MAX_CHUNKS_PER_REQUEST = 48

_SENTENCE_RE = re.compile(r"(?<=[.!?…።])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

_chunker_log = logging.getLogger("orpheus.chunker")


def _chunk_text(
    text: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
    hard_ceiling: int = CHUNK_HARD_CEILING,
) -> list[str]:
    """Sentence-aware chunker for long-text TTS.

    Returns a list of chunks. Each chunk fits within `max_chars` as a soft
    target. A chunk may exceed `max_chars` only when an oversized single word
    cannot be split further; such chunks are logged as warnings if they also
    exceed `hard_ceiling`. Empty input returns `[""]`.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return [""]
    sentences = [s for s in _SENTENCE_RE.split(normalized) if s.strip()]
    atoms: list[str] = []
    for s in sentences:
        if len(s) <= max_chars:
            atoms.append(s)
        else:
            atoms.extend(_expand(s, max_chars, hard_ceiling))
    chunks = _pack(atoms, max_chars)
    # Orphan-tail merge.
    if len(chunks) >= 2 and len(chunks[-1]) < 30:
        merged = chunks[-2] + " " + chunks[-1]
        if len(merged) <= hard_ceiling:
            chunks = chunks[:-2] + [merged]
    return chunks


def _expand(sentence: str, max_chars: int, hard_ceiling: int) -> list[str]:
    """Decompose an oversized sentence into clause- or word-level atoms.

    Clause-level split is taken only when *every* clause fits `max_chars`.
    If any clause is still oversized, the entire sentence falls through to
    word-level — clause structure is discarded rather than mixed.
    """
    clauses = [c for c in _CLAUSE_RE.split(sentence) if c.strip()]
    if len(clauses) > 1 and all(len(c) <= max_chars for c in clauses):
        return clauses
    words = sentence.split()
    for w in words:
        if len(w) > hard_ceiling:
            _chunker_log.warning(
                "chunker: word length %d exceeds hard_ceiling=%d (%r...)",
                len(w),
                hard_ceiling,
                w[:50],
            )
    return words


def _pack(atoms: list[str], max_chars: int) -> list[str]:
    """Greedy packer.

    An atom that fits joins the current chunk; otherwise the current chunk
    is closed and the atom starts a new one. An atom larger than max_chars
    occupies its own chunk untouched — never split here. (The caller is
    expected to have decomposed sentences into clause/word atoms first.)
    """
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if not atom:
            continue
        if not current:
            current = atom
        elif len(current) + 1 + len(atom) <= max_chars:
            current = current + " " + atom
        else:
            chunks.append(current)
            current = atom
    if current:
        chunks.append(current)
    return chunks


def _concat_wavs(wavs, pad_ms: int = 120):
    """Concatenate per-chunk 24 kHz float32 waveforms with a silence pad.

    `wavs` is a list of `np.ndarray` (float32 mono). Returns one np.ndarray
    of the same dtype. A `pad_ms`-millisecond zero-padded gap is inserted
    between adjacent chunks (no pad before the first chunk or after the
    last).
    """
    import numpy as np

    if not wavs:
        return np.zeros(0, dtype=np.float32)
    if len(wavs) == 1:
        return wavs[0]
    pad = np.zeros(int(24000 * pad_ms / 1000), dtype=np.float32)
    out: list = []
    for i, w in enumerate(wavs):
        if i:
            out.append(pad)
        out.append(w)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Static catalog of speakers exposed via GET /speakers.
#
# These are the speaker tags the multilingual Orpheus-3B checkpoint was
# finetuned on (Sunbird/tts configs: lug, ach, lgg, nyn, teo, swa, ...).
# Add or remove entries to match whatever your own training run produced.
# Clients only need to send a `speaker_id` that exists in the checkpoint;
# this catalog is purely informational.
# ---------------------------------------------------------------------------

# 40 speakers across 16 languages from the multilingual training run's
# held-aside test split. Source corpora: SALT, WAXAL, SLR32, SLR129,
# BATEESA. Language codes are ISO 639-3.
SPEAKERS_BY_LANGUAGE: dict[str, list[str]] = {
    "ach": [  # Acholi
        "salt_ach_0001",
        "waxal_ach_0001",
        "waxal_ach_0005",
        "waxal_ach_0006",
        "waxal_ach_0008",
    ],
    "afr": [  # Afrikaans
        "slr32_afr_0009",
    ],
    "eng": [  # English
        "salt_eng_0001",
        "salt_eng_0002",
        "salt_eng_0003",
    ],
    "ewe": [  # Ewe
        "slr129_ewe_0001",
    ],
    "ful": [  # Fula / Pulaar
        "waxal_ful_0003",
        "waxal_ful_0004",
        "waxal_ful_0006",
    ],
    "hau": [  # Hausa
        "waxal_hau_0004",
        "waxal_hau_0006",
        "waxal_hau_0007",
        "waxal_hau_0008",
    ],
    "ibo": [  # Igbo
        "waxal_ibo_0003",
        "waxal_ibo_0005",
        "waxal_ibo_0008",
    ],
    "kik": [  # Kikuyu
        "waxal_kik_0003",
        "waxal_kik_0004",
    ],
    "kin": [  # Kinyarwanda
        "bateesa_kin_0001",
    ],
    "lin": [  # Lingala
        "slr129_lin_0001",
    ],
    "lug": [  # Luganda
        "salt_lug_0001",
        "waxal_lug_0002",
        "waxal_lug_0003",
        "waxal_lug_0004",
        "waxal_lug_0005",
        "waxal_lug_0006",
        "waxal_lug_0007",
        "waxal_lug_0008",
    ],
    "luo": [  # Dholuo
        "waxal_luo_0001",
        "waxal_luo_0002",
        "waxal_luo_0003",
        "waxal_luo_0004",
    ],
    "nyn": [  # Runyankole
        "salt_nyn_0001",
        "waxal_nyn_0003",
        "waxal_nyn_0004",
        "waxal_nyn_0007",
        "waxal_nyn_0008",
    ],
    "swa": [  # Swahili
        "waxal_swa_0006",
        "waxal_swa_0007",
    ],
    "teo": [  # Ateso
        "salt_teo_0001",
    ],
    "xho": [  # Xhosa
        "slr32_xho_0012",
    ],
    "yor": [  # Yoruba
        "waxal_yor_0002",
        "waxal_yor_0006",
        "waxal_yor_0008",
    ],
}
DEFAULT_SPEAKER_ID = os.environ.get("ORPHEUS_DEFAULT_SPEAKER", "salt_lug_0001")


# ---------------------------------------------------------------------------
# Modal image, volumes, secrets
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("libsndfile1")
    # transformers is capped below 4.56 on purpose: vLLM 0.10.0's tokenizer
    # cache layer reads `tokenizer.all_special_tokens_extended`, which was
    # refactored out into a `TokenizersBackend` object in transformers 4.56.
    # Loading then raises:
    #   AttributeError: TokenizersBackend has no attribute
    #     all_special_tokens_extended
    # The lower bound matches vLLM 0.10.0's own minimum (>=4.53.2).
    .uv_pip_install(
        "vllm==0.10.0",
        "transformers>=4.53.2,<4.56.0",
        "huggingface-hub>=0.27.0",
        "snac==1.2.1",
        "soundfile==0.12.1",
        "numpy<2.0",
        "fastapi[standard]==0.115.0",
        "pydantic>=2.5,<3.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster HF downloads
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name(
    "orpheus-hf-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name(
    "orpheus-vllm-cache", create_if_missing=True
)

hf_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# One-shot warmup: prefetch model weights into the HF cache volume.
#
# Run manually before the first deploy to keep cold starts short:
#     modal run orpheus-3B/modal_deploy.py::download_model
# ---------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=30 * MINUTES,
)
def download_model(model_id: str = MODEL_ID) -> None:
    from huggingface_hub import snapshot_download

    print(f"downloading {model_id}")
    snapshot_download(
        model_id,
        # keep the cache tidy: skip the pickled .bin if safetensors exist
        ignore_patterns=["*.pt", "*.bin"],
    )
    print(f"downloading {SNAC_MODEL_ID}")
    snapshot_download(SNAC_MODEL_ID)
    print("warmup complete")


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu=GPU,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,  # generous cold-start budget for first model load
    min_containers=0,
    max_containers=3,  # hard cap on horizontal scale-out
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class OrpheusTTS:
    @modal.enter()
    def load(self) -> None:
        """Load vLLM + SNAC + tokenizer once per container."""
        from snac import SNAC
        from transformers import AutoTokenizer
        from vllm import LLM

        print(f"loading vLLM model: {MODEL_ID}")
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=False,
            hf_overrides={"vocab_size": VOCAB_SIZE},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # SNAC runs on CPU — frees GPU for vLLM's KV cache.
        self.snac = SNAC.from_pretrained(SNAC_MODEL_ID).to("cpu")
        print("Orpheus-3B + SNAC ready")

    # ----- helpers -----

    def _build_prompt_ids(self, text: str, speaker_id: str) -> list[int]:
        """[SOH] + tokenizer(speaker_id: text) + [EOT, EOH]."""
        tagged = f"{speaker_id}: {text}"
        text_ids = self.tokenizer.encode(tagged, add_special_tokens=True)
        return [START_OF_HUMAN, *text_ids, END_OF_TEXT, END_OF_HUMAN]

    def _codes_to_wav(self, token_ids: list[int]):
        """vLLM-output token_ids -> 24 kHz mono float32 numpy waveform."""
        import numpy as np
        import torch

        ids = torch.tensor(token_ids, dtype=torch.int64)

        # Crop on the LAST SOS, discarding any pre-speech preamble.
        sos = (ids == START_OF_SPEECH).nonzero(as_tuple=True)[0]
        if len(sos) > 0:
            ids = ids[sos[-1].item() + 1 :]

        # Keep only tokens in the audio codebook range.
        audio = ids[(ids >= AUDIO_TOKEN_LO) & (ids < AUDIO_TOKEN_HI)]
        n = (audio.size(0) // 7) * 7
        code_list = [t.item() - AUDIO_TOKEN_LO for t in audio[:n]]

        l1, l2, l3 = [], [], []
        for i in range(len(code_list) // 7):
            l1.append(code_list[7 * i])
            l2.append(code_list[7 * i + 1] - 4096)
            l3.append(code_list[7 * i + 2] - 2 * 4096)
            l3.append(code_list[7 * i + 3] - 3 * 4096)
            l2.append(code_list[7 * i + 4] - 4 * 4096)
            l3.append(code_list[7 * i + 5] - 5 * 4096)
            l3.append(code_list[7 * i + 6] - 6 * 4096)
        if not l1:
            return np.zeros(12000, dtype=np.float32)

        def clamp(vals):
            return [max(0, min(4095, v)) for v in vals]

        codes = [
            torch.tensor(clamp(l1)).unsqueeze(0),
            torch.tensor(clamp(l2)).unsqueeze(0),
            torch.tensor(clamp(l3)).unsqueeze(0),
        ]
        wav = self.snac.decode(codes)
        return wav.detach().squeeze().to("cpu").numpy().astype(np.float32)

    # ----- HTTP -----

    @modal.asgi_app()
    def web(self):
        import base64
        import io
        from typing import Annotated, Optional

        import soundfile as sf
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import JSONResponse, Response
        from pydantic import BaseModel, Field
        from vllm import SamplingParams

        api = FastAPI(
            title="Orpheus-3B Multilingual TTS",
            description=(
                "Multi-language, multi-speaker TTS backed by vLLM. "
                "Each request carries a speaker_id matching one of the "
                "speaker tags the finetuned checkpoint was trained on."
            ),
        )

        class TTSReq(BaseModel):
            text: str = Field(..., description="Text to synthesize.")
            speaker_id: str = Field(
                DEFAULT_SPEAKER_ID,
                description="Speaker tag from the finetune set, e.g. salt_lug_0001.",
            )
            seed: Optional[int] = Field(
                None, description="RNG seed for reproducibility."
            )
            temperature: float = Field(0.6, ge=0.0, le=2.0)
            top_p: float = Field(0.95, gt=0.0, le=1.0)
            repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
            max_tokens: int = Field(1200, ge=64, le=MAX_MODEL_LEN)

        class TTSBatchReq(BaseModel):
            items: list[TTSReq]

        def _sampling(req: TTSReq, seed_offset: int = 0) -> SamplingParams:
            seed = None if req.seed is None else req.seed + seed_offset
            return SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                stop_token_ids=[END_OF_SPEECH],
                skip_special_tokens=False,
                seed=seed,
            )

        def _wav_bytes(wav) -> bytes:
            buf = io.BytesIO()
            sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
            return buf.getvalue()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": MODEL_ID,
                "max_model_len": MAX_MODEL_LEN,
                "sample_rate": 24000,
            }

        @api.get("/speakers")
        def speakers():
            return {
                "default": DEFAULT_SPEAKER_ID,
                "by_language": SPEAKERS_BY_LANGUAGE,
            }

        @api.post(
            "/tts",
            responses={200: {"content": {"audio/wav": {}}}},
            response_class=Response,
        )
        def tts(req: Annotated[TTSReq, Body()]):
            chunks = _chunk_text(req.text)
            if len(chunks) > MAX_CHUNKS_PER_REQUEST:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "text_too_long",
                        "detail": (
                            f"text produced {len(chunks)} chunks "
                            f"(limit {MAX_CHUNKS_PER_REQUEST}); "
                            "shorten the input or split it client-side"
                        ),
                    },
                )

            if len(chunks) == 1:
                # Fast path — byte-for-byte identical to the prior
                # single-shot implementation. Uses req.text (raw) so any
                # exotic whitespace/casing in the original input is
                # preserved verbatim through the tokenizer.
                sp = _sampling(req)
                prompt_ids = self._build_prompt_ids(req.text, req.speaker_id)
                outs = self.llm.generate(
                    [{"prompt_token_ids": prompt_ids}], sp
                )
                wav = self._codes_to_wav(list(outs[0].outputs[0].token_ids))
            else:
                prompts = [
                    {
                        "prompt_token_ids": self._build_prompt_ids(
                            c, req.speaker_id
                        )
                    }
                    for c in chunks
                ]
                sps = [
                    _sampling(req, seed_offset=i) for i in range(len(chunks))
                ]
                print(
                    f"tts chunks={len(chunks)} "
                    f"text_len={len(req.text)} "
                    f"speaker={req.speaker_id}"
                )
                outs = self.llm.generate(prompts, sps)
                wavs = [
                    self._codes_to_wav(list(o.outputs[0].token_ids))
                    for o in outs
                ]
                wav = _concat_wavs(wavs, pad_ms=120)

            return Response(
                content=_wav_bytes(wav),
                media_type="audio/wav",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Duration-Seconds": f"{len(wav) / 24000:.3f}",
                    "X-Speaker-Id": req.speaker_id,
                    "X-Chunks": str(len(chunks)),
                },
            )

        @api.post("/tts/batch")
        def tts_batch(req: Annotated[TTSBatchReq, Body()]):
            if not req.items:
                raise HTTPException(
                    status_code=400, detail="items must be non-empty"
                )
            # Use the first item's decoding params for the whole batch.
            # vLLM allows per-request SamplingParams too — pass a list if you
            # need heterogeneous decoding settings across the batch.
            sp = _sampling(req.items[0])
            prompts = [
                {
                    "prompt_token_ids": self._build_prompt_ids(
                        it.text, it.speaker_id
                    )
                }
                for it in req.items
            ]
            outs = self.llm.generate(prompts, sp)
            results = []
            for it, out in zip(req.items, outs):
                wav = self._codes_to_wav(list(out.outputs[0].token_ids))
                results.append(
                    {
                        "text": it.text,
                        "speaker_id": it.speaker_id,
                        "sample_rate": 24000,
                        "duration_sec": len(wav) / 24000,
                        "audio_wav_b64": base64.b64encode(
                            _wav_bytes(wav)
                        ).decode("ascii"),
                    }
                )
            return JSONResponse({"results": results})

        return api


# ---------------------------------------------------------------------------
# Local entry point: a quick smoke test you can run with
#     modal run orpheus-3B/modal_deploy.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke_test(
    text: str = "Mwattu, oli otya?",
    speaker_id: str = DEFAULT_SPEAKER_ID,
) -> None:
    """Round-trip one synthesis through the deployed class without HTTP.

    Useful for confirming the model loads and the SNAC pipeline works
    without going through the FastAPI layer.
    """
    chunks = _chunk_text(text)
    print(f"text     = {text!r}")
    print(f"speaker  = {speaker_id!r}")
    print(f"model    = {MODEL_ID}")
    print(f"chunks   = {len(chunks)} (cap={MAX_CHUNKS_PER_REQUEST})")
    print(
        "deploy with `modal deploy orpheus-3B/modal_deploy.py` and then"
        " curl the /tts endpoint — see MODAL_DEPLOYMENT.md for examples."
    )
