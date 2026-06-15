---
license: apache-2.0
language:
  - lg
library_name: transformers
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - orpheus
  - luganda
  - low-resource-language
  - sunbird
  - snac
  - unsloth
datasets:
  - Sunbird/tts
base_model: unsloth/orpheus-3b-0.1-pretrained
---

# Orpheus-3B Sunbird Luganda TTS — `salt_lug_0001`

A Luganda text-to-speech model fine-tuned from
[`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained)
on the [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts) corpus
(`lug` config), filtered to a single speaker (`salt_lug_0001`).

The model speaks **Luganda** in the voice of `salt_lug_0001`. It accepts
arbitrary Luganda text and emits 24 kHz mono speech via the
[SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) audio codec.

## Quick links

- **Base model:** [`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) (Llama-3 architecture)
- **Audio codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (24 kHz, 7 codes per ~12 ms frame)
- **Training dataset:** [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts) — `lug` (Luganda) config, speaker `salt_lug_0001`
- **Training framework:** [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace Trainer

## TL;DR

```python
# After installing the dependencies (see "Inference" below)
wav = synthesize("Mwattu, oli otya?", speaker_id="salt_lug_0001")
# 1-D numpy float32 at 24 kHz mono — write with soundfile
```

---

## Inference

The model wraps every prompt in a multi-speaker tagged format:

```
[SOH] + tokenize("salt_lug_0001: <your text>") + [EOT, EOH]
```

and the model autoregressively emits Llama-3 special tokens followed by
SNAC audio codes that decode to a 24 kHz waveform. Two reference
implementations follow.

### Option A — `transformers` + `unsloth` (single request)

Best for development, notebook-driven iteration, and small batch sizes.

**Install:**

```bash
pip install unsloth snac soundfile torchcodec "datasets>=3.4.1,<4.0.0"
```

**Run:**

```python
import os
import numpy as np
import torch
import soundfile as sf
from unsloth import FastLanguageModel
from snac import SNAC

MODEL_ID   = "sunbird/orpheus-3b-tts-salt-lug-0001"
SPEAKER_ID = "salt_lug_0001"

# Special tokens — must match the training format
END_OF_TEXT     = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH   = 128258
START_OF_HUMAN  = 128259
END_OF_HUMAN    = 128260
PAD_TOKEN       = 128263
AUDIO_TOKEN_LO  = 128266
AUDIO_TOKEN_HI  = 128266 + 7 * 4096   # exclusive

# 1) Load the LM (LoRA already merged into 16-bit weights at training time)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = 4096,
    dtype          = None,         # auto bf16 / fp16
    load_in_4bit   = False,        # set True to halve VRAM at slight quality cost
    token          = os.environ.get("HF_TOKEN"),
)
FastLanguageModel.for_inference(model)

# 2) Load SNAC decoder (CPU is fine — frees GPU for the LM)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")


def _redistribute_codes(code_list: list[int]) -> torch.Tensor:
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(len(code_list) // 7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i + 1] - 4096)
        layer_3.append(code_list[7*i + 2] - 2*4096)
        layer_3.append(code_list[7*i + 3] - 3*4096)
        layer_2.append(code_list[7*i + 4] - 4*4096)
        layer_3.append(code_list[7*i + 5] - 5*4096)
        layer_3.append(code_list[7*i + 6] - 6*4096)
    if not layer_1:
        return torch.zeros(1, 1, 12000)   # ~0.5s silence fallback
    clamp = lambda vals: [max(0, min(4095, v)) for v in vals]
    codes = [torch.tensor(clamp(layer_1)).unsqueeze(0),
             torch.tensor(clamp(layer_2)).unsqueeze(0),
             torch.tensor(clamp(layer_3)).unsqueeze(0)]
    return snac_model.decode(codes)


def synthesize(text: str, speaker_id: str = SPEAKER_ID,
               *, max_new_tokens: int = 1200,
               temperature: float = 0.6, top_p: float = 0.95,
               repetition_penalty: float = 1.1,
               seed: int | None = None) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)

    # Build prompt: [SOH] + tokenizer("speaker_id: text") + [EOT, EOH]
    tagged   = f"{speaker_id}: {text}"
    text_ids = tokenizer(tagged, return_tensors="pt").input_ids
    soh = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    input_ids      = torch.cat([soh, text_ids, end], dim=1).to("cuda")
    attention_mask = torch.ones_like(input_ids)

    generated = model.generate(
        input_ids = input_ids, attention_mask = attention_mask,
        max_new_tokens = max_new_tokens,
        do_sample = True,
        temperature = temperature, top_p = top_p,
        repetition_penalty = repetition_penalty,
        eos_token_id = END_OF_SPEECH, use_cache = True,
    )

    # Crop on last SOS, filter to audio token range, redistribute, decode
    sos_indices = (generated == START_OF_SPEECH).nonzero(as_tuple=True)
    cropped = generated[:, sos_indices[1][-1].item() + 1:] if len(sos_indices[1]) > 0 else generated
    row = cropped[0]
    audio_only = row[(row >= AUDIO_TOKEN_LO) & (row < AUDIO_TOKEN_HI)]
    n = (audio_only.size(0) // 7) * 7
    code_list = [t.item() - AUDIO_TOKEN_LO for t in audio_only[:n]]
    waveform  = _redistribute_codes(code_list)
    return waveform.detach().squeeze().to("cpu").numpy().astype(np.float32)


# 3) Use it
wav = synthesize("Mwattu, Mukama yeebazibwe.", seed=42)
sf.write("output.wav", wav, 24000)
print(f"saved {len(wav)/24000:.2f}s of audio at 24 kHz")
```

### Option B — `vllm` (high throughput, batched, deployment)

Best for serving traffic. PagedAttention + continuous batching gives
roughly **5–10× faster** single-request latency and **10–100× higher**
throughput on batched requests vs. the `transformers` path.

> **Important:** vLLM ships its own torch/transformers and conflicts
> with Unsloth's pinned versions. Use a fresh Python environment for
> vLLM serving — do not install on top of an Unsloth env.

**Install:**

```bash
pip install vllm snac soundfile torchcodec "datasets>=3.4.1,<4.0.0"
```

**Run:**

```python
import os
import numpy as np
import torch
import soundfile as sf
from snac import SNAC
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID   = "sunbird/orpheus-3b-tts-salt-lug-0001"
SPEAKER_ID = "salt_lug_0001"

END_OF_TEXT     = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH   = 128258
START_OF_HUMAN  = 128259
END_OF_HUMAN    = 128260
AUDIO_TOKEN_LO  = 128266
AUDIO_TOKEN_HI  = 128266 + 7 * 4096

# 1) Load LM into vLLM
llm = LLM(
    model = MODEL_ID,
    dtype = "bfloat16",
    max_model_len = 4096,
    gpu_memory_utilization = 0.85,
)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu")


def _build_prompt_token_ids(text: str, speaker_id: str) -> list[int]:
    tagged = f"{speaker_id}: {text}"
    text_ids = tokenizer.encode(tagged, add_special_tokens=True)
    return [START_OF_HUMAN] + text_ids + [END_OF_TEXT, END_OF_HUMAN]


def _codes_to_waveform(generated_token_ids: list[int]) -> np.ndarray:
    ids = torch.tensor(generated_token_ids, dtype=torch.int64)
    sos_pos = (ids == START_OF_SPEECH).nonzero(as_tuple=True)[0]
    if len(sos_pos) > 0:
        ids = ids[sos_pos[-1].item() + 1:]
    audio = ids[(ids >= AUDIO_TOKEN_LO) & (ids < AUDIO_TOKEN_HI)]
    n = (audio.size(0) // 7) * 7
    cl = [t.item() - AUDIO_TOKEN_LO for t in audio[:n]]
    l1, l2, l3 = [], [], []
    for i in range(len(cl) // 7):
        l1.append(cl[7*i])
        l2.append(cl[7*i+1] - 4096); l3.append(cl[7*i+2] - 2*4096)
        l3.append(cl[7*i+3] - 3*4096); l2.append(cl[7*i+4] - 4*4096)
        l3.append(cl[7*i+5] - 5*4096); l3.append(cl[7*i+6] - 6*4096)
    if not l1:
        return np.zeros(12000, dtype=np.float32)
    cb = lambda v: [max(0, min(4095, x)) for x in v]
    codes = [torch.tensor(cb(l1)).unsqueeze(0),
             torch.tensor(cb(l2)).unsqueeze(0),
             torch.tensor(cb(l3)).unsqueeze(0)]
    return snac_model.decode(codes).detach().squeeze().cpu().numpy().astype(np.float32)


def synthesize(text: str, speaker_id: str = SPEAKER_ID,
               *, max_tokens: int = 1200,
               temperature: float = 0.6, top_p: float = 0.95,
               repetition_penalty: float = 1.1,
               seed: int | None = None) -> np.ndarray:
    sp = SamplingParams(
        temperature = temperature, top_p = top_p,
        repetition_penalty = repetition_penalty,
        max_tokens = max_tokens,
        stop_token_ids = [END_OF_SPEECH],
        skip_special_tokens = False,    # we need raw token_ids for SNAC
        seed = seed,
    )
    pids = _build_prompt_token_ids(text, speaker_id)
    out  = llm.generate([{"prompt_token_ids": pids}], sp)
    return _codes_to_waveform(list(out[0].outputs[0].token_ids))


def synthesize_batch(items: list[dict], **kwargs) -> list[np.ndarray]:
    """items: list of {"text": str, "speaker_id": str (optional)}"""
    sp = SamplingParams(
        temperature = kwargs.get("temperature", 0.6),
        top_p = kwargs.get("top_p", 0.95),
        repetition_penalty = kwargs.get("repetition_penalty", 1.1),
        max_tokens = kwargs.get("max_tokens", 1200),
        stop_token_ids = [END_OF_SPEECH],
        skip_special_tokens = False,
        seed = kwargs.get("seed"),
    )
    prompts = [{"prompt_token_ids": _build_prompt_token_ids(
                    it["text"], it.get("speaker_id", SPEAKER_ID))}
               for it in items]
    outputs = llm.generate(prompts, sp)
    return [_codes_to_waveform(list(o.outputs[0].token_ids)) for o in outputs]


# 2) Use it — single
wav = synthesize("Mwattu, Mukama yeebazibwe.", seed=42)
sf.write("output.wav", wav, 24000)

# 3) Use it — batched (much higher throughput)
items = [
    {"text": "Mwattu, oli otya?"},
    {"text": "Webale nyo okwagala Uganda."},
    {"text": "Tunaagenda mu Kampala olwa leero."},
]
wavs = synthesize_batch(items, seed=123)
for i, w in enumerate(wavs):
    sf.write(f"batch_{i:02d}.wav", w, 24000)
```

### Generation parameters

| Param | Default | What it does |
|---|---|---|
| `temperature` | 0.6 | Lower = more deterministic, slightly flatter prosody. |
| `top_p` | 0.95 | Nucleus sampling. Don't drop below 0.9 — produces robotic audio. |
| `repetition_penalty` | 1.1 | Discourages stuck-on-one-frame artefacts. 1.0 disables it. |
| `max_new_tokens` / `max_tokens` | 1200 | ≈ 9–10 s of audio. Raise for longer utterances. |
| `seed` | `None` | Pass an int for reproducible output across runs. |

---

## Token format

The tokenizer is Llama-3's, with Orpheus's audio-codebook special tokens
laid out above the standard text vocabulary:

| Token | ID | Purpose |
|---|---|---|
| `<\|begin_of_text\|>` | 128000 | Llama-3 BOS (auto-prepended by tokenizer) |
| `<\|end_of_text\|>` | 128009 | end of human turn (text portion) |
| `START_OF_SPEECH` | 128257 | model emits this just before audio codes |
| `END_OF_SPEECH` | 128258 | model emits this when it finishes — used as `eos_token_id` / `stop_token_ids` |
| `START_OF_HUMAN` | 128259 | wrap the text prompt |
| `END_OF_HUMAN` | 128260 | wrap the text prompt |
| `START_OF_AI` | 128261 | model emits this to begin its response |
| `END_OF_AI` | 128262 | model emits this when fully done |
| `PAD_TOKEN` | 128263 | left-padding for batched generation |
| audio codebook | 128266 + N·4096 | SNAC codes, N ∈ {0..6} for 7-frame layout |

**Training prompt structure** (and what the model expects at inference):

```
[SOH] + tokenize("salt_lug_0001: <text>") + [EOT] + [EOH]
↳ model autoregressively emits:
[SOA] + [SOS] + audio_codes... + [EOS] + [EOA]
```

To recover audio: find the **last** `START_OF_SPEECH` (128257) in the
output, take everything after it, drop any token outside the audio
codebook range, group into 7-token frames, undo the per-position offsets,
and feed the three layers to `SNAC.decode`. Both inference snippets above
implement this end-to-end.

---

## Training details

| Setting | Value |
|---|---|
| Base model | `unsloth/orpheus-3b-0.1-pretrained` (raw pretrained, not the `-ft` voice-actor variant) |
| Adapter | LoRA r=64, α=64, dropout=0, bias=none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | `adamw_8bit`, weight decay 0.001 |
| LR schedule | linear, lr=2e-4, warmup steps=5 |
| Per-device batch size | 1 (with `gradient_accumulation_steps=4`, effective batch = 4) |
| Epochs | 3 |
| `max_seq_length` | 4096 |
| Precision | bfloat16 weights, 16-bit LoRA |
| Seed | 3407 |
| Hardware | single NVIDIA RTX 4090 (24 GB) |
| Gradient checkpointing | Unsloth's optimised variant |
| Final save | LoRA merged into 16-bit weights via `save_pretrained_merged(save_method="merged_16bit")` |

The pretrained variant of Orpheus was chosen over the `-ft` voice-actor
variant because that variant has a strong English-voice-actor prior that
fights low-resource-language fine-tuning.

### Data prep summary

1. Load `Sunbird/tts` config `lug`, splits `train` and `test`.
2. Filter both splits to `speaker_id == "salt_lug_0001"`.
3. Tag each row with `source = "salt_lug_0001"` (multi-speaker prompt
   format — every row carries the speaker tag, so adding more speakers
   later is continued training, not a re-architecture).
4. Cast `audio` column to 24 kHz via `Audio(sampling_rate=24000)`.
5. Drop rows whose tokenised text alone exceeds `max_seq_length`.
6. Encode each audio clip with `hubertsiuzdak/snac_24khz` → 7 codes per
   frame, flattened with per-layer offsets `(+128266, +4096, +2·4096, …)`.
7. Filter out rows with empty/None codes; drop consecutive duplicate
   frames.
8. Build `input_ids = [SOH] + text_ids + [EOT] + [EOH] + [SOA] + [SOS] + audio_codes + [EOS] + [EOA]`.
9. Drop rows whose total tokenised length still exceeds `max_seq_length`.

---

## Evaluation

Quality was evaluated qualitatively on the held-out `test` split for
`salt_lug_0001` (un-seen utterances, same speaker). Generated audio is
compared A/B against the ground-truth recordings; samples are saved to
`outputs/inference_samples/` by both the
[`Orpheus_3B_Sunbird_Luganda.ipynb`](https://github.com/SunbirdAI/Qwen3-TTS/blob/main/orpheus-3B/Orpheus_3B_Sunbird_Luganda.ipynb)
training notebook and the
[`Orpheus_3B_Sunbird_Luganda_Inference.ipynb`](https://github.com/SunbirdAI/Qwen3-TTS/blob/main/orpheus-3B/Orpheus_3B_Sunbird_Luganda_Inference.ipynb)
inference notebook.

We did **not** run automated metrics (WER on a downstream STT, MOS
prediction, etc.) for this release. Numbers will be added if/when those
become part of the evaluation pipeline.

---

## Intended uses & out-of-scope

**Intended:**

- Luganda voice synthesis for accessibility, language learning,
  human–computer interaction, audio content creation, and downstream
  speech research on low-resource Bantu languages.
- A reference checkpoint for the Sunbird/tts → Orpheus-3B fine-tuning
  pipeline; reproducible training recipe in the
  [companion notebooks](https://github.com/SunbirdAI/Qwen3-TTS/tree/main/orpheus-3B).

**Out of scope:**

- **Voice impersonation / deception.** The model imitates the timbre of a
  consenting Sunbird voice donor (`salt_lug_0001`). Do not use the
  generated audio to impersonate identifiable real persons or to
  produce content that could mislead listeners about who is speaking.
- **High-stakes decisions.** Generated speech may contain pronunciation
  errors, prosodic artefacts, or hallucinated phrases — do not deploy
  in safety-critical contexts (medical, legal, emergency) without
  human review.
- **Languages other than Luganda.** This single-speaker checkpoint was
  trained only on `salt_lug_0001` Luganda data. For other Sunbird
  languages or speakers, see `sunbird/orpheus-3b-tts-multilingual`
  (when available) or fine-tune your own from the same recipe.

---

## Limitations & risks

- **Single-speaker.** Only `salt_lug_0001` is faithfully reproduced. The
  prompt format supports `f"{speaker_id}: {text}"` but the model has not
  seen any other speaker_id during training.
- **Vocabulary coverage.** Limited to the lexicon present in the
  `Sunbird/tts (lug)` training subset (~hundreds of utterances).
  Unfamiliar words, code-switching with English, and out-of-distribution
  proper nouns may produce artefacts.
- **Long utterances.** The model was trained on utterances up to ~16 s
  of audio (`max_seq_length=4096`). Generation may degrade or truncate
  beyond ~10 s of speech.
- **Sampling variance.** With `do_sample=True`, identical prompts can
  produce noticeably different deliveries between runs. Pass `seed=` for
  reproducibility.
- **No emotion/style control.** Unlike the upstream `orpheus-3b-0.1-ft`,
  this fine-tune was not exposed to in-text emotion tags
  (`<laugh>`, `<sigh>`, …). Such tags will be tokenised as ordinary
  text and produce no special prosodic effect.
- **Bias.** Inherits any biases present in the Sunbird/tts corpus and in
  Llama-3's pretraining; we have not audited these systematically.

---

## Hardware requirements

| Mode | Min VRAM | Recommended |
|---|---|---|
| `transformers` + Unsloth, fp16 | 8 GB (with `load_in_4bit=True`) | 16 GB |
| `transformers` + Unsloth, bf16 | 14 GB | 24 GB |
| vLLM, bf16, `max_model_len=4096` | 14 GB | 24 GB |

Audio decoding via SNAC runs on CPU and adds ~50–150 ms per utterance.

---

## License & attribution

This fine-tune is released under **Apache-2.0**, matching the upstream
[`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained)
license. It transitively inherits obligations from:

- The [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) project (CanopyAI).
- The [Llama-3](https://llama.meta.com/llama3/) base architecture and weights — Meta Llama 3 Community License.
- The [SNAC](https://github.com/hubertsiuzdak/snac) audio codec (Hubert Siuzdak, MIT).
- The [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts) dataset.

If you redistribute the merged weights, please carry these attributions
forward.

---

## Citation

If you use this model in your work, please cite both the dataset and the
fine-tuning project:

```bibtex
@misc{sunbird_orpheus3b_lug_2026,
  title        = {Orpheus-3B Sunbird Luganda TTS (salt_lug_0001)},
  author       = {Sunbird AI},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sunbird/orpheus-3b-tts-salt-lug-0001}},
}

@misc{sunbird_tts_dataset,
  title        = {Sunbird Speech Dataset},
  author       = {Sunbird AI},
  howpublished = {\url{https://huggingface.co/datasets/Sunbird/tts}},
}

@misc{orpheus_tts_2025,
  title        = {Orpheus-TTS},
  author       = {Canopy Labs},
  year         = {2025},
  howpublished = {\url{https://github.com/canopyai/Orpheus-TTS}},
}
```

---

## Adapting this card

If you fine-tune additional speakers / languages from the same recipe,
the card above is mostly drop-in — just swap:

- `MODEL_ID` and `SPEAKER_ID` constants in both code blocks.
- The `language:` field in the YAML metadata.
- The "Training data summary" section.
- The dataset citation.

For the multi-speaker variant trained on the full `Sunbird/tts` corpus,
also remove the "Single-speaker" limitation note and add a row to the
prompt examples showing several speaker_ids in one batch.
