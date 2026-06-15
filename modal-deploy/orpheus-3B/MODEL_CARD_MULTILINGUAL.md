---
license: apache-2.0
language:
  - ach
  - af
  - en
  - ee
  - ff
  - ha
  - ig
  - ki
  - rw
  - lgg
  - ln
  - lg
  - luo
  - nyn
  - st
  - sw
  - teo
  - tn
  - xh
  - yo
library_name: transformers
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - orpheus
  - multilingual
  - multi-speaker
  - african-languages
  - low-resource-language
  - sunbird
  - snac
  - unsloth
datasets:
  - Sunbird/tts
base_model: unsloth/orpheus-3b-0.1-pretrained
---

# Orpheus-3B Sunbird Multilingual TTS

A multilingual, multi-speaker text-to-speech model fine-tuned from
[`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained)
on the **full** [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts)
corpus — 20 language configurations and every speaker present in the
dataset.

The model accepts arbitrary text and emits 24 kHz mono speech via the
[SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) audio codec.
Voice selection happens at the prompt level: prepend the chosen
`speaker_id` followed by `": "` to your text, and the model produces
audio in that speaker's voice.

## Quick links

- **Base model:** [`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) (Llama-3 architecture)
- **Audio codec:** [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz) (24 kHz, 7 codes per ~12 ms frame)
- **Training dataset:** [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts) — all 20 configs, all speakers
- **Training framework:** [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace Trainer

## Languages covered

Speaker IDs encode both the source corpus (`salt_*`, `waxal_*`, `slr32_*`,
`slr129_*`, `bateesa_*`) and the language. Languages marked with an em dash
in the Speaker IDs column are present in the model's training mix but do
not currently expose individual voice IDs in this checkpoint.

| Config | Language | ISO 639-1 | Region | Speaker IDs |
|---|---|---|---|---|
| `ach` | Acholi | — | Uganda, South Sudan | `salt_ach_0001`<br>`waxal_ach_0001`<br>`waxal_ach_0005`<br>`waxal_ach_0006`<br>`waxal_ach_0008` |
| `afr` | Afrikaans | af | South Africa, Namibia | `slr32_afr_0009` |
| `eng` | English | en | (control language) | `salt_eng_0001`<br>`salt_eng_0002`<br>`salt_eng_0003` |
| `ewe` | Ewe | ee | Ghana, Togo | `slr129_ewe_0001` |
| `ful` | Fulah | ff | West Africa (Sahel) | `waxal_ful_0003`<br>`waxal_ful_0004`<br>`waxal_ful_0006` |
| `hau` | Hausa | ha | Nigeria, Niger, Chad | `waxal_hau_0004`<br>`waxal_hau_0006`<br>`waxal_hau_0007`<br>`waxal_hau_0008` |
| `ibo` | Igbo | ig | Nigeria | `waxal_ibo_0003`<br>`waxal_ibo_0005`<br>`waxal_ibo_0008` |
| `kik` | Kikuyu | ki | Kenya | `waxal_kik_0003`<br>`waxal_kik_0004` |
| `kin` | Kinyarwanda | rw | Rwanda | `bateesa_kin_0001` |
| `lgg` | Lugbara | — | Uganda, DRC | — |
| `lin` | Lingala | ln | DRC, Republic of Congo | `slr129_lin_0001` |
| `lug` | Luganda | lg | Uganda | `salt_lug_0001`<br>`waxal_lug_0002`<br>`waxal_lug_0003`<br>`waxal_lug_0004`<br>`waxal_lug_0005`<br>`waxal_lug_0006`<br>`waxal_lug_0007`<br>`waxal_lug_0008` |
| `luo` | Luo (Dholuo) | — | Kenya, Tanzania | `waxal_luo_0001`<br>`waxal_luo_0002`<br>`waxal_luo_0003`<br>`waxal_luo_0004` |
| `nyn` | Runyankole | — | Uganda | `salt_nyn_0001`<br>`waxal_nyn_0003`<br>`waxal_nyn_0004`<br>`waxal_nyn_0007`<br>`waxal_nyn_0008` |
| `sot` | Sesotho | st | Lesotho, South Africa | — |
| `swa` | Swahili | sw | East Africa | `waxal_swa_0006`<br>`waxal_swa_0007` |
| `teo` | Ateso | — | Uganda, Kenya | `salt_teo_0001` |
| `tsn` | Setswana | tn | Botswana, South Africa | — |
| `xho` | Xhosa | xh | South Africa | `slr32_xho_0012` |
| `yor` | Yoruba | yo | Nigeria, Benin | `waxal_yor_0002`<br>`waxal_yor_0006`<br>`waxal_yor_0008` |

Per-language quality scales with the amount of training data Sunbird
collected for that language; some configs have many more speaker hours
than others. **Audition the test split** for each language before relying
on a particular speaker — see the discovery snippet below.

## TL;DR

```python
# After installing the dependencies (see "Inference" below)
wav = synthesize("Mwattu, oli otya?",  speaker_id="salt_lug_0001")  # Luganda
wav = synthesize("Habari yako rafiki.", speaker_id="waxal_swa_0006")  # Swahili
wav = synthesize("Bawo ni, ọrẹ mi?",     speaker_id="waxal_yor_0002")  # Yoruba
```

The model has no explicit "language" knob — the language identity
travels via the speaker tag, since each `<corpus>_<lang>_<NNNN>` voice
(`salt_*`, `waxal_*`, `slr32_*`, `slr129_*`, `bateesa_*`) was recorded in
exactly one language.

---

## Discovering speaker IDs

The exact speaker_ids in each config can be enumerated from the dataset:

```python
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names

CONFIGS = get_dataset_config_names("Sunbird/tts")  # the 20 languages

speakers_by_lang = defaultdict(set)
for cfg in CONFIGS:
    ds = load_dataset("Sunbird/tts", cfg, split="train")
    for sid in ds["speaker_id"]:
        speakers_by_lang[cfg].add(sid)

for cfg, sids in sorted(speakers_by_lang.items()):
    print(f"{cfg}: {len(sids)} speaker(s) — {sorted(sids)[:3]}{'...' if len(sids) > 3 else ''}")
```

Speaker IDs encode both the source corpus and the language — e.g.
`salt_lug_0001` (a Luganda speaker from SALT), `waxal_yor_0002` (a Yoruba
speaker from WaxalNLP), `slr32_xho_0012` (an OpenSLR Xhosa speaker),
`bateesa_kin_0001` (a Kinyarwanda speaker from the Bateesa corpus). See
the **Languages covered** table above for the full list of IDs currently
exposed by this checkpoint. Pass any one of them as `speaker_id` to either
inference function below.

---

## Inference

The model wraps every prompt in a multi-speaker tagged format:

```
[SOH] + tokenize("<speaker_id>: <your text>") + [EOT, EOH]
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

MODEL_ID = "sunbird/orpheus-3b-tts-multilingual"

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


def synthesize(text: str, speaker_id: str,
               *, max_new_tokens: int = 1200,
               temperature: float = 0.6, top_p: float = 0.95,
               repetition_penalty: float = 1.1,
               seed: int | None = None) -> np.ndarray:
    """Synthesize speech for `text` in the voice of `speaker_id`.

    `speaker_id` must be one of the speakers seen during training,
    e.g. "salt_lug_0001" (Luganda) or "waxal_swa_0007" (Swahili).
    """
    if seed is not None:
        torch.manual_seed(seed)

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


# 3) Use it — pick a speaker per language
wav = synthesize("Mwattu, Mukama yeebazibwe.", speaker_id="salt_lug_0001", seed=42)
sf.write("luganda.wav", wav, 24000)

wav = synthesize("Habari yako rafiki.", speaker_id="waxal_swa_0006", seed=42)
sf.write("swahili.wav", wav, 24000)
```

### Option B — `vllm` (high throughput, batched, deployment)

Best for serving traffic. PagedAttention + continuous batching gives
roughly **5–10× faster** single-request latency and **10–100× higher**
throughput on batched requests vs. the `transformers` path. Multi-speaker
batching (different `speaker_id`s in one call) gets the full benefit.

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

MODEL_ID = "sunbird/orpheus-3b-tts-multilingual"

END_OF_TEXT     = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH   = 128258
START_OF_HUMAN  = 128259
END_OF_HUMAN    = 128260
AUDIO_TOKEN_LO  = 128266
AUDIO_TOKEN_HI  = 128266 + 7 * 4096

# 1) Load
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


def synthesize(text: str, speaker_id: str,
               *, max_tokens: int = 1200,
               temperature: float = 0.6, top_p: float = 0.95,
               repetition_penalty: float = 1.1,
               seed: int | None = None) -> np.ndarray:
    sp = SamplingParams(
        temperature = temperature, top_p = top_p,
        repetition_penalty = repetition_penalty,
        max_tokens = max_tokens,
        stop_token_ids = [END_OF_SPEECH],
        skip_special_tokens = False,
        seed = seed,
    )
    pids = _build_prompt_token_ids(text, speaker_id)
    out  = llm.generate([{"prompt_token_ids": pids}], sp)
    return _codes_to_waveform(list(out[0].outputs[0].token_ids))


def synthesize_batch(items: list[dict], **kwargs) -> list[np.ndarray]:
    """items: list of {"text": str, "speaker_id": str} — different speakers
    can be mixed in one batch."""
    sp = SamplingParams(
        temperature = kwargs.get("temperature", 0.6),
        top_p = kwargs.get("top_p", 0.95),
        repetition_penalty = kwargs.get("repetition_penalty", 1.1),
        max_tokens = kwargs.get("max_tokens", 1200),
        stop_token_ids = [END_OF_SPEECH],
        skip_special_tokens = False,
        seed = kwargs.get("seed"),
    )
    prompts = [{"prompt_token_ids": _build_prompt_token_ids(it["text"], it["speaker_id"])}
               for it in items]
    outputs = llm.generate(prompts, sp)
    return [_codes_to_waveform(list(o.outputs[0].token_ids)) for o in outputs]


# 2) Single — pick a speaker per language
wav = synthesize("Mwattu, oli otya?", speaker_id="salt_lug_0001", seed=42)
sf.write("luganda.wav", wav, 24000)

# 3) Batched — different languages and speakers in one GPU pass
items = [
    {"text": "Mwattu, oli otya?",        "speaker_id": "salt_lug_0001"},
    {"text": "Habari yako rafiki.",      "speaker_id": "waxal_swa_0006"},
    {"text": "Bawo ni, ọrẹ mi?",          "speaker_id": "waxal_yor_0002"},
    {"text": "Sannu, ina kwana?",        "speaker_id": "waxal_hau_0004"},
    {"text": "Goeie môre, hoe gaan dit?", "speaker_id": "slr32_afr_0009"},
]
wavs = synthesize_batch(items, seed=123)
for i, (it, w) in enumerate(zip(items, wavs)):
    sf.write(f"batch_{i:02d}_{it['speaker_id']}.wav", w, 24000)
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
[SOH] + tokenize("<corpus>_<lang>_<NNNN>: <text>") + [EOT] + [EOH]
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
| `save_total_limit` | 2 |
| Precision | bfloat16 weights, 16-bit LoRA |
| Seed | 3407 |
| Hardware | single NVIDIA RTX 4090 (24 GB) |
| Gradient checkpointing | Unsloth's optimised variant |
| Final save | LoRA merged into 16-bit weights via `save_pretrained_merged(save_method="merged_16bit")` |

The pretrained variant of Orpheus was chosen over the `-ft` voice-actor
variant because that variant has a strong English-voice-actor prior that
fights low-resource-language fine-tuning.

### Data prep summary

1. Load all 20 configs of `Sunbird/tts` (`get_dataset_config_names`)
   and `concatenate_datasets` their `train` and `test` splits into one
   training set and one held-out evaluation set. **No** speaker filter.
2. Tag each row with `source = example["speaker_id"]` (per-row, not
   constant) — the model learns the multi-speaker prompt format
   `f"{speaker_id}: {text}"` across every speaker it sees.
3. Cast `audio` to 24 kHz via `Audio(sampling_rate=24000)`.
4. Drop rows whose tokenised text alone exceeds `max_seq_length` —
   saves expensive SNAC encoding on rows that would be filtered out
   downstream.
5. Encode each remaining audio clip with `hubertsiuzdak/snac_24khz` →
   7 codes per frame, flattened with per-layer offsets
   `(+128266, +4096, +2·4096, …)`.
6. Filter out rows with empty/None codes; drop consecutive duplicate
   frames.
7. Build `input_ids = [SOH] + text_ids + [EOT] + [EOH] + [SOA] + [SOS] + audio_codes + [EOS] + [EOA]`.
8. Drop rows whose total tokenised length exceeds `max_seq_length`
   (safety net for rows where text fits but text + audio together
   overflow the budget).

---

## Evaluation

Quality was evaluated qualitatively on a diverse held-out test sample:
during training, up to 10 utterances are pulled from
`ds_test.shuffle(seed=42)` covering as many distinct speaker_ids as
possible. Generated audio is saved next to the ground-truth recording
under `inference_samples/sample_<idx>_<speaker_id>.wav` so each language
/ voice combination can be auditioned individually.

We did **not** run automated metrics (WER on a downstream STT, MOS
prediction, language-confusion eval, etc.) for this release. Numbers
will be added if/when those become part of the evaluation pipeline.

**Important caveat — quality varies by language.** The training corpus
is unbalanced across the 20 configs; languages with more speaker hours
in `Sunbird/tts` get more training signal and produce more natural
speech. Audition the per-language samples before relying on a specific
voice for production traffic.

---

## Intended uses & out-of-scope

**Intended:**

- Multilingual voice synthesis for accessibility, language learning,
  human–computer interaction, audio content creation, and downstream
  speech research on the 20 covered languages.
- A reference checkpoint for the Sunbird/tts → Orpheus-3B multilingual
  fine-tuning pipeline; reproducible training recipe in
  [`Orpheus_3B_Sunbird_Multilingual.ipynb`](https://github.com/SunbirdAI/Qwen3-TTS/blob/main/orpheus-3B/Orpheus_3B_Sunbird_Multilingual.ipynb).

**Out of scope:**

- **Voice impersonation / deception.** The model imitates the timbres
  of consenting Sunbird voice donors. Do not use the generated audio
  to impersonate identifiable real persons or to produce content that
  could mislead listeners about who is speaking.
- **High-stakes decisions.** Generated speech may contain pronunciation
  errors, prosodic artefacts, or hallucinated phrases — do not deploy
  in safety-critical contexts (medical, legal, emergency) without
  human review.
- **Languages outside the 20 configs.** The model has no signal for
  languages not present in `Sunbird/tts`; sending German text to any
  speaker will produce garbled output, not "German with a Luganda
  accent".
- **Code-switching.** Each speaker_id was recorded in a single language;
  the model has not seen mixed-language utterances and will likely
  produce phonetic artefacts at language boundaries within one prompt.
- **Cross-language voice transfer.** Sending Acholi text to
  `salt_lug_0001` (a Luganda speaker_id) is undefined behaviour. The
  model has no language-conditioning input separate from the speaker
  tag, so language identity travels via the speaker_id. Use a speaker
  whose `<corpus>_<lang>_<NNNN>` middle field matches the language of
  your text (see the **Languages covered** table for the full list).

---

## Limitations & risks

- **Quality varies by language.** Per-language data volume in
  `Sunbird/tts` is unbalanced. Languages with fewer hours produce
  noticeably less natural speech. Run the per-language test-split
  audit (script below) before committing to a particular voice.
- **No language conditioning.** There is no `language` token; the
  model relies entirely on the speaker_id to disambiguate. Mismatching
  speaker_id and text language is undefined behaviour (see above).
- **Vocabulary coverage.** Limited to the lexicon present in each
  config's training subset. Unfamiliar words, code-switching, and
  out-of-distribution proper nouns may produce artefacts.
- **Long utterances.** The model was trained on utterances up to ~16 s
  of audio (`max_seq_length=4096`). Generation may degrade or
  truncate beyond ~10 s of speech.
- **Sampling variance.** With `do_sample=True`, identical prompts can
  produce noticeably different deliveries between runs. Pass `seed=`
  for reproducibility.
- **No emotion/style control.** Unlike the upstream `orpheus-3b-0.1-ft`,
  this fine-tune was not exposed to in-text emotion tags
  (`<laugh>`, `<sigh>`, …). Such tags will be tokenised as ordinary
  text and produce no special prosodic effect.
- **Bias.** Inherits any biases present in the Sunbird/tts corpus and
  in Llama-3's pretraining; we have not audited these systematically
  per language.

### Quick per-language audit script

```python
from datasets import load_dataset, Audio, get_dataset_config_names
import soundfile as sf
from pathlib import Path

CONFIGS = get_dataset_config_names("Sunbird/tts")
out_dir = Path("language_audit"); out_dir.mkdir(exist_ok=True)

for cfg in CONFIGS:
    ds = load_dataset("Sunbird/tts", cfg, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=24000))
    row = ds[0]
    sid, text = row["speaker_id"], row["text"]
    print(f"{cfg}: {sid} -> {text[:80]}")
    wav = synthesize(text, speaker_id=sid, seed=0)
    sf.write(out_dir / f"{cfg}_{sid}.wav", wav, 24000)
    sf.write(out_dir / f"{cfg}_{sid}_groundtruth.wav",
             row["audio"]["array"], 24000)
```

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
- The [`Sunbird/tts`](https://huggingface.co/datasets/Sunbird/tts) dataset and the SALT voice donors who contributed recordings.

If you redistribute the merged weights, please carry these attributions
forward.

---

## Citation

If you use this model in your work, please cite both the dataset and the
fine-tuning project:

```bibtex
@misc{sunbird_orpheus3b_multilingual_2026,
  title        = {Orpheus-3B Sunbird Multilingual TTS},
  author       = {Sunbird AI},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sunbird/orpheus-3b-tts-multilingual}},
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

## Single-speaker variant

If you only need one specific voice and want a smaller, more focused
checkpoint, see
[`sunbird/orpheus-3b-tts-salt-lug-0001`](https://huggingface.co/sunbird/orpheus-3b-tts-salt-lug-0001)
— same recipe, scoped to a single Luganda speaker.
