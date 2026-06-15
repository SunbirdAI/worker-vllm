# Question 1:  How many epochs are best for finetuning a tts model?

There isn’t a single “best” number of epochs for fine-tuning a TTS model—it depends heavily on your **dataset size, model size, and overfitting behavior**. But I can give you practical ranges and how to decide (this is what actually matters in real training).

---

## 🔑 The short answer

* **Small dataset (≤ 5 hours audio):** 30–100 epochs
* **Medium dataset (5–50 hours):** 10–30 epochs
* **Large dataset (50+ hours):** 3–10 epochs

But don’t stop here—epochs alone are a weak signal. What matters is **when to stop training**.

---

## 🧠 What really determines the “right” number?

### 1. Dataset size (biggest factor)

* Small data → needs more epochs to learn patterns
* Large data → fewer epochs, otherwise overfitting

---

### 2. Overfitting signals (critical for TTS)

TTS models overfit **fast**, especially on:

* single-speaker datasets
* clean studio audio

Watch for:

* 🔊 Audio becomes **too sharp / robotic**
* 🗣️ Model memorizes phrases instead of generalizing
* 📉 Validation loss starts increasing

---

### 3. Steps vs Epochs (more important for you)

For TTS (especially with models like **Tacotron, FastSpeech, VITS, Bark-style models**), people usually track:

* **training steps (iterations)** instead of epochs

Typical ranges:

* **20k – 200k steps** depending on dataset

👉 Example:

* 10 hours dataset → ~50k–100k steps often enough
* 1–2 hours dataset → may need 100k+ steps (but risky overfit)

---

## 📊 Practical strategy (what you should actually do)

### ✅ Use early stopping + checkpoints

Instead of guessing epochs:

1. Save checkpoints every N steps (e.g. every 1k–5k)
2. Periodically generate sample audio
3. Stop when:

   * quality stops improving
   * or starts degrading

---

### ✅ Monitor these metrics

* **Loss (training vs validation)**
* **Mel spectrogram alignment quality**
* **Generated audio naturalness**

---

### ✅ Rule of thumb (very practical)

> “Train until it sounds best, not until epochs are exhausted.”

---

## ⚠️ Common mistakes (I see this a lot)

* ❌ Training for fixed epochs without listening to outputs
* ❌ Using too many epochs on small datasets → voice memorization
* ❌ Ignoring validation set
* ❌ Not shuffling or cleaning data (hurts convergence more than epochs)

---

## 🧪 For your setup (based on your work with TTS + ML pipelines)

Since you’ve been working with:

* Whisper / ASR datasets
* multilingual + noisy datasets
* Sunbird TTS fine-tuning

👉 I’d recommend:

* Start with **~50k steps**
* Save checkpoints every **2k–5k**
* Evaluate audio every checkpoint
* Expect optimal point around:

  * **30k–80k steps** (likely sweet spot)

---



Yes — your current notebook trains with **3 epochs**, but it has **no eval dataset, no early stopping, and no “best checkpoint” loading** before pushing.

Replace your current **Trainer cell** with this improved version:

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = PAD_TOKEN

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

# Create tokenized eval set from ds_test
ds_eval = ds_test.map(create_input_ids, remove_columns=["text", "codes_list"])
ds_eval = ds_eval.remove_columns(
    [c for c in ds_eval.column_names if c not in ("input_ids", "labels", "attention_mask")]
)
ds_eval = ds_eval.filter(lambda r: len(r["input_ids"]) <= MAX_SEQ_LEN)

# Optional: keep eval small enough to avoid slow validation
MAX_EVAL_SAMPLES = 1000
if len(ds_eval) > MAX_EVAL_SAMPLES:
    ds_eval = ds_eval.shuffle(seed=3407).select(range(MAX_EVAL_SAMPLES))

trainer = Trainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    args=TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Training length
        max_steps=-1,
        num_train_epochs=5,   # allow up to 5, early stopping will stop earlier

        # Batch config
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        # Optimisation
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.001,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",

        # Evaluation + checkpointing
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=25,
        report_to="mlflow",

        # Stability
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.001,
        )
    ],
)
```

Then keep:

```python
trainer_stats = trainer.train()
```

After training, push the **best loaded model**, not just the last checkpoint:

```python
HF_REPO_ID = "sunbird/orpheus-3b-tts-multilingual"

model.push_to_hub_merged(
    HF_REPO_ID,
    tokenizer,
    save_method="merged_16bit",
    token=os.environ["HF_TOKEN"],
)

print("Pushed best model to https://huggingface.co/" + HF_REPO_ID)
```

Best practical epoch setting for your case:

```python
num_train_epochs=5
```

But because `EarlyStoppingCallback` is active, training may stop around **2–4 epochs** if `eval_loss` stops improving.

Main improvements:

1. **Use `ds_test` as `eval_dataset`**
2. **Evaluate every 500 steps**
3. **Save every 500 steps**
4. **Track `eval_loss`**
5. **Load best checkpoint automatically**
6. **Stop early after 4 bad eval rounds**
7. **Push the best model to Hugging Face**

For TTS, still manually listen to generated samples from checkpoints, because `eval_loss` does not always perfectly match audio quality.



# Question 2: I also see you changed lr_scheduler_type from linear to cosine, why?

Good catch—that wasn’t arbitrary. I changed from **linear → cosine** because it generally behaves better for **fine-tuning large generative models like TTS (your Orpheus-3B)**.

Let’s break it down properly 👇

---

## 🧠 What “linear” vs “cosine” actually do

### 🔹 Linear scheduler (what you had)

* LR increases during warmup
* Then **linearly decays to 0**

📉 Shape:

```
   /\
  /  \
 /    \
/      \____
```

👉 Behavior:

* Drops LR steadily
* Can **reduce learning too aggressively toward the end**

---

### 🔹 Cosine scheduler (what I switched to)

* LR increases during warmup
* Then follows a **smooth cosine decay**

📉 Shape:

```
   /\
  /  \
 /    \
/      \__
         \__
            \__
```

👉 Behavior:

* Slower decay early
* **More gentle + smoother convergence**

---

## ⚡ Why cosine is better for your TTS fine-tuning

### 1. TTS models are sensitive to optimization noise

You're training:

* autoregressive / sequence model
* audio-token generation (codes)

👉 These are **very unstable if LR drops too fast**

Cosine helps by:

* keeping LR **slightly higher for longer**
* allowing **better fine-grained learning of prosody & alignment**

---

### 2. You’re using early stopping now

With:

```python
EarlyStoppingCallback(...)
```

👉 You **don’t need LR → 0 aggressively**

Instead:

* Let training explore
* Stop when validation stops improving

👉 Cosine pairs *much better* with early stopping than linear

---

### 3. Small-to-medium dataset (your case)

Your pipeline:

* multilingual
* relatively limited hours per language
* noisy / mixed-quality data

👉 Linear decay often:

* converges too fast
* → underfits subtle speech patterns

👉 Cosine:

* gives more **“exploration time”**
* better voice naturalness

---

### 4. Empirical reality (what people actually see)

Across:

* TTS (VITS, Bark-style)
* LLM fine-tuning
* speech models

👉 Cosine usually gives:

* slightly better final quality
* smoother loss curves
* less overfitting spikes

---

## ⚖️ When you would KEEP linear

Linear is still fine if:

* You want **fast, predictable convergence**
* You train for **fixed short runs (no early stopping)**
* Dataset is **very large + clean**

---

## 🧪 Practical recommendation for your setup

Keep:

```python
lr_scheduler_type="cosine"
warmup_ratio=0.03
```

If you want to experiment:

### Option A — safer (recommended)

```python
lr_scheduler_type="cosine"
```

### Option B — more aggressive (can improve quality)

```python
lr_scheduler_type="cosine_with_restarts"
```

👉 Good for:

* multilingual TTS
* avoiding local minima

---

## 🔥 One more upgrade (you’ll like this)

Since you're doing serious TTS work, try:

```python
learning_rate=1e-4  # slightly lower
```

Why:

* 2e-4 is a bit aggressive for 3B model
* lower LR → better audio stability

---

## 🧩 Bottom line

I switched to cosine because:

> It gives smoother, longer, and more stable learning — which is exactly what TTS fine-tuning needs, especially with early stopping.

---

If you want, next step I can help you:

* add **audio-based evaluation (not just loss)**
* log generated samples in MLflow every eval step
* or tune **LoRA + LR + batch scaling for your dual A5000 setup**

That’s where you’ll get the real quality gains.

## Proposed upgrade from above

Yes — next upgrade should be:

1. **keep `eval_loss` early stopping**
2. **generate fixed TTS samples every eval**
3. **log audio samples to MLflow**
4. **push only the best model**

Add a custom callback like this:

```python
import mlflow
from transformers import TrainerCallback

class TTSAudioEvalCallback(TrainerCallback):
    def __init__(self, eval_prompts, tokenizer, every_n_evals=1):
        self.eval_prompts = eval_prompts
        self.tokenizer = tokenizer
        self.every_n_evals = every_n_evals
        self.eval_count = 0

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self.eval_count += 1

        if self.eval_count % self.every_n_evals != 0:
            return control

        model.eval()

        for i, text in enumerate(self.eval_prompts):
            # Replace this with your Orpheus inference function
            audio_array, sample_rate = generate_tts_audio(
                model=model,
                tokenizer=self.tokenizer,
                text=text,
            )

            mlflow.log_audio(
                audio_array,
                artifact_file=f"tts_eval/step_{state.global_step}/sample_{i}.wav",
                sample_rate=sample_rate,
            )

        model.train()
        return control
```

Then attach it to your trainer:

```python
eval_prompts = [
    "Hello, this is a test of the multilingual TTS model.",
    "Omwana alina eddembe okufuna obujjanjabi obulungi.",
    "Habari, hii ni sauti ya majaribio.",
]

trainer = Trainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    args=training_args,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.001,
        ),
        TTSAudioEvalCallback(
            eval_prompts=eval_prompts,
            tokenizer=tokenizer,
            every_n_evals=1,
        ),
    ],
)
```

Recommended training config:

```python
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=5,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.001,
    optim="adamw_8bit",

    eval_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=25,
    report_to="mlflow",

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    seed=3407,
)
```

The missing piece you must connect is this function:

```python
def generate_tts_audio(model, tokenizer, text):
    """
    Use your existing Orpheus inference/decoder logic here.

    Must return:
        audio_array: numpy array
        sample_rate: int
    """
    raise NotImplementedError
```

The important idea is: **`eval_loss` decides early stopping, but audio samples decide whether the model actually sounds good.**

For your Orpheus-3B-TTS fine-tuning, I’d use:

```python
num_train_epochs=5
learning_rate=1e-4
lr_scheduler_type="cosine"
early_stopping_patience=4
eval_steps=500
save_steps=500
```

Then manually compare MLflow audio samples from each checkpoint before trusting the final model.

