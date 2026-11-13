---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - qwen2.5
  - title-generation
  - arxiv
---

# Qwen2.5-1.5B-Instruct LoRA: arXiv Abstract → Title

A LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct` that writes a paper title from an arXiv abstract. It was trained with QLoRA (4-bit NF4 base model) on 4,750 abstract-title pairs from `CShorten/ML-ArXiv-Papers`.

This folder holds only the adapter weights and tokenizer files. The base model is downloaded separately from Hugging Face.

## Intended use

Generating a short, paper-style title for a machine learning abstract. It is a small experimental model, so review outputs before using them anywhere important.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "path/to/qwen-arxiv-title-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

abstract = "Paste an abstract here."

prompt = (
    f"<|im_start|>user\nWrite a title for this abstract:\n{abstract}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

The prompt must follow this exact format, because the adapter was trained on it:

```
<|im_start|>user
Write a title for this abstract:
{abstract}<|im_end|>
<|im_start|>assistant
```

## Training details

**Data**
- Source: `CShorten/ML-ArXiv-Papers`, 5,000 random samples (`seed=42`)
- Split: 4,750 train and 250 validation (`seed=42`)
- Whitespace and line breaks cleaned in abstracts and titles

**Quantization and LoRA**

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization, float16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 (1.18% of 1,562,179,072) |

**Hyperparameters**

| Setting | Value |
|---|---|
| Epochs | 2 (594 steps) |
| Effective batch size | 16 (4 per device × 4 accumulation) |
| Learning rate | 2e-4, cosine schedule, 20 warmup steps |
| Weight decay | 0.01 |
| Optimizer | paged_adamw_8bit |
| Precision | fp16, gradient checkpointing |
| Max sequence length | 512 |
| Hardware | 1× Tesla T4 (Kaggle), about 57 minutes |

Final training loss was 2.205 and final validation loss was 2.328.

## Evaluation

250 held-out validation samples, greedy decoding, `max_new_tokens=40`:

| Metric | Base model | This adapter |
|---|---|---|
| ROUGE-1 | 0.405 | 0.448 |
| ROUGE-2 | 0.212 | 0.249 |
| ROUGE-L | 0.348 | 0.399 |

## Limitations

- Trained on only 5,000 machine learning abstracts, so it may do worse on other fields.
- Evaluated on 250 samples from the same dataset, in a single run. The gain over the base model is modest.
- Generated titles can sound right but miss the real topic.
- The loss was computed over the full formatted text, not only the title tokens.

## Framework versions

Trained on Kaggle in November 2026 with the latest releases of `transformers`, `peft`, `trl` and `bitsandbytes` available at that time, on PyTorch 2.11.0 (CUDA 13.0).

## Author

Md Shahid Afridi ([@ms-afridi](https://github.com/ms-afridi))
