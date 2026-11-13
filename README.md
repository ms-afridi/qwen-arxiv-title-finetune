# Qwen2.5-1.5B-Instruct QLoRA: arXiv Abstract → Title

Fine-tuning [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with QLoRA to generate a paper title from an arXiv abstract. The model is trained on a 5,000-sample subset of ML-ArXiv-Papers on a single Kaggle T4 GPU, and the saved LoRA adapter is included in this repository.

## Results

Evaluated on 250 held-out validation samples with greedy decoding (`max_new_tokens=40`):

| Metric | Base model | Fine-tuned | Gain |
|---|---|---|---|
| ROUGE-1 | 0.405 | 0.448 | +0.042 |
| ROUGE-2 | 0.212 | 0.249 | +0.037 |
| ROUGE-L | 0.348 | 0.399 | +0.051 |

The base numbers come from the same model with the LoRA adapter disabled, scored on the same 250 samples.

### Example outputs

| Original title | Generated title |
|---|---|
| Machine learning structure preserving brackets for forecasting irreversible processes | Learning Dissipative Systems with Metriplectic Brackets |
| Foveation for Segmentation of Ultra-High Resolution Images | Foveation: A Data Loader for Ultra-High Resolution Image Segmentation |
| Data Interpolating Prediction: Alternative Interpretation of Mixup | Data Interpolating Prediction: A Framework for Learning with Augmented Samples |
| Grammar Filtering For Syntax-Guided Synthesis | Learning to Solve Programming-by-Example Problems |
| Loss-Aversively Fair Classification | Loss-Averse Updates: A Behavioral Perspective on Fairness in Algorithmic Decision Making |

Four of the five generated titles are on topic. The fourth is a miss: the model drifts to programming-by-example instead of grammar filtering for syntax-guided synthesis.

## Project overview

| Item | Details |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | QLoRA (4-bit NF4 base model + LoRA adapters) |
| Task | Abstract → title generation |
| Dataset | [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (117,592 papers, `title` and `abstract` columns) |
| Hardware | Kaggle, 1× Tesla T4 |
| Training time | about 57 minutes |

## Data

1. Loaded the full dataset (117,592 rows) and kept a random 5,000-sample subset (`seed=42`).
2. Cleaned whitespace and line breaks in both the abstract and the title.
3. Wrapped each pair in Qwen's ChatML format, with the abstract as the user turn and the title as the assistant turn.
4. Split into 4,750 training and 250 validation samples (`seed=42`).

Prompt format:

```
<|im_start|>user
Write a title for this abstract:
{abstract}<|im_end|>
<|im_start|>assistant
{title}<|im_end|>
```

## Training setup

**Quantization (4-bit)**

| Setting | Value |
|---|---|
| Quantization type | NF4 |
| Double quantization | Yes |
| Compute dtype | float16 |

**LoRA**

| Setting | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 18,464,768 of 1,562,179,072 (1.18%) |

**Trainer (`trl` SFTTrainer)**

| Setting | Value |
|---|---|
| Epochs | 2 (594 steps) |
| Per-device batch size | 4 |
| Gradient accumulation | 4 (effective batch size 16) |
| Learning rate | 2e-4, cosine schedule |
| Warmup | 20 steps |
| Weight decay | 0.01 |
| Optimizer | `paged_adamw_8bit` |
| Precision | fp16 with gradient checkpointing |
| Max sequence length | 512 |
| Evaluation | every 100 steps |

**Training log**

| Step | Training loss | Validation loss | Token accuracy |
|---|---|---|---|
| 100 | 2.292 | 2.333 | 0.503 |
| 200 | 2.305 | 2.328 | 0.504 |
| 300 | 2.310 | 2.323 | 0.505 |
| 400 | 2.259 | 2.329 | 0.503 |
| 500 | 2.216 | 2.329 | 0.504 |
| 594 | 2.205 | 2.328 | 0.504 |

Validation loss stayed flat at about 2.33 while training loss went down slowly, so there was no sign of overfitting. The loss is computed over the whole formatted text (abstract and title), and the abstract tokens make up most of the sequence. That is why the loss and token accuracy look high even though the generated titles are good, and why ROUGE on the generated titles is the better quality measure.

## Repository structure

```
qwen-arxiv-title-finetune/
├── finetune-qwen2-5-1-5b-arxiv-abstract-to-title.ipynb   # full training and evaluation notebook
├── qwen-arxiv-title-adapter/                             # trained LoRA adapter and tokenizer
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   └── README.md
└── README.md
```

## Usage

Install the dependencies:

```bash
pip install -U transformers peft accelerate torch
```

Generate a title from an abstract:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "**Model on Hugging Face:** [ms-afridi/qwen2.5-1.5b-arxiv-title-lora](https://huggingface.co/ms-afridi/qwen2.5-1.5b-arxiv-title-lora)"

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

title = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(title.strip())
```

To reproduce the training, open the notebook on Kaggle with a GPU T4 accelerator and run the cells in order.

## Limitations

- Evaluation uses 250 samples from the same dataset as training, with a single run and no significance test. Treat the gain of about 5 ROUGE-L points as modest.
- The base model starts from a strong baseline, since Qwen has likely seen arXiv text during pretraining. Much of the gain is style: short, paper-like titles without extra words.
- The model only learned from machine learning papers (ML-ArXiv-Papers), so titles for other fields may be weaker.
- Titles can be plausible but wrong in topic, as the grammar-filtering example shows.
- Only 5,000 samples and 2 epochs were used. More data or a larger model would likely improve results.

## Possible improvements

- Use 10k to 20k training samples.
- Compute the loss only on the title tokens instead of the full sequence.
- Try a larger base model, or compare against other small models.
- Add a stronger evaluation, such as BERTScore or human review.

## Acknowledgements

- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0)
- Dataset: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
- Libraries: Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `evaluate`

## Author

Md Shahid Afridi ([@ms-afridi](https://github.com/ms-afridi))
