# nanosarvam

![nanosarvam architecture](assets/image.png)


A compact, from-scratch implementation of a Mixture-of-Experts (MoE) language model inspired by the [Sarvam-30B](https://www.sarvam.ai/blogs/sarvam-30b-105b) and [DeepSeek-V2](https://arxiv.org/abs/2405.04434) architectures.

Designed to be readable, hackable, and trainable on consumer GPUs.

---

## Architecture

| Component | Detail |
|---|---|
| Attention | Grouped-Query Attention (GQA) with QK-Norm and RoPE |
| FFN | SwiGLU dense block (layer 0) + MoE blocks (remaining layers) |
| MoE routing | Top-K sparse routing with a shared always-active expert |
| Normalisation | RMSNorm pre-norm throughout |
| Position encoding | Rotary Position Embeddings (RoPE) |

### Full config (~30 B parameters)

| Hyperparameter | Value |
|---|---|
| Hidden dim | 4096 |
| Layers | 19 |
| Attention heads | 64 |
| KV heads | 4 |
| Head dim | 64 |
| Vocab size | 262 144 |
| Max sequence length | 131 072 |
| Routed experts | 128 |
| Active experts per token | 6 |
| Expert hidden dim | 1024 |

### Tiny config (~300 M parameters, fits on 8 GB VRAM)

Enabled with `--tiny`. Suitable for experimentation on a consumer GPU.

| Hyperparameter | Value |
|---|---|
| Hidden dim | 1024 |
| Layers | 12 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dim | 64 |
| Routed experts | 8 |
| Active experts per token | 2 |
| Expert hidden dim | 512 |

---

## Installation

```bash
git clone https://github.com/cneuralnetwork/nanosarvam
cd nanosarvam
pip install -r requirements.txt
```

Requirements: Python 3.11+, PyTorch 2.2+, a CUDA-capable GPU.

---

## Quickstart

### Train on TinyStories (8 GB GPU, recommended defaults)

```bash
export WANDB_API_KEY=your_key_here        # or pass --wandb_api_key

python train.py \
  --tiny \
  --use_fp16 \
  --use_8bit_adam \
  --grad_checkpoint \
  --batch_size 1 \
  --accumulation_steps 16 \
  --max_seq_len 512
```

Expected VRAM usage: ~2.5 GB weights + optimizer, ~0.4 GB activations.

### Train on a custom Hugging Face dataset

Any public or gated Hugging Face dataset works. Specify the repo ID with
`--dataset`, the text column with `--text_field`, and the split names:

```bash
# OpenWebText
python train.py --tiny --use_fp16 --use_8bit_adam \
  --dataset "Skylion007/openwebtext" \
  --text_field text \
  --val_split none          # openwebtext has no validation split

# FineWeb (10 BT sample, gated — requires HF token)
python train.py --tiny --use_fp16 --use_8bit_adam \
  --dataset "HuggingFaceFW/fineweb" \
  --dataset_name "sample-10BT" \
  --text_field text \
  --val_split none \
  --hf_token YOUR_HF_TOKEN

# SlimPajama
python train.py --tiny --use_fp16 --use_8bit_adam \
  --dataset "cerebras/SlimPajama-627B" \
  --text_field text \
  --train_split train \
  --val_split validation
```

---

## Full argument reference

### Dataset

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `roneneldan/TinyStories` | Hugging Face dataset repo ID |
| `--dataset_name` | `None` | Dataset config name (e.g. `sample-10BT` for FineWeb) |
| `--text_field` | `text` | Column that contains raw text |
| `--train_split` | `train` | Split name for training data |
| `--val_split` | `validation` | Split name for validation data; set to `none` to skip |
| `--hf_token` | `$HF_TOKEN` | HuggingFace token for gated datasets |

### Tokenizer

| Argument | Default | Description |
|---|---|---|
| `--tokenizer` | `EleutherAI/gpt-neo-125M` | HuggingFace tokenizer repo ID |

### Model

| Argument | Default | Description |
|---|---|---|
| `--tiny` | off | Use the ~300 M-param config instead of the full ~30 B config |
| `--max_seq_len` | `512` | Maximum context length in tokens |

### Training

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `3` | Number of full passes over the training data |
| `--batch_size` | `1` | Per-step batch size |
| `--accumulation_steps` | `16` | Gradient accumulation steps (effective batch = batch × accumulation) |
| `--lr` | `1e-4` | Peak learning rate |
| `--device` | auto | `cuda` or `cpu` |
| `--num_workers` | `4` | DataLoader worker processes |

### Memory optimisation

| Argument | Default | Description |
|---|---|---|
| `--use_fp16` | off | FP16 mixed-precision training (recommended) |
| `--use_8bit_adam` | off | 8-bit AdamW via `bitsandbytes` (~4× smaller optimizer state) |
| `--grad_checkpoint` | off | Gradient checkpointing (saves activation memory, ~30% slower) |

### Checkpointing

| Argument | Default | Description |
|---|---|---|
| `--save_dir` | `./checkpoints` | Directory for checkpoints and logs |
| `--save_every` | `500` | Save a checkpoint every N optimizer steps |
| `--resume_from` | `None` | Path to a `.pt` checkpoint to resume training from |

### Weights & Biases

| Argument | Default | Description |
|---|---|---|
| `--wandb_api_key` | `$WANDB_API_KEY` | W&B API key (prefer env var over CLI to avoid leaking in shell history) |
| `--wandb_project` | `nanosarvam` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--log_every` | `10` | Log metrics every N optimizer steps |

---

## API keys and secrets

Never hardcode API keys in source files. nanosarvam reads credentials
from environment variables and falls back to CLI args as a convenience
for one-off runs.

```bash
# Recommended: set in your shell profile or a .env file (never commit .env)
export WANDB_API_KEY=...
export HF_TOKEN=...         # only needed for gated datasets

python train.py --tiny --use_fp16 --use_8bit_adam ...
```

If no W&B key is found, the run is logged offline to `./wandb/`.

---

## VRAM reference (tiny config, seq_len=512)

| Configuration | Estimated VRAM |
|---|---|
| FP32 + standard Adam | ~11.6 GB |
| FP16 + standard Adam | ~10.2 GB |
| FP16 + 8-bit Adam | ~6.0 GB |
| FP16 + 8-bit Adam + grad checkpoint | ~5.5 GB |

---

## Resuming a run

```bash
python train.py --tiny --use_fp16 --use_8bit_adam \
  --resume_from ./checkpoints/checkpoint_step_1000.pt
```

---

## Project structure

```
nanosarvam/
├── model.py          # Model definition (RMSNorm, RoPE, GQA, SwiGLU, MoE)
├── train.py          # Training loop, dataset loading, CLI
├── requirements.txt
└── README.md
```

---

