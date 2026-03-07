"""
nanosarvam — training script
A compact MoE language model inspired by Sarvam-1 / DeepSeek-V2.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from model import CustomMoEModel, ModelArgs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(save_dir: str = "./logs") -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nanosarvam — train a compact MoE language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- dataset ---
    ds = parser.add_argument_group("dataset")
    ds.add_argument(
        "--dataset",
        type=str,
        default="roneneldan/TinyStories",
        help="Hugging Face dataset repository ID (e.g. 'roneneldan/TinyStories', "
             "'openwebtext', 'HuggingFaceFW/fineweb')",
    )
    ds.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Optional dataset config name (e.g. 'sample-10BT' for fineweb)",
    )
    ds.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field in the dataset that contains the raw text",
    )
    ds.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use for training",
    )
    ds.add_argument(
        "--val_split",
        type=str,
        default="validation",
        help="Dataset split to use for validation (set to 'none' to skip validation)",
    )
    ds.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for gated/private datasets "
             "(falls back to HF_TOKEN environment variable)",
    )

    # --- tokenizer ---
    tok = parser.add_argument_group("tokenizer")
    tok.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="Hugging Face tokenizer to use",
    )

    # --- model ---
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--tiny", action="store_true",
                     help="Use the ~300M-param tiny config (fits on 8 GB VRAM)")
    mdl.add_argument("--max_seq_len", type=int, default=512,
                     help="Maximum sequence length in tokens")

    # --- training ---
    trn = parser.add_argument_group("training")
    trn.add_argument("--epochs", type=int, default=3)
    trn.add_argument("--batch_size", type=int, default=1,
                     help="Per-step batch size (use gradient accumulation to scale effective batch)")
    trn.add_argument("--accumulation_steps", type=int, default=16,
                     help="Gradient accumulation steps (effective batch = batch_size × accumulation_steps)")
    trn.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    trn.add_argument("--device", type=str,
                     default="cuda" if torch.cuda.is_available() else "cpu")
    trn.add_argument("--num_workers", type=int, default=4,
                     help="DataLoader worker processes")

    # --- memory optimisation ---
    mem = parser.add_argument_group("memory optimisation")
    mem.add_argument("--use_fp16", action="store_true",
                     help="FP16 mixed-precision training")
    mem.add_argument("--use_8bit_adam", action="store_true",
                     help="8-bit AdamW via bitsandbytes (~4× smaller optimizer states)")
    mem.add_argument("--grad_checkpoint", action="store_true",
                     help="Gradient checkpointing (recompute activations on backward, ~30%% slower)")

    # --- checkpointing ---
    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument("--save_dir", type=str, default="./checkpoints")
    ckpt.add_argument("--save_every", type=int, default=500,
                      help="Save a checkpoint every N optimizer steps")
    ckpt.add_argument("--resume_from", type=str, default=None,
                      help="Path to a checkpoint .pt file to resume from")

    # --- wandb ---
    wb = parser.add_argument_group("wandb")
    wb.add_argument("--wandb_project", type=str, default="nanosarvam",
                    help="Weights & Biases project name")
    wb.add_argument("--wandb_run_name", type=str, default=None,
                    help="Weights & Biases run name (auto-generated if omitted)")
    wb.add_argument("--wandb_api_key", type=str, default=None,
                    help="Weights & Biases API key "
                         "(falls back to WANDB_API_KEY environment variable)")
    wb.add_argument("--log_every", type=int, default=10,
                    help="Log metrics to wandb every N optimizer steps")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFTextDataset(torch.utils.data.Dataset):
    """
    Generic text dataset backed by any Hugging Face dataset.

    Loads the requested split and caches the raw text strings.
    Tokenisation is deferred to the collate function so the same
    dataset object works with any tokenizer.
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        text_field: str = "text",
        dataset_name: str | None = None,
        hf_token: str | None = None,
    ):
        from datasets import load_dataset

        token = hf_token or os.environ.get("HF_TOKEN")
        print(f"Loading '{dataset_id}' ({split} split)...")

        load_kwargs: dict = dict(split=split, token=token)
        if dataset_name:
            load_kwargs["name"] = dataset_name

        dataset = load_dataset(dataset_id, **load_kwargs)

        if text_field not in dataset.column_names:
            available = dataset.column_names
            raise ValueError(
                f"Field '{text_field}' not found in dataset. "
                f"Available fields: {available}. "
                f"Use --text_field to specify the correct one."
            )

        self.examples: list[str] = []
        for example in tqdm(dataset, desc=f"  caching {split}"):
            text = example[text_field]
            if text and text.strip():
                self.examples.append(text)

        print(f"  {len(self.examples):,} examples loaded from '{dataset_id}' ({split})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> str:
        return self.examples[idx]


def collate_fn(batch: list[str], tokenizer, max_length: int) -> dict:
    """Tokenise a batch of raw strings into input_ids and shifted labels."""
    encodings = tokenizer(
        batch,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model(
    args: argparse.Namespace,
    vocab_size: int,
    use_fp16: bool = False,
    use_8bit_adam: bool = False,
    grad_checkpoint: bool = False,
) -> tuple[CustomMoEModel, ModelArgs, dict]:
    """
    Build model and config from CLI args.

    Returns (model, config, info_dict).

    Tiny config parameter budget (~300 M total):
      embeddings + output : 2 × 50k × 1024       = 103 M
      attention (12 layers): 12 × 2.6 M           =  31 M
      dense FFN (layer 0)  : 3 × 1024 × 2048      =   6 M
      MoE (11 layers)      : 11 × (8 experts × 3 × 1024 × 512 + shared)
                                                   = 155 M
                                            Total ≈ 295 M
    """
    if args.tiny:
        config = ModelArgs(
            dim=1024,
            n_layers=12,
            n_heads=16,
            n_kv_heads=4,
            head_dim=64,
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
            dense_ffn_hidden_dim=2048,
            n_experts=8,
            n_active_experts=2,
            moe_expert_hidden_dim=512,
            use_grad_checkpoint=grad_checkpoint,
        )
        model_type = "tiny (~300 M)"
    else:
        config = ModelArgs()
        config.max_seq_len = args.max_seq_len
        config.vocab_size = vocab_size
        config.use_grad_checkpoint = grad_checkpoint
        model_type = "full (~30 B)"

    model = CustomMoEModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Bytes per param accounting for optimizer states:
    #   FP16 layout : fp16 weights (2) + fp32 master (4) + fp32 grad (4) + m + v
    #   FP32 layout : fp32 weights (4) + fp32 grad (4) + m + v
    opt_bytes = 1 if use_8bit_adam else 4
    bytes_per_param = (2 + 4 + 4 + 2 * opt_bytes) if use_fp16 else (4 + 4 + 2 * opt_bytes)
    estimated_vram_gb = (total_params * bytes_per_param) / (1024 ** 3)

    info = {
        "model_type": model_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "estimated_vram_gb": estimated_vram_gb,
    }
    return model, config, info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logger = setup_logging(save_dir=args.save_dir)

    logger.info("=" * 60)
    logger.info("nanosarvam — starting training run")
    logger.info("=" * 60)

    # --- wandb setup ---
    # Priority: CLI arg > env var. Never hardcode API keys.
    wandb_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        logger.info("Weights & Biases API key configured")
    else:
        logger.warning(
            "No W&B API key found. Pass --wandb_api_key or set the "
            "WANDB_API_KEY environment variable. Running offline."
        )
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "dataset": args.dataset,
            "dataset_name": args.dataset_name,
            "text_field": args.text_field,
            "tokenizer": args.tokenizer,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": args.batch_size * args.accumulation_steps,
            "learning_rate": args.lr,
            "max_seq_len": args.max_seq_len,
            "epochs": args.epochs,
            "tiny": args.tiny,
            "use_fp16": args.use_fp16,
            "use_8bit_adam": args.use_8bit_adam,
            "grad_checkpoint": args.grad_checkpoint,
        },
    )

    # --- system info ---
    logger.info(f"Device     : {args.device}")
    logger.info(f"PyTorch    : {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count  : {torch.cuda.device_count()}")

    # --- tokenizer ---
    logger.info(f"Loading tokenizer '{args.tokenizer}' ...")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer  : {type(tokenizer).__name__}, vocab size {tokenizer.vocab_size:,}")

    # --- checkpoints dir ---
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Checkpoints: {args.save_dir}")

    # --- datasets ---
    logger.info(f"Loading dataset '{args.dataset}' ...")

    train_dataset = HFTextDataset(
        dataset_id=args.dataset,
        split=args.train_split,
        text_field=args.text_field,
        dataset_name=args.dataset_name,
        hf_token=hf_token,
    )

    skip_validation = args.val_split.lower() == "none"
    val_dataset = None
    if not skip_validation:
        try:
            val_dataset = HFTextDataset(
                dataset_id=args.dataset,
                split=args.val_split,
                text_field=args.text_field,
                dataset_name=args.dataset_name,
                hf_token=hf_token,
            )
        except Exception as e:
            logger.warning(
                f"Could not load validation split '{args.val_split}': {e}. "
                "Skipping validation. Use --val_split none to suppress this warning."
            )
            skip_validation = True

    logger.info(f"Train      : {len(train_dataset):,} examples")
    if val_dataset:
        logger.info(f"Validation : {len(val_dataset):,} examples")

    _collate = lambda b: collate_fn(b, tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=True,
        )

    logger.info(f"Train batches : {len(train_loader):,}")
    if val_loader:
        logger.info(f"Val batches   : {len(val_loader):,}")

    # --- model ---
    logger.info("Initialising model ...")
    model, config, model_info = setup_model(
        args,
        vocab_size=tokenizer.vocab_size,
        use_fp16=args.use_fp16,
        use_8bit_adam=args.use_8bit_adam,
        grad_checkpoint=args.grad_checkpoint,
    )
    model = model.to(args.device)

    fp_label = (
        "FP16 + 8-bit Adam" if (args.use_fp16 and args.use_8bit_adam)
        else ("FP16" if args.use_fp16 else "FP32")
    )
    logger.info(f"Model type    : {model_info['model_type']}")
    logger.info(f"Parameters    : {model_info['total_params']:,}")
    logger.info(f"Precision     : {fp_label}")
    logger.info(f"Grad ckpt     : {config.use_grad_checkpoint}")
    logger.info(f"Estimated VRAM: {model_info['estimated_vram_gb']:.2f} GB (weights + optimizer)")
    logger.info(
        f"Architecture  : dim={config.dim}, layers={config.n_layers}, "
        f"heads={config.n_heads}, kv_heads={config.n_kv_heads}"
    )
    logger.info(
        f"MoE           : {config.n_experts} experts, {config.n_active_experts} active, "
        f"hidden={config.moe_expert_hidden_dim}"
    )

    # --- mixed precision scaler ---
    scaler = torch.amp.GradScaler("cuda") if args.use_fp16 else None
    if scaler:
        logger.info("FP16 GradScaler enabled")

    # --- resume ---
    start_step = 0
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from} ...")
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed at step {start_step}")

    # --- optimizer ---
    logger.info("Setting up optimizer ...")
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
            logger.info("Optimizer: 8-bit AdamW (bitsandbytes)")
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — falling back to standard AdamW. "
                "Run: pip install bitsandbytes"
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
        )
        logger.info("Optimizer: AdamW (FP32 states)")

    # --- LR schedule: linear warmup then cosine-ish decay with 10% floor ---
    num_training_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = max(1, num_training_steps // 20)
    logger.info(f"Total steps   : {num_training_steps:,} | Warmup: {warmup_steps:,}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- training loop ---
    logger.info("=" * 60)
    logger.info("Training loop started")
    logger.info("=" * 60)

    global_step = start_step
    model.train()

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            if args.use_fp16:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = criterion(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1),
                    ) / args.accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                ) / args.accumulation_steps
                loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0:
                if args.use_fp16:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    train_loss = loss.item() * args.accumulation_steps
                    wandb.log({
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/step": global_step,
                    })
                    logger.info(
                        f"step {global_step:6d} | loss {train_loss:.4f} "
                        f"| lr {lr:.2e} | grad_norm {grad_norm.item():.3f}"
                    )

                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(
                        args.save_dir, f"checkpoint_step_{global_step}.pt"
                    )
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config,
                    }, ckpt_path)
                    logger.info(f"Checkpoint saved → {ckpt_path}")

            progress.set_postfix({
                "loss": f"{loss.item() * args.accumulation_steps:.4f}",
                "step": global_step,
            })

        # --- validation ---
        if val_loader:
            model.eval()
            logger.info(f"Running validation ...")
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    input_ids = batch["input_ids"].to(args.device)
                    labels = batch["labels"].to(args.device)

                    if args.use_fp16:
                        with torch.amp.autocast("cuda"):
                            logits = model(input_ids)
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = criterion(
                                shift_logits.view(-1, config.vocab_size),
                                shift_labels.view(-1),
                            )
                    else:
                        logits = model(input_ids)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = criterion(
                            shift_logits.view(-1, config.vocab_size),
                            shift_labels.view(-1),
                        )

                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= val_batches
            wandb.log({"val/loss": val_loss, "val/epoch": epoch + 1, "train/step": global_step})
            logger.info(f"Epoch {epoch + 1} | val_loss {val_loss:.4f}")
            model.train()

    # --- final checkpoint ---
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, final_path)
    logger.info(f"Training complete. Final model saved → {final_path}")
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
