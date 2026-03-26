"""
finetune.py — THE AGENT'S SANDBOX
===================================
The AI agent can modify ANYTHING in this file.
The human should NOT touch this file once the ratchet loop starts.

Full fine-tuning (all 9B parameters) of Qwen3.5-9B on DGX Spark.
Memory budget: ~108GB for weights+grads+optimizer, ~20GB headroom.
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ============================================================
# KNOBS — the agent modifies these to optimize eval_loss
# ============================================================

# Model
MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 4096

# Optimizer
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

# Schedule
LR_SCHEDULER = "cosine"          # cosine, linear, constant, cosine_with_restarts
WARMUP_RATIO = 0.03
NUM_EPOCHS = 1
MAX_STEPS = 200                   # -1 to use NUM_EPOCHS instead

# Batching (effective batch = BATCH_SIZE * GRAD_ACCUM = 8)
BATCH_SIZE = 1
GRAD_ACCUM = 8

# Data
DATASET_NAME = "zake7749/Qwen3-Coder-Next-Open-Code-SFT"
TRAIN_SIZE = 48874                # total minus 500 held-out for eval
DATASET_SEED = 42
EVAL_HOLDOUT = 500

# DGX Spark constraints
DATALOADER_WORKERS = 0            # MUST be 0 — fork deadlock on Spark
DATASET_NUM_PROC = 1              # MUST be 1 — same reason

# Layer freezing (set > 0 to freeze first N transformer layers)
FREEZE_LAYERS = 0                 # 0 = train everything
FREEZE_EMBEDDINGS = False         # True = freeze embed_tokens

# ============================================================
# MODEL SETUP — full fine-tune, no PEFT/LoRA
# ============================================================
print(f"[finetune] Loading {MODEL_NAME} for full fine-tuning...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=True,         # <-- KEY: full params, no adapters
)

# Gradient checkpointing — MANDATORY at 9B full fine-tune on 128GB
model.gradient_checkpointing_enable()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# OPTIONAL: LAYER FREEZING STRATEGY
# ============================================================
if FREEZE_EMBEDDINGS:
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    print("[finetune] Froze embedding layer")

if FREEZE_LAYERS > 0:
    for i, layer in enumerate(model.model.layers):
        if i < FREEZE_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False
    print(f"[finetune] Froze first {FREEZE_LAYERS} transformer layers")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[finetune] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ============================================================
# DATASET
# ============================================================
print(f"[finetune] Loading dataset: {DATASET_NAME}")
full_dataset = load_dataset(DATASET_NAME, split="train")
full_dataset = full_dataset.shuffle(seed=DATASET_SEED)

# Deterministic split: last EVAL_HOLDOUT for eval, rest for training
train_dataset = full_dataset.select(range(0, len(full_dataset) - EVAL_HOLDOUT))
if TRAIN_SIZE < len(train_dataset):
    train_dataset = train_dataset.select(range(TRAIN_SIZE))

print(f"[finetune] Training examples: {len(train_dataset)}")


def format_example(example):
    """Convert dataset example to chat-templated text."""
    messages = []
    if "system" in example and example["system"]:
        messages.append({"role": "system", "content": example["system"]})

    if "conversations" in example:
        for turn in example["conversations"]:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt"):
                messages.append({"role": "assistant", "content": content})
    elif "instruction" in example:
        messages.append({"role": "user", "content": example["instruction"]})
        if "output" in example:
            messages.append({"role": "assistant", "content": example["output"]})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


train_dataset = train_dataset.map(
    format_example,
    num_proc=DATASET_NUM_PROC,
    desc="Formatting",
)

# ============================================================
# TRAINING
# ============================================================
output_dir = "./output"

training_args = SFTConfig(
    # Core
    output_dir=output_dir,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS if MAX_STEPS == -1 else -1,

    # Optimizer
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    adam_epsilon=ADAM_EPSILON,
    max_grad_norm=MAX_GRAD_NORM,
    optim="adamw_torch",

    # Schedule
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,

    # Precision
    bf16=True,
    fp16=False,

    # Logging
    logging_steps=10,
    logging_first_step=True,
    report_to="none",

    # Saving
    save_strategy="steps",
    save_steps=MAX_STEPS if MAX_STEPS > 0 else 500,
    save_total_limit=1,

    # Data
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataloader_num_workers=DATALOADER_WORKERS,
    dataset_num_proc=DATASET_NUM_PROC,

    # Memory
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
)

print("[finetune] Starting training...")
train_result = trainer.train()

# Save final checkpoint
print("[finetune] Saving final checkpoint...")
final_dir = os.path.join(output_dir, "checkpoint-final")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

# Print metrics for agent parsing
train_loss = train_result.training_loss
peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
print(f"\n[finetune] === TRAINING COMPLETE ===")
print(f"  train_loss: {train_loss:.6f}")
print(f"  peak_vram_gb: {peak_mem:.1f}")
print(f"  steps: {trainer.state.global_step}")
print(f"  checkpoint: {final_dir}")
