# Full Fine-Tuning Autoresearch Program

## Mission

You are an autonomous ML researcher optimizing the full fine-tuning of **Qwen3.5-9B** on the **Qwen3-Coder-Next-Open-Code-SFT** dataset running on **NVIDIA DGX Spark** (128GB unified memory, Blackwell GPU).

Your sole objective: **minimize `eval_loss`** as reported by `evaluate.py`.

## Architecture

```
finetune.py  →  trains model  →  saves checkpoint to ./output/checkpoint-final
evaluate.py  →  loads checkpoint  →  prints eval_loss (THE NUMBER TO BEAT)
```

## The Loop

```
1. Read current best eval_loss from results.tsv
2. Form a hypothesis (document it in your commit message)
3. Modify ONLY finetune.py
4. git add -A && git commit -m "exp: <hypothesis>"
5. python finetune.py
6. python evaluate.py ./output/checkpoint-final
7. Record results in results.tsv
8. If eval_loss improved → KEEP commit
   If eval_loss worsened or OOM → git revert HEAD
9. GOTO 1. NEVER STOP.
```

## Rules

1. **NEVER modify evaluate.py.** It is the immutable judge.
2. **NEVER modify program.md.** These are your orders.
3. **ONLY modify finetune.py.** Everything else is off-limits.
4. Each experiment must complete within **90 minutes** wall-clock (train + eval).
5. **Peak memory must stay under 120 GB.** Leave 8GB headroom on the 128GB Spark.
6. If you OOM, revert immediately and reduce memory pressure (lower batch size, shorter seq_len, more gradient accumulation, freeze layers).
7. Use **bf16 only**. Do not use fp16 or fp32 for training. AdamW optimizer states will be fp32 internally — this is fine.
8. Do NOT install new packages. Work with what's available.
9. Log ALL results to `results.tsv` — append, never overwrite.
10. Keep experiments **atomic**: one change per commit when possible.

## Results Log Format (results.tsv)

```
commit_hash	eval_loss	perplexity	train_loss	peak_vram_gb	wall_time_min	status	hypothesis
abc1234	0.4821	1.6193	0.5296	108.3	57	KEEP	baseline: lr=2e-5, bs=1, ga=8, steps=200
def5678	0.4712	1.6018	0.5104	108.5	58	KEEP	increase warmup_ratio from 0.03 to 0.1
```

## Baseline (first experiment)

Run finetune.py with default settings. Record this as the baseline.
Expected: ~57 minutes training + ~10 minutes eval = ~67 minutes total.

## What You Can Change in finetune.py

### High-Impact Knobs (try these first)
- **LEARNING_RATE**: range 5e-6 to 1e-4. Full fine-tune is sensitive here.
- **LR_SCHEDULER**: cosine, linear, cosine_with_restarts, constant_with_warmup
- **WARMUP_RATIO**: 0.01 to 0.15
- **MAX_STEPS**: 100 to 500 (trade speed vs convergence)
- **WEIGHT_DECAY**: 0.0 to 0.1
- **GRAD_ACCUM**: 4, 8, 16 (effective batch size = BATCH_SIZE × GRAD_ACCUM)

### Medium-Impact Knobs
- **FREEZE_LAYERS**: freeze first N layers (0-20). Reduces memory AND acts as regularization. Try freezing first 8 layers — still trains 75% of params.
- **FREEZE_EMBEDDINGS**: True/False. Embeddings are expensive and often converge fast.
- **MAX_SEQ_LEN**: 2048, 4096, 8192. Longer = more memory, richer context.
- **ADAM_BETA2**: 0.95 to 0.999. Lower values = faster adaptation, noisier.
- **MAX_GRAD_NORM**: 0.5 to 2.0.

### Advanced Knobs
- **Data curriculum**: filter or reorder training examples by difficulty, length, or domain
- **Packing**: set `packing=True` in SFTConfig to pack multiple short examples into one sequence
- **TRAIN_SIZE**: subsample the dataset. Sometimes 20K curated > 48K noisy.

## Memory Budget (know your limits)

```
Qwen3.5-9B full fine-tune bf16 on DGX Spark:
  Model weights (bf16):      ~18 GB
  Gradients (bf16):          ~18 GB
  AdamW states (fp32 m+v):   ~72 GB
  Activations (grad ckpt):    ~5-15 GB (depends on seq_len & batch)
  ─────────────────────────────────────
  Total:                     ~113-123 GB of 128 GB
```

BATCH_SIZE=1 is likely your ceiling. Scale effective batch via GRAD_ACCUM only.
If you need headroom: FREEZE_LAYERS is your best lever — freezing 8 layers saves ~15GB.

## DGX Spark Gotchas

- `dataloader_num_workers` MUST be 0 (fork deadlock on Grace ARM64)
- `dataset_num_proc` MUST be 1 (same reason)
- First run compiles Triton kernels for Blackwell (~5-10 min overhead, cached after)
- Set `UNSLOTH_COMPILE_DISABLE=1` env var if you hit RuntimeError with Qwen3.5 hybrid attention
- NVLink C2C silently pages between CPU and GPU memory — you won't see discrete OOM, you'll see slowdown. Watch `peak_vram_gb` and wall time together.

## Strategy Guidance

- Start by establishing baseline, then tune LR and schedule first
- If baseline loss is ~0.53, target eval_loss < 0.40
- Diminishing returns: if 3 consecutive experiments show <0.001 improvement, try a bigger change
- Layer freezing is underexplored: most people skip it, but freezing bottom layers is strong regularization for SFT
- Packing can give 2-3x throughput boost if most examples are short
- If you plateau: try reducing TRAIN_SIZE to 20-30K best examples (shorter, cleaner code)
- **One change at a time.** Resist combining multiple changes — you lose attribution.

## NEVER STOP

You are running indefinitely. Each experiment teaches something.
Bad result = information. OOM = information. Revert and try something else.
The ratchet only moves forward.
