# CLAUDE.md — Agent Instructions

You are an autonomous ML researcher. Read `program.md` for full instructions.

## Quick Reference

- **Modify**: `finetune.py` ONLY
- **Never touch**: `evaluate.py`, `program.md`
- **Train**: `python finetune.py`
- **Eval**: `python evaluate.py ./output/checkpoint-final`
- **Log**: append results to `results.tsv`
- **Keep/Revert**: based on `eval_loss` — lower wins
- **Memory limit**: 120 GB peak (128 GB total, 8 GB headroom)
- **Time limit**: 90 minutes per experiment

## Workflow

```bash
# 1. Modify finetune.py with your hypothesis
# 2. Commit
git add finetune.py && git commit -m "exp: <your hypothesis>"
# 3. Train
python finetune.py
# 4. Evaluate
python evaluate.py ./output/checkpoint-final
# 5. Parse eval_loss from output, log to results.tsv
# 6. If improved: keep. If not: git revert HEAD --no-edit
# 7. Repeat forever.
```

## Environment

- Hardware: NVIDIA DGX Spark, 128GB unified memory, Blackwell GPU
- MUST set: `UNSLOTH_COMPILE_DISABLE=1`
- MUST keep: `dataloader_num_workers=0`, `dataset_num_proc=1`
- Model: Qwen3.5-9B full fine-tune (all 9B params, NO LoRA)
