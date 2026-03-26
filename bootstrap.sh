#!/bin/bash
# bootstrap.sh — Initialize the autoresearch-ft repo and run baseline
# Run this ONCE, then hand off to your AI agent (Claude Code / Cursor)

set -euo pipefail

echo "=== Autoresearch Full Fine-Tune Bootstrap ==="

# DGX Spark env flags
export UNSLOTH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# Initialize git repo for ratchet tracking
if [ ! -d ".git" ]; then
    git init
    git add -A
    git commit -m "init: autoresearch full fine-tune setup"
    echo "[bootstrap] Git repo initialized"
fi

# Create results log
if [ ! -f "results.tsv" ]; then
    echo -e "commit_hash\teval_loss\tperplexity\ttrain_loss\tpeak_vram_gb\twall_time_min\tstatus\thypothesis" > results.tsv
    echo "[bootstrap] results.tsv created"
fi

# Run baseline experiment
echo ""
echo "=== Running baseline experiment ==="
START=$(date +%s)

python finetune.py

echo ""
echo "=== Running evaluation ==="
python evaluate.py ./output/checkpoint-final

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=== Baseline complete in ${ELAPSED} minutes ==="
echo "Now open Claude Code or Cursor in this directory and say:"
echo '  "Read program.md and start experimenting. The baseline is in results.tsv."'
