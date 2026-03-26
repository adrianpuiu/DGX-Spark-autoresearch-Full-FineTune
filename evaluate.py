"""
evaluate.py — THE IMMUTABLE JUDGE
==================================
Neither the agent NOR the human modifies this file. Ever.
It loads the fully fine-tuned model checkpoint and computes eval metrics
against a held-out split of the Qwen3-Coder-Next-Open-Code-SFT dataset.

Returns: eval_loss (lower is better) — the sole optimization target.
Optionally: pass@1 if you wire up code execution sandboxing.
"""

import torch
import json
import sys
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIGURATION — set once, never change
# ============================================================
BASE_MODEL = "Qwen/Qwen3.5-9B"
DATASET_NAME = "zake7749/Qwen3-Coder-Next-Open-Code-SFT"
EVAL_SIZE = 500                    # held-out examples for evaluation
MAX_SEQ_LEN = 4096                 # must match training
EVAL_BATCH_SIZE = 1                # safe for memory after training cleanup
SEED = 42

# ============================================================
# LOAD EVAL DATASET (deterministic split)
# ============================================================
print("[evaluate] Loading eval dataset...")
full_dataset = load_dataset(DATASET_NAME, split="train")
full_dataset = full_dataset.shuffle(seed=SEED)
eval_dataset = full_dataset.select(range(len(full_dataset) - EVAL_SIZE, len(full_dataset)))
print(f"[evaluate] Eval set: {len(eval_dataset)} examples")


def format_chat(example, tokenizer):
    """Format example into chat template — mirrors training format exactly."""
    messages = []
    if "system" in example and example["system"]:
        messages.append({"role": "system", "content": example["system"]})

    # Handle both single-turn and multi-turn formats
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
    return text


def compute_eval_loss(checkpoint_dir: str) -> dict:
    """
    Load fully fine-tuned checkpoint, compute average eval loss.
    Returns dict with metrics for the results log.
    """
    print(f"[evaluate] Loading checkpoint from {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_examples = 0

    print("[evaluate] Computing eval loss...")
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            text = format_chat(example, tokenizer)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]

            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
            n_examples += 1

            if (i + 1) % 50 == 0:
                running_avg = total_loss / total_tokens
                print(f"  [{i+1}/{len(eval_dataset)}] running eval_loss: {running_avg:.6f}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Clean up GPU memory for next training run
    del model
    torch.cuda.empty_cache()

    results = {
        "eval_loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 4),
        "eval_examples": n_examples,
        "eval_tokens": total_tokens,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n[evaluate] === RESULTS ===")
    print(f"  eval_loss:  {results['eval_loss']}")
    print(f"  perplexity: {results['perplexity']}")
    print(f"  examples:   {results['eval_examples']}")
    print(f"  tokens:     {results['eval_tokens']}")

    return results


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "./output/checkpoint-final"
    results = compute_eval_loss(checkpoint)

    # Write results to JSON for agent consumption
    results_file = os.path.join(os.path.dirname(checkpoint), "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Results written to {results_file}")
    print(f"eval_loss: {results['eval_loss']}")
