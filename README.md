This is your own code we just built — here's the step-by-step to get it running on the Spark.

**1. Get the zip onto your DGX Spark**
```bash
# From your local machine (or wherever you downloaded it)
scp autoresearch-ft.zip user@your-dgx-spark:/home/user/
```

**2. Unzip and enter**
```bash
ssh user@your-dgx-spark
cd /home/user
unzip autoresearch-ft.zip -d autoresearch-ft
cd autoresearch-ft
```

**3. Make sure Unsloth's Docker container is running**
```bash
# If you already have the container from your current training run, just exec into it:
docker exec -it <your-container-name> bash

# Or start a new one:
docker run -it --gpus=all --net=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/autoresearch-ft:/workspace/autoresearch-ft \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/autoresearch-ft \
  <your-unsloth-dgx-spark-image>
```

**4. Set the env flags**
```bash
export UNSLOTH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
```

**5. Quick sanity check — make sure deps are there**
```bash
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
python -c "from trl import SFTTrainer; print('TRL OK')"
python -c "from datasets import load_dataset; print('Datasets OK')"
```

**6. Inspect the actual dataset columns** (important — you may need to tweak `format_example`)
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('zake7749/Qwen3-Coder-Next-Open-Code-SFT', split='train[:1]')
print(ds.column_names)
print(ds[0].keys())
for k, v in ds[0].items():
    print(f'\n--- {k} ---')
    print(str(v)[:300])
"
```

Whatever columns you see, update `format_example()` in **both** `finetune.py` and `evaluate.py` to match. The function I wrote handles `conversations`, `instruction/output`, and `system` — but the real schema might be different.

**7. Run the baseline**
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```

This does three things: inits a git repo, runs `finetune.py` (your ~57 min training), then runs `evaluate.py` on the checkpoint. You'll see the baseline `eval_loss` printed at the end.

**8. Hand off to the AI agent**
```bash
# Option A: Claude Code
claude
# Then type: "Read program.md and start experimenting. The baseline is in results.tsv."

# Option B: Cursor
# Just open the directory in Cursor and prompt the same thing
```

From there, the agent reads `program.md`, modifies `finetune.py`, trains, evals, keeps or reverts, and loops forever. Each cycle is ~67 minutes, so you get **~20 experiments per day** running unattended.

**The one thing to watch:** if the first run OOMs, reduce `MAX_SEQ_LEN` to 2048 or set `FREEZE_LAYERS = 8` in `finetune.py` before handing off to the agent. Your current run at step 27 proves it fits, so you should be fine.
