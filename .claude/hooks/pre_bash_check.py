#!/usr/bin/env python3
"""
PreToolUse hook for Bash calls.
Reads tool input from stdin (JSON), detects RAPTOR commands, prints advisory warnings.
Exit code 0 = allow the tool call to proceed.
Exit code 2 = block the tool call (used only for critical errors).
"""
import json
import os
import sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")


# ── Training / ablation ──────────────────────────────────────────────────────
if "train_dinov3_rtdetr_ov" in cmd or "run_ablation" in cmd:
    print("[RAPTOR pre-hook] Detected training command.")

    # GPU check
    try:
        import torch
        if not torch.cuda.is_available():
            print("[RAPTOR pre-hook] WARNING: No CUDA GPU detected. Training will be impractically slow on CPU.")
        else:
            n = torch.cuda.device_count()
            mem = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(n)]
            print(f"[RAPTOR pre-hook] GPU OK — {n} device(s), VRAM: {mem} GB")
    except ImportError:
        print("[RAPTOR pre-hook] WARNING: torch not importable. Run: pip install -r requirements.txt")

    # W&B check
    if not os.environ.get("WANDB_PROJECT"):
        print("[RAPTOR pre-hook] WARNING: WANDB_PROJECT is not set. "
              "Set it with: export WANDB_PROJECT=<your-project-name>")
    else:
        print(f"[RAPTOR pre-hook] W&B project: {os.environ['WANDB_PROJECT']}")

    # Config check
    if not os.path.exists("config.json"):
        print("[RAPTOR pre-hook] WARNING: config.json not found in working directory.")


# ── Data download / prepare ──────────────────────────────────────────────────
elif "download_and_prepare" in cmd or "convert_and_merge" in cmd:
    print("[RAPTOR pre-hook] Detected data pipeline command.")

    import shutil
    _, _, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    print(f"[RAPTOR pre-hook] Free disk space: {free_gb:.1f} GB")
    if free_gb < 100:
        print("[RAPTOR pre-hook] WARNING: Less than 100 GB free. "
              "COCO + LVIS + OpenImages-V7 needs ~80-100 GB.")


# ── Inference / eval ─────────────────────────────────────────────────────────
elif any(x in cmd for x in ["infer_cli.py", "eval_and_infer.py", "zero_shot_prompt.py"]):
    checkpoint_dir = "runs/dinov3_rtdetr/final"
    if not os.path.exists(checkpoint_dir):
        print(f"[RAPTOR pre-hook] WARNING: No checkpoint at {checkpoint_dir}. "
              f"Run training first (/train or python train/train_dinov3_rtdetr_ov.py).")
    else:
        print(f"[RAPTOR pre-hook] Checkpoint found at {checkpoint_dir}/")

sys.exit(0)
