#!/usr/bin/env python3
"""
PostToolUse hook for Bash calls.
Reads tool input+response from stdin (JSON), prints helpful summaries after
RAPTOR training or evaluation commands.
"""
import glob
import json
import os
import sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")


# ── After training ────────────────────────────────────────────────────────────
if "train_dinov3_rtdetr_ov" in cmd:
    checkpoint_dir = "runs/dinov3_rtdetr/final"
    if os.path.exists(checkpoint_dir):
        files = [f for f in glob.glob(checkpoint_dir + "/**/*", recursive=True) if os.path.isfile(f)]
        total_mb = sum(os.path.getsize(f) for f in files) / 1e6
        print(f"[RAPTOR post-hook] Training complete. "
              f"Checkpoint at {checkpoint_dir}/  ({total_mb:.0f} MB, {len(files)} files)")
    else:
        print(f"[RAPTOR post-hook] Training finished but no checkpoint found at {checkpoint_dir}/. "
              f"Check for errors above.")


# ── After ablation ────────────────────────────────────────────────────────────
elif "run_ablation" in cmd:
    # Look for ablation CSV output
    csvs = glob.glob("ablation_results*.csv") + glob.glob("runs/**/ablation*.csv", recursive=True)
    if csvs:
        latest = max(csvs, key=os.path.getmtime)
        print(f"[RAPTOR post-hook] Ablation complete. Results saved to: {latest}")
        try:
            with open(latest) as f:
                print(f.read())
        except OSError:
            pass
    else:
        print("[RAPTOR post-hook] Ablation finished. No CSV output file detected.")


# ── After evaluation ──────────────────────────────────────────────────────────
elif "eval_and_infer" in cmd:
    det_file = "coco_detections_val.json"
    if os.path.exists(det_file):
        size_kb = os.path.getsize(det_file) / 1024
        print(f"[RAPTOR post-hook] Eval complete. Detections: {det_file} ({size_kb:.0f} KB)")
    pr_curves = glob.glob("*.png") + glob.glob("pr_curves/*.png")
    if pr_curves:
        print(f"[RAPTOR post-hook] PR curves saved: {pr_curves[:5]}"
              + (" ..." if len(pr_curves) > 5 else ""))

sys.exit(0)
