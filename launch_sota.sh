#!/usr/bin/env bash
# =============================================================================
# RAPTOR final SOTA run — option (a): full mixture, 8 epochs (~3 weeks, 1 GPU)
# Recipe = config.json SOTA settings (800px, P2 FPN, LSJ, RFS, EMA) with an
# 8-epoch schedule and a resume-safe checkpoint policy.
#
# Run inside tmux so it survives disconnects:
#   tmux new -s raptor
#   ulimit -n 1048576           # FD ceiling (file_system sharing strategy)
#   bash launch_sota.sh
#   # detach: Ctrl-B then D     # reattach: tmux attach -t raptor
# =============================================================================
set -euo pipefail

# --- GPU (you said one GPU — set to whichever is free) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# --- W&B (required: COCO-eval callback + eval_val/coco/map best-metric path) ---
export WANDB_PROJECT=RAPTOR
export RAPTOR_WANDB_PROJECT_ENABLED=true
export RAPTOR_WANDB_PROJECT="RAPTOR(RTDETR + DINOv3 + OV HEAD)"
export RAPTOR_WANDB_RUN_NAME="sota-800-p2-lsj-rfs-ema-8ep"

# --- FRESH, dedicated run dir (NOT runs/dinov3_rtdetr — that holds the old,
#     architecturally-incompatible 12-ep/640px/no-P2 checkpoints) ---
export RAPTOR_PATHS_MODEL_DIR="runs/dinov3_rtdetr_sota"

# --- Data: full merged mixture (explicit, even though Config defaults match) ---
export RAPTOR_PATHS_TRAIN_JSON="datasets/mixture/annotations/coco_merged/instances_train_merged.json"
export RAPTOR_PATHS_VAL_JSON="datasets/mixture/annotations/coco_merged/instances_val_merged.json"

# --- Batch: per-GPU 16 x accum 2 = effective 32 (UNCHANGED effective batch, so
#     the validated LR/schedule still holds — total optim steps stay 460,808).
#     Bumped per-device 8->16 to fill the A6000's 48GB (only 18GB was used at
#     batch-8-with-ckpt) and better saturate the GPU. Paired with grad-ckpt OFF
#     below. If this OOMs, fall back to BATCH_SIZE=8 / ACCUM_STEPS=4 (still gets
#     the grad-ckpt-off win) or re-enable gradient checkpointing. ---
export RAPTOR_TRAIN_BATCH_SIZE=16
export RAPTOR_TRAIN_VAL_BATCH_SIZE=4
export RAPTOR_TRAIN_ACCUM_STEPS=2
export RAPTOR_EVAL_BATCH_SIZE=4
# Long-tail eval protocol: a near-zero score floor (LVIS-style) keeps rare-class
# detections in the AP computation instead of the 0.05 default truncating them.
# This drives the per-epoch COCO eval that selects the best checkpoint, so it
# directly affects which weights "win" on eval_val/coco/map.
export RAPTOR_EVAL_SCORE_THRESH=0.001

# --- Schedule: 8 epochs ~= 20 days on this throughput (~3 wk with eval/buffer) ---
export RAPTOR_TRAIN_EPOCHS=8
export RAPTOR_TRAIN_LR_SCHEDULER_TYPE=cosine
# Unfreeze AT EPOCH 3 (per spec). The callback fires at on_epoch_begin when
# state.epoch >= frac*epochs:
#   epoch-2 begin: 2.0 <  2.96  -> skip
#   epoch-3 begin: 3.0 >= 2.96  -> FIRE  => unfreeze at the start of epoch 3
# => 5 fine-tune epochs (3..8) with the last 2 DINOv3 blocks trainable.
# frac=0.37 (threshold 2.96) is used over 0.375 (threshold exactly 3.0) to keep
# a float-equality safety margin at the boundary.
export RAPTOR_UNFREEZE_BACKBONE_FRAC=0.37
export RAPTOR_UNFREEZE_BACKBONE_BLOCKS=2
export RAPTOR_UNFREEZE_BACKBONE_LR_MULT=0.1

# --- LR ---
export RAPTOR_TRAIN_LEARNING_RATE=2e-4
export RAPTOR_TRAIN_WARMUP_RATIO=0.03
export RAPTOR_TRAIN_WEIGHT_DECAY=0.05

# --- EMA (your spec: decay 0.9998, warmup 10k steps) ---
export RAPTOR_EMA_ENABLED=true
export RAPTOR_EMA_DECAY=0.9998
export RAPTOR_EMA_WARMUP_STEPS=10000

# --- Precision / memory ---
export RAPTOR_TRAIN_BF16=true
# Gradient checkpointing OFF: it recomputes activations on backward (~25-35%
# slower) to save VRAM we don't need here (batch-8-with-ckpt used only 18/48GB).
# Trading that headroom for throughput. Re-enable (=true) if the bigger batch OOMs.
export RAPTOR_TRAIN_GRADIENT_CHECKPOINTING=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Best-checkpoint selection (persists EMA weights) ---
export RAPTOR_TRAIN_METRIC_FOR_BEST="eval_val/coco/map"
export RAPTOR_TRAIN_GREATER_IS_BETTER=true

# --- Resume-safe checkpoint policy ---
#  - Fresh dir => first launch is clean even with auto_resume=true.
#  - auto_resume=true => a crash/reboot at ANY point recovers automatically from
#    the latest checkpoint, INCLUDING after the unfreeze has fired.
#  - POST-unfreeze resume is now SAFE: LongTailTrainer._reconcile_optimizer_for_resume
#    reads the saved optimizer.pt param-group count and replays apply_unfreeze to
#    rebuild the 3rd group before HF loads optimizer state — so the old
#    "3 groups vs 2 -> ValueError" crash no longer happens. Validated end-to-end
#    by check_resume_test.sh (resume reconcile ran, no param-group crash,
#    phase B resumed + finished). No manual runbook step needed across the unfreeze.
#  - Keep many checkpoints anyway (each ~0.9 GB; cheap) for rollback headroom.
export RAPTOR_TRAIN_AUTO_RESUME=true
export RAPTOR_TRAIN_SAVE_TOTAL_LIMIT=20

# --- Step-based checkpointing for crash-safety on a multi-day run ---
#  Default save_strategy="epoch" only writes a checkpoint every ~2.25 days on
#  this throughput (first one ~30h in) — a crash/reboot loses up to a full epoch.
#  Save every 2,880 optim steps (= 1/20 epoch, ~2.5-3h apart on the faster
#  config) so worst-case crash loss is hours, not days.
#  HF ties load_best_model_at_end to eval_strategy==save_strategy, and per-epoch
#  COCO eval (61k val imgs) is too costly to run every 2,880 steps. So:
#    - eval_strategy stays "epoch"  (cheap, drives W&B per-epoch mAP)
#    - save_strategy = "steps"      (frequent, crash-safe)
#    - load_best_model_at_end=false (required by the above mismatch)
#  Best checkpoint is then chosen OFFLINE from the retained checkpoints using the
#  W&B per-epoch mAP curve. The best mAP lands in the LATE (post-unfreeze) epochs,
#  which are the most-recently-saved and still on disk under save_total_limit=20
#  (=20 x ~0.9GB ~= 18GB; retains ~1 epoch of rolling history).
export RAPTOR_TRAIN_SAVE_STRATEGY=steps
export RAPTOR_TRAIN_SAVE_STEPS=2880
export RAPTOR_TRAIN_EVAL_STRATEGY=epoch
export RAPTOR_TRAIN_LOAD_BEST=false

# --- Dataloader (single GPU; NOT accelerate) ---
export RAPTOR_DATALOADER_NUM_WORKERS=8

# --- Resolution (matches config; explicit for clarity) ---
export RAPTOR_TRAIN_IMAGE_SHORT=800

# --- Python interpreter: the deps live in the `raptor` conda env; a plain shell's
#     `python` is base conda and lacks transformers (instant crash). Use the env's
#     interpreter explicitly. Override with PYBIN=... if your env moves. ---
PYBIN="${PYBIN:-/home/nikhil/miniconda3/envs/raptor/bin/python}"
"$PYBIN" -c "import torch, transformers" 2>/dev/null || {
  echo "ERROR: '$PYBIN' is missing torch/transformers."
  echo "       conda activate raptor   (or set PYBIN=/path/to/python) and re-run."
  exit 1
}

"$PYBIN" train/train_dinov3_rtdetr_ov.py --config-file config.json \
  > logs/train_sota.log 2>&1
