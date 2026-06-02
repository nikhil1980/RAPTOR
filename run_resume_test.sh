#!/usr/bin/env bash
# Validate the post-unfreeze resume patch on the sub-5k subsample.
# Phase A: train past the unfreeze, kill mid-training (leaves a 3-group ckpt).
# Phase B: resume -> must NOT raise the param-group ValueError; must continue.
set -uo pipefail
cd /home/nikhil/source_code/RAPTOR

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export RAPTOR_WANDB_PROJECT_ENABLED=false            # keep light; metric=eval_loss
export RAPTOR_PATHS_MODEL_DIR=runs/resume_test
export RAPTOR_PATHS_TRAIN_JSON=datasets/mixture/annotations/coco_merged/instances_train_sub5k.json
export RAPTOR_PATHS_VAL_JSON=datasets/mixture/annotations/coco_merged/instances_val_sub1k.json
export RAPTOR_TRAIN_EPOCHS=3
export RAPTOR_UNFREEZE_BACKBONE_FRAC=0.25            # thr 0.75 -> fires at epoch 1.0 (start of ep2)
export RAPTOR_UNFREEZE_BACKBONE_BLOCKS=2
export RAPTOR_UNFREEZE_BACKBONE_LR_MULT=0.1
export RAPTOR_TRAIN_BATCH_SIZE=8
export RAPTOR_TRAIN_VAL_BATCH_SIZE=4
export RAPTOR_TRAIN_ACCUM_STEPS=4                    # 5000/32 ~= 156 steps/epoch
export RAPTOR_TRAIN_LEARNING_RATE=2e-4
export RAPTOR_TRAIN_METRIC_FOR_BEST=eval_loss
export RAPTOR_TRAIN_GREATER_IS_BETTER=false
export RAPTOR_TRAIN_BF16=true
export RAPTOR_TRAIN_GRADIENT_CHECKPOINTING=true
export RAPTOR_EMA_ENABLED=true
export RAPTOR_DATALOADER_NUM_WORKERS=4
export RAPTOR_TRAIN_SAVE_TOTAL_LIMIT=6
export RAPTOR_TRAIN_AUTO_RESUME=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 1048576 || true

rm -rf runs/resume_test
mkdir -p runs/resume_test logs

echo "===== PHASE A (train past unfreeze, then kill) ====="
python train/train_dinov3_rtdetr_ov.py --config-file config.json > logs/resume_testA.log 2>&1 &
PID=$!
echo "Phase A PID=$PID"
# Wait for the first post-unfreeze checkpoint (epoch 2 = checkpoint-312).
for i in $(seq 1 120); do
  if compgen -G "runs/resume_test/checkpoint-312" > /dev/null; then
    echo "[$i] checkpoint-312 present (post-unfreeze)"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "Phase A exited unexpectedly"; break; fi
  sleep 20
done
echo "--- killing Phase A to simulate interruption ---"
kill "$PID" 2>/dev/null || true
sleep 5
pkill -9 -f train_dinov3_rtdetr_ov 2>/dev/null || true
sleep 8
echo "Checkpoints after Phase A:"; ls -d runs/resume_test/checkpoint-* 2>/dev/null

echo "===== PHASE B (resume from post-unfreeze checkpoint) ====="
python train/train_dinov3_rtdetr_ov.py --config-file config.json > logs/resume_testB.log 2>&1
RC=$?
echo "Phase B exit code: $RC"
echo "===== DONE ====="
