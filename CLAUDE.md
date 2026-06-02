# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

RAPTOR — real-time open-vocabulary object detection:
- **DINOv3 ViT-B/16** backbone, frozen during training (last 2 blocks unfreeze at 80% of epochs at 0.1× LR)
- **RT-DETR** detection head (NMS-free, closed-set)
- **OV Head** — linear projection from RT-DETR decoder hidden states to SigLIP embedding space + focal BCE
- **Mixture training** on LVIS + OpenImages-V7 + COCO unified to COCO format (~1,579 classes)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login  # DINOv3 license gate
```

**Torch `>=2.6` required** — `transformers>=4.51` runs `check_torch_load_is_safe()` on `.pt` files (CVE-2025-32434). First run trains fine; resume dies in `trainer._load_optimizer_and_scheduler` on torch <2.6.

## Common Commands

| Task | Command |
|------|---------|
| Download + prepare datasets | `python data/download_and_prepare.py && python data/convert_and_merge.py` |
| Pre-training smoke test (always first) | `python train/smoke_test.py --config-file config.json` |
| Sanity dry-run on subsample | `python data/subsample_mixture.py`, then train with `RAPTOR_PATHS_{TRAIN,VAL}_JSON` overrides |
| Train | `python train/train_dinov3_rtdetr_ov.py --config-file config.json` |
| Evaluate (COCO + LVIS metrics) | `python eval/eval_and_infer.py --config-file config.json` |
| Zero-shot single image | `python eval/zero_shot_prompt.py --test-image <p> --test-tags "<labels>"` |
| Production inference | `python serve/infer_cli.py --config-file config.json --test-image <p> --test-tags "<labels>"` |
| SOTA final run (8ep, unfreeze@ep3, P2+LSJ+RFS+EMA) | `bash launch_sota.sh` (in tmux; uses the `raptor` conda env via explicit `PYBIN`) |
| Focal loss ablation sweep | `python train/run_ablation.py --config-file config.json` |

> `serve/infer_cli.py` takes `--config-file` / `--test-image` / `--test-tags` (the lexicon path comes from `RAPTOR_PATHS_LEXICON_PATH`, default `resources/open_vocab_lexicon.txt`). It does **not** take `--image/--lexicon/--prompts`. Run it with the `raptor` conda env (base `python` lacks `transformers`).

## Configuration

`config.json` → `common/env.py` flattens to env vars (`RAPTOR_TRAIN_BATCH_SIZE` etc.) → `common/config.py` `Config` reads them. **Shell env wins over JSON** (12-factor): `load_env_from_json(..., override=False)` skips keys already set, so `export RAPTOR_*` before launch overrides the file.

## Critical Gotchas

### Padded-frame coordinate convention
`CocoDetDataset.__getitem__` letterboxes via the RT-DETR processor then **bottom-right pads to a fixed S×S square** (`Config.IMAGE_SIZE.longest_edge`=640), rescaling GT boxes by `(W/S, H/S)`. Everything downstream — loss, predictions, IoU, post-processing — lives in [0,1] of the padded S×S frame, **not** the original image.

`post_process_object_detection(target_sizes=(orig_h, orig_w))` is **wrong** — squashes predictions along the shorter axis. Losses look fine; mAP collapses to ~1%.

**Correct pattern** — `_post_process_padded`, duplicated across `train/wandb_callbacks.py`, `eval/eval_and_infer.py`, `serve/predictor.py`:
1. Per image: `max_dim = max(orig_h, orig_w)`.
2. `post_process_object_detection(target_sizes=[(max_dim, max_dim), ...])`.
3. Clip to `[0, orig_w] × [0, orig_h]`, drop zero-area boxes.

Eval and serve must **also** reproduce the dataset's input pad (`_letterbox_pad`). Helpers are intentionally duplicated rather than imported so `eval/` and `serve/` don't take a W&B dependency.

`EndOfRunArtifactsCallback._log_confusion_pairs` also rescales **GT** cxcywh→xyxy by `(max_dim, max_dim)` so pred and GT IoU share a frame. Fix one side without the other → IoU=0, confusion stats vanish.

### Class-label space
Three places must agree on what a class index means:
1. **Model classifier output** — dense `[0, K)`, `K = num_labels`.
2. **Loss/matcher input** — every `class_labels` from `__getitem__` is dense `[0, K)`.
3. **pycocotools eval input** — **original** sparse COCO `category_id` (~1579). Use `cat_id_to_idx` / `idx_to_cat_id` to translate dense → original before `coco_gt.loadRes(...)`.

Mismatch surfaces as a deferred `IndexKernel.cu:93: index out of bounds` from inside `generalized_box_iou`. Set `CUDA_LAUNCH_BLOCKING=1` for a real trace.

### Failure signatures vs. under-training
**Geometry / class-id bug** (do not blame the model):
- `val/coco/map_50` near zero while `eval_loss` falls
- `map_small ≪ map_large`, both tiny
- `ap_rare/common/frequent` collapse together
- AR@100 collapses (`< 0.10`)

**Under-training** (wait, don't debug):
- `map_50 / map_75 ≈ 1.5×` *plus* AR@100 >0.25 and AR_large >0.4 — localization works, scoring not calibrated. Expected pre-unfreeze.

**Always check AR@100 alongside AP.** Healthy AR + low AP = scoring (more epochs); collapsed AR + low AP = geometry/ID bug.

### Healthy mid-training band
At ~5/12 epochs, backbone frozen on full 1,579-class mixture: AP ~0.05–0.08, AR@100 ~0.3, AR_large ~0.5+. The mAP step typically arrives after the 80%-unfreeze. Confirmed May 14–15 run: AP=0.057, AR_100=0.338 at epoch 5.0.

If mAP **plateaus then gently regresses** in this band across 3+ epochs (e.g. 0.073→0.068→0.059→0.060), check the unfreeze first via `grep BackboneUnfreezeCallback logs/train.log`:
- `unfroze last 2 blocks (...)` → good
- `transformer blocks not found; skipping` → `_blocks()` path bug

### Where training output lands
W&B's stdout hook captures HF Trainer's `{'loss': ..., 'grad_norm': ..., ...}` lines into `wandb/latest-run/files/output.log` — **not** shell-redirected `logs/train.log`. The redirected file gets `[transformers]` notices and `logger.*(...)` calls (incl. `BackboneUnfreezeCallback` log, `PeriodicCOCOEvalCallback` mAP). Tail `wandb/latest-run/files/output.log` for live loss.

### Step-based checkpointing & best-model selection
`eval_strategy` / `save_strategy` / `save_steps` / `load_best_model_at_end` are env-driven (`RAPTOR_TRAIN_EVAL_STRATEGY` / `_SAVE_STRATEGY` / `_SAVE_STEPS` / `_LOAD_BEST`); **defaults preserve the old per-epoch behavior.** Default `save_strategy="epoch"` on an ~8-epoch/~3-week run only writes the **first checkpoint ~30h in**, then every ~2 days — a crash before epoch 1 loses everything (June-1 SOTA run died with an empty `runs/dinov3_rtdetr_sota`, nothing to resume). `launch_sota.sh` now sets `SAVE_STRATEGY=steps`, `SAVE_STEPS=2880` (~2.5–3h apart), `EVAL_STRATEGY=epoch`, `LOAD_BEST=false`, `SAVE_TOTAL_LIMIT=20`.

**HF forces `eval_strategy==save_strategy` when `load_best_model_at_end=True`**, and per-epoch COCO eval (61k imgs) is far too costly to run as often as you want to checkpoint — so **frequent step-saves require `LOAD_BEST=false`**. Consequence: the end-of-run model is the *last* checkpoint, not the best; **pick the best checkpoint offline from the W&B per-epoch mAP curve** (best mAP lands in the late post-unfreeze epochs, which are the most-recently-saved and survive the rolling `SAVE_TOTAL_LIMIT` window). `LongTailTrainer._save_checkpoint` swaps EMA weights in for step-saved checkpoints (so a mid-epoch save isn't live/non-EMA weights).

### Throughput is GPU-compute-bound (not data/IO)
Measured 2026-06-02 on the SOTA recipe (DINOv3 ViT-B @800px + P2 FPN, 1 GPU): **GPU-0 holds 94–100% util**, CPU load ~5/40, disk idle (mixture is on local ext4, not the `/mnt/nas224` CIFS mount). ~2.8s/optim-step × 460,808 steps ≈ ~15 days compute + evals ≈ **~3 weeks for 8 epochs.** **Single-GPU throughput knobs barely move it:** turning OFF `RAPTOR_TRAIN_GRADIENT_CHECKPOINTING` and going batch-16×accum-2 (effective batch UNCHANGED at 32) gained only ~4% — per-step work is fixed at 32 imgs and cost is dominated by the *frozen* backbone forward; grad-ckpt only saved recompute on the minority trainable path. batch-16+no-ckpt does fit (38.8/49 GB). The only ~1.8–2× lever is the **2nd GPU** (often idle; `launch_sota.sh` pins `CUDA_VISIBLE_DEVICES=0`) via `accelerate launch` DDP — but DDP is **unvalidated** here (custom weighted sampler / EMA / unfreeze callbacks need a distributed smoke-test first).

### Resume-specific gotchas
**Stale `state.best_metric` silently throws away resumed training.** Switching `metric_for_best_model` or flipping `greater_is_better` between runs makes `_determine_best_metric` compare against a leftover from the old metric; `load_best_model_at_end=True` then reverts to the pre-resume checkpoint at train end. Null both before any such switch:
```python
import json, pathlib
p = pathlib.Path('runs/dinov3_rtdetr/checkpoint-<N>/trainer_state.json')
s = json.loads(p.read_text()); s['best_metric'] = s['best_model_checkpoint'] = None
p.write_text(json.dumps(s, indent=2))
```

**`BackboneUnfreezeCallback` on a late resume only trains the tail.** `on_epoch_begin` checks `state.epoch >= frac * num_train_epochs`. Resume at epoch=9.0 with frac=0.8 and `num_train_epochs=12` (threshold 9.6) → unfreeze fires at epoch 10.0, leaving few epochs at cosine-tail LR. Lower `RAPTOR_UNFREEZE_BACKBONE_FRAC` or bump `RAPTOR_TRAIN_EPOCHS`.

**Optimizer param-group count on resume — now handled automatically.** The unfreeze callback adds a 3rd param group when it fires, so a checkpoint saved *after* unfreeze has 3 groups while a fresh HF optimizer at resume (re-frozen at setup) has 2 → historically a `ValueError` in HF's optimizer-state load. `LongTailTrainer._reconcile_optimizer_for_resume` (called from `create_optimizer`) now reads the saved `optimizer.pt` param-group count and replays `apply_unfreeze` to rebuild the 3rd group *before* the load, then sets `unfreeze_cb._unfrozen=True` so `on_epoch_begin` won't add a duplicate. So **auto-resume across the unfreeze is safe** — no manual pre-unfreeze step needed. Validated end-to-end by `check_resume_test.sh` (reconcile ran, no param-group crash, phase B resumed + finished). Block/param iteration order in `apply_unfreeze` must stay identical to the original run (optimizer state matches by insertion order, not name).

**`RAPTOR_ACCELERATE_NUM_PROCESSES` is a no-op under `python train/...`.** Only `accelerate launch` reads it. Single-GPU `python` uses `RAPTOR_DATALOADER_NUM_WORKERS`.

## Architecture

### Training (`train/train_dinov3_rtdetr_ov.py`)
- `CocoDetDataset` — COCO loader. Builds `cat_ids_sorted` / `cat_id_to_idx` / `idx_to_cat_id`. `__getitem__` filters degenerate boxes / unknown categories, processor-resizes, square-pads to `Config.IMAGE_SIZE.longest_edge`, rescales boxes by `(sx, sy)`. Label tensors **not squeezed** — single-annotation images would collapse `(1,4)`→`(4,)`.
- `build_idmaps()` — dense `id2label = {idx: c["name"] for idx, c in enumerate(sorted(categories, key=id))}`. Must match dataset's dense space.
- `DinoV3FPNBackbone` — ViTDet-style adapter: DINOv3 stride-16 → strides 8/16/32 via ConvTranspose2d 2×, 1×1, stride-2 3×3. Swapped into `model.model.backbone` after RT-DETR is built with a placeholder ResNet for sizing. Patch tokens sliced as `last[:, -N:, :]` to skip register tokens.
- `OVHead` — linear `decoder_hidden → SigLIP embed` + learned CLIP-style `logit_scale`.
- `RAPTORHungarianMatcher` — local matcher (focal cost + L1 + GIoU). Local copy because `RTDetrHungarianMatcher` isn't import-stable across `transformers` releases.
- `ModelWithOV` — wraps `RTDetrForObjectDetection`. Forces `output_hidden_states=True` in training; matcher assigns queries → GT class IDs, used as multi-hot focal-BCE target for OV head (NOT closed-set argmax — that's self-distillation). `text_embeds` is non-persistent buffer. Exposes `num_parameters(...)` for HF's `WandbCallback`. Forward is `**batch` (kwargs-only), defeating `find_labels()` — **`TrainingArguments.label_names=["labels"]` must be set explicitly** or `eval_loss` is silently never emitted. **OV-head loss fires only when `self.training=True`**: `train_loss = RT-DETR + OV`, `eval_loss = RT-DETR only`; not directly comparable.
- `build_text_embeddings()` — uses `SiglipTextModel.pooler_output` directly (batched). `SiglipModel.get_text_features` now returns `BaseModelOutputWithPooling`, not a tensor.
- `compute_image_weights_from_json()` — inverse-frequency class-aware sampler (β=0.5 or 1.0).
- HF `Trainer` with **bf16** (fp16 destabilizes RT-DETR's focal/VFL), gradient accumulation, cosine LR. W&B gated on `RAPTOR_WANDB_PROJECT_ENABLED`.
- `torch.multiprocessing.set_sharing_strategy("file_system")` at module top — default `file_descriptor` burns 2 FDs per shared tensor → EMFILE under `num_workers≥8` + `pin_memory=True`. Pair with `ulimit -n 1048576` and free GB in `/dev/shm`.
- Dataloader workers via `RAPTOR_DATALOADER_NUM_WORKERS` (default 8), decoupled from DDP world-size. `persistent_workers=True` on train loader.

### Smoke test (`train/smoke_test.py`)
**Always run before a long run.** Validates: dataset/processor build, model assembly, DINOv3 fully frozen, FPN stride-8/16/32 shapes, gradients reach OV head + FPN but not body, single-batch overfit >50% in 300 steps (best predictor of learning), eval forward finite, sampler weights non-uniform. Fails fast.

### Sanity dry-run on subsample
`data/subsample_mixture.py` emits `instances_train_sub5k.json` / `instances_val_sub1k.json` (greedy category-coverage + random fill), preserving all 1,579 categories. Pair with `RAPTOR_PATHS_{TRAIN,VAL}_JSON` + distinct `RAPTOR_PATHS_MODEL_DIR` + `RAPTOR_TRAIN_AUTO_RESUME=false` to exercise the full train→eval→callback→artifact loop in ~1 hour. Smoke checks static correctness; this checks the dynamic loop (checkpoint save/load, mid-run unfreeze, W&B artifact upload).

### W&B callbacks (`train/wandb_callbacks.py`)
Attached when `RAPTOR_WANDB_PROJECT_ENABLED`:
- `PeriodicCOCOEvalCallback` — full COCO mAP every epoch. Hooks `on_evaluate` (NOT `on_epoch_end`) so it mutates the `metrics` dict that `Trainer._determine_best_metric` reads next. Pushes flat `val/coco/{map,map_50,map_75,map_small/medium/large,ar_100,ap_rare/common/frequent}` AND injects `eval_*` mirrors into `metrics` so `metric_for_best_model="eval_val/coco/map"` resolves. Translates dense → original `category_id` via `idx_to_cat_id` before `coco_gt.loadRes`. **Switching back to `on_epoch_end` makes Trainer miss the COCO keys → `KeyError: 'eval_val/coco/map'` ~24h in (May 18 incident).** **Caps detections to `max_dets_per_image=100` (top-K by score) per image before `loadRes`** — metric-neutral (COCO mAP scores only `maxDets[-1]=100`/image) but essential: with a low `RAPTOR_EVAL_SCORE_THRESH` (e.g. 0.001) on an uncalibrated early model ~all 300 RT-DETR queries/image pass, so 61k val imgs × ~299 ≈ 18M detection dicts OOM-kill (SIGKILL, no traceback) inside `loadRes`/`COCOeval` over the 1565-class space — died right after logging `... N predictions`, before `wandb.log`, so `eval_loss` appears but no `val/coco/*` graph (May 31 SOTA incident).
- `SamplePredictionsCallback` — every 2 epochs, inference on a fixed pool of 8 val images.
- `BackboneUnfreezeCallback` — at 80% of `num_train_epochs`, unfreezes last 2 DINOv3 blocks at 0.1× LR by adding a 3rd optimizer param group. `_blocks()` must search multiple attribute paths — HF's `DINOv3ViTModel` names its encoder `model` (blocks at `body.model.layer`), so the path list includes `model.layer` / `model.layers`. Path mismatch fails *open* with `WARNING transformer blocks not found; skipping` and sets `_unfrozen=True` — backbone stays frozen for the run; symptom is mAP plateau in frozen band, then gentle regression as cosine LR decays. Correct unfreeze logs `unfroze last 2 blocks (34 tensors, lr=...)` (34 = 17 trainable/block × 2 for ViT-B). Expect a **loss bump (~0.5–1.0) for ~1–2h** after — not a regression.
- `EndOfRunArtifactsCallback` — on train end, uploads `final/` as a versioned Artifact + tables for mAP-history, top/bottom-50 per-class AP, AP-by-frequency-slice, top-30 confused pairs. `_name_for(orig_cid)` translates pycocotools' original-id keys back through `cat_id_to_idx` → `id2label`.

### Production inference (`serve/predictor.py`)
**`build_and_load_trained_model(model_dir, device)` — the ONLY correct way to load a RAPTOR checkpoint for serve.** `Predictor.__init__` calls it instead of `RTDetrForObjectDetection.from_pretrained(model_dir)`. The naive `from_pretrained` builds a vanilla ResNet-backbone RT-DETR whose keys lack the `base.*`/`ov_head.*` prefixes the `ModelWithOV` checkpoint uses → **every trained tensor is dropped and the model runs on random init** (130 garbage `LABEL_0/LABEL_1` boxes, the May-29 symptom). The loader reuses training's `DinoV3FPNBackbone`/`OVHead`/`ModelWithOV` classes (so key names match), **auto-detects** `num_labels` (from `base.model.decoder.class_embed.0.bias`) and `use_p2` (presence of `.backbone.lat4.`) from the checkpoint, replicates `build_model`'s `RTDetrConfig` block (**keep in sync**), passes zeros for the non-persistent `text_embeds` buffer (serve OV path uses OpenCLIP, not the OV head), and `load_state_dict(strict=False)`. Logs `matched N/N checkpoint tensors` — anything less than 100% is a silent arch drift. id2label is **self-resolved** by scanning the annotations dir smallest-file-first for a json whose `categories` count == checkpoint `num_labels` (the merged space drifted 1579→1565, so `Config.VAL_JSON` no longer matches the old checkpoint — it finds `instances_val_sub1k.bak.json`); hard-fails if none match.

Two-stage `Predictor.predict()`:
1. **Closed-set** — RT-DETR head → boxes above `CLOSED_SCORE_THRESH=0.30` + per-label NMS. Uses `_letterbox_pad` + `_post_process_padded` matching training input. OV path crops `image` with these boxes before CLIP, so a squashed box poisons every OV label.
2. **Open-vocab** — global image-CLIP similarity shortlists lexicon to top-`TOPK_CLIP_LEXICON=200`; each crop matched against text embeddings above `OPEN_SCORE_THRESH=0.28` cosine.
3. Merged + deduped via `IOU_MERGE_THRESH=0.55`.

**Serve env overrides** (read in `Predictor.__init__`, env wins over `config.py` defaults — no code edit to tune):
- `RAPTOR_SERVE_IMAGE_SHORT` — square input edge; re-binds `Config.IMAGE_SIZE`. **MUST match the checkpoint's training resolution** (640 for the old run, 800 for SOTA) or confidence drops and detections vanish. Serve does *not* auto-honor `RAPTOR_TRAIN_IMAGE_SHORT` (that's a train-`main()`-only rebind).
- `RAPTOR_CLOSED_SCORE_THRESH` / `RAPTOR_OPEN_SCORE_THRESH` — override the 0.30 / 0.28 defaults. Under-calibrated checkpoints (e.g. the old May-21 model tops out ~0.20) return 0 detections at 0.30; demo it at `closed=0.10 open=0.18`. The SOTA model should score higher and need no override.

`ClipHelper` — OpenCLIP `ViT-L-14` wrapper. **Must** instantiate with `model_name=Config.CLIP_MODEL, pretrained=Config.CLIP_SOURCE`; missing either → runtime `TypeError` when `use_openclip=True` (default).

### Evaluation (`eval/`)
- `eval_and_infer.py` — inference on val, COCO mAP@0.5:0.95 + LVIS APr/APc/APf, PR plots. Local `_letterbox_pad` + `_post_process_padded`. Translates dense → original `category_id` via `idx_to_cat_id`.
- `zero_shot_prompt.py` — demo OV inference with user-supplied prompts.

### Data pipeline (`data/`)
- `download_and_prepare.py` — downloads COCO, LVIS, OpenImages-V7 via FiftyOne.
- `convert_and_merge.py` — converts OpenImages→COCO, LVIS→COCO, unifies category space. Output: `datasets/mixture/{images,annotations}/`.

## Tunable env knobs (training)

| Env var | Default | Purpose |
|---------|---------|---------|
| `RAPTOR_TRAIN_BF16` | `true` | bf16 over fp16. Auto-fallback if no GPU support. fp16 destabilizes RT-DETR focal/VFL. |
| `RAPTOR_TRAIN_METRIC_FOR_BEST` | `eval_loss` | Best-checkpoint metric. `eval_val/coco/*` keys exist only when `RAPTOR_WANDB_PROJECT_ENABLED=true`; `train()` fail-fasts at startup if unreachable. |
| `RAPTOR_TRAIN_GREATER_IS_BETTER` | `false` | `true` for mAP, `false` for loss. |
| `RAPTOR_TRAIN_SAVE_TOTAL_LIMIT` | `3` | Max checkpoints on disk. |
| `RAPTOR_TRAIN_SAVE_STRATEGY` | `epoch` | `epoch` or `steps`. `steps` (+ `SAVE_STEPS`) gives crash-safety between the expensive per-epoch COCO evals on multi-day runs. See "Step-based checkpointing" gotcha. |
| `RAPTOR_TRAIN_SAVE_STEPS` | `500` | Optimizer steps between saves when `SAVE_STRATEGY=steps`. `launch_sota.sh` uses `2880` (≈1/20 epoch, ~2.5–3h apart). |
| `RAPTOR_TRAIN_EVAL_STRATEGY` | `epoch` | `epoch` or `steps`. Keep `epoch` — per-epoch COCO eval (61k val imgs) is too costly to run step-wise. |
| `RAPTOR_TRAIN_LOAD_BEST` | `true` | `load_best_model_at_end`. **Must be `false` when `SAVE_STRATEGY=steps` while `EVAL_STRATEGY=epoch`** (HF requires the two strategies match when `true`). Then best checkpoint is chosen offline from W&B per-epoch mAP. |
| `RAPTOR_EVAL_BATCH_SIZE` | `8` | COCO-eval inference batch size. |
| `RAPTOR_EVAL_SCORE_THRESH` | `0.05` | Min detection score in COCO eval. `launch_sota.sh` sets `0.001` (recall sweep) — safe for offline `eval_and_infer.py` and now safe in-loop too thanks to `PeriodicCOCOEvalCallback`'s top-100/image cap (pre-cap it OOM-killed epoch-1 eval). |
| `RAPTOR_WANDB_SAMPLE_PREDS_EVERY` | `2` | Epoch cadence for sample-prediction viz. |
| `RAPTOR_WANDB_NUM_SAMPLE_PREDS` | `8` | Fixed val images visualized per pass. |
| `RAPTOR_UNFREEZE_BACKBONE_FRAC` | `0.8` | Epoch fraction for unfreeze. `>=1.0` disables. |
| `RAPTOR_UNFREEZE_BACKBONE_BLOCKS` | `2` | Last-N blocks to unfreeze. |
| `RAPTOR_UNFREEZE_BACKBONE_LR_MULT` | `0.1` | LR mult for unfrozen backbone params. |
| `RAPTOR_DATALOADER_NUM_WORKERS` | `8` | DataLoader workers. Decoupled from `RAPTOR_ACCELERATE_NUM_PROCESSES` — sharing causes EMFILE under `pin_memory`. |
| `RAPTOR_PATHS_TRAIN_JSON` | `Config.TRAIN_JSON` | Override train JSON. Applied in `main()` after `load_env_from_json()`. |
| `RAPTOR_PATHS_VAL_JSON` | `Config.VAL_JSON` | Same for val. |
| `RAPTOR_PATHS_MODEL_DIR` | `runs/dinov3_rtdetr` | Checkpoints, `final/`, W&B artifacts. Override for sanity runs. |
| `RAPTOR_TRAIN_AUTO_RESUME` | `true` | Auto-resume from latest `checkpoint-*`. `false` for clean restart. |
