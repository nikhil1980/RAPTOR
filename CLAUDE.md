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
huggingface-cli login  # required for DINOv3 license gate
```

**Torch floor `>=2.6`** — `transformers>=4.51` runs `check_torch_load_is_safe()` on `.pt` files (CVE-2025-32434). `model.safetensors` is fine, but `optimizer.pt` / `scheduler.pt` block resume on torch <2.6. The first run trains and writes checkpoints; the *second* (resume) launch dies in `trainer._load_optimizer_and_scheduler` with `ValueError: Due to a serious vulnerability issue in torch.load, ...`.

## Common Commands

| Task | Command |
|------|---------|
| Download + prepare datasets | `python data/download_and_prepare.py && python data/convert_and_merge.py` |
| Pre-training smoke test (always run first) | `python train/smoke_test.py --config-file config.json` |
| Sanity dry-run on subsample | `python data/subsample_mixture.py`, then train with `RAPTOR_PATHS_{TRAIN,VAL}_JSON` overrides |
| Train | `python train/train_dinov3_rtdetr_ov.py --config-file config.json` |
| Evaluate (COCO + LVIS metrics) | `python eval/eval_and_infer.py --config-file config.json` |
| Zero-shot single image | `python eval/zero_shot_prompt.py --test-image <p> --test-tags "<labels>"` |
| Production inference | `python serve/infer_cli.py --image <p> --lexicon resources/open_vocab_lexicon.txt --prompts "<labels>"` |
| Focal loss ablation sweep | `python train/run_ablation.py --config-file config.json` |

## Configuration

`config.json` → `common/env.py` flattens into env vars (`RAPTOR_TRAIN_BATCH_SIZE` etc.) → `common/config.py` `Config` dataclass reads them. **Shell env wins over JSON** (12-factor): `load_env_from_json(..., override=False)` skips keys already in `os.environ`, so `export RAPTOR_*` before launch overrides the file without editing it.

## Critical Gotchas

### Padded-frame coordinate convention
`CocoDetDataset.__getitem__` letterboxes via the RT-DETR processor then **bottom-right pads to a fixed S×S square** (`Config.IMAGE_SIZE.longest_edge`=640), rescaling GT boxes by `(W/S, H/S)`. After this, **everything downstream of the model** — loss, predictions, IoU, post-processing — lives in [0,1] of the padded S×S frame, **not** the original image.

The natural-looking call `post_process_object_detection(target_sizes=(orig_h, orig_w))` is **wrong** here: it squashes predictions along the shorter axis by `min(h,w)/max(h,w)` (a 16:9 image gets squashed to 56% along x). Train_loss and eval_loss still look healthy (self-consistent in the wrong frame), but every reported mAP collapses to ~1%.

**Correct pattern** — `_post_process_padded`, duplicated across `train/wandb_callbacks.py`, `eval/eval_and_infer.py`, `serve/predictor.py`:
1. Per image: `max_dim = max(orig_h, orig_w)`.
2. `post_process_object_detection(target_sizes=[(max_dim, max_dim), ...])`.
3. Clip boxes to `[0, orig_w] × [0, orig_h]`, drop zero-area boxes (predictions that fell in the pad).

Eval and serve must **also** reproduce the dataset's input pad (`_letterbox_pad`). Feeding raw variable-aspect tensors is a distribution shift the model never saw. The helpers are intentionally duplicated rather than imported so `eval/` and `serve/` don't take a W&B dependency.

`EndOfRunArtifactsCallback._log_confusion_pairs` also rescales **GT** cxcywh→xyxy by `(max_dim, max_dim)`, not `(orig_w, orig_h)`, so prediction- and GT-side IoU live in the same frame. Fix one side without the other → IoU=0 everywhere, confusion stats vanish.

### Class-label space
Three places must agree on what a class index means:
1. **Model classifier output** — dense `[0, K)`, `K = num_labels`.
2. **Loss/matcher input** — every `class_labels` from `__getitem__` is dense `[0, K)`.
3. **pycocotools eval input** — **original** COCO `category_id` (sparse, up to ~1579). `CocoDetDataset` stores `cat_id_to_idx` / `idx_to_cat_id`; eval callbacks and `eval/eval_and_infer.py` must translate dense → original before `coco_gt.loadRes(...)`. Forgetting this collapses every metric to ~0 even with correct boxes — same visible failure as the coordinate bug, different root cause.

A class-id mismatch surfaces as a deferred `IndexKernel.cu:93: index out of bounds` from inside `generalized_box_iou` (CUDA errors are async). Set `CUDA_LAUNCH_BLOCKING=1` to get a trace at the real line.

### Failure signatures vs. under-training
Geometry / class-id bugs (do not blame the model):
- `val/coco/map_50` near zero while `eval_loss` is falling
- `map_small ≪ map_large` with both still tiny — small objects can't survive the coordinate squish
- `ap_rare`, `ap_common`, `ap_frequent` collapse together — geometry, not class imbalance
- AR@100 also collapses (`AR_100 < 0.10`) — the matcher isn't finding objects at all

Looks similar but is actually under-training (wait, don't debug):
- `map_50 / map_75 ≈ 1.5×` *plus* AR@100 healthy (>0.25) and AR_large healthy (>0.4) — localization is working, score calibration isn't yet. Expected mid-training before the backbone unfreeze.

**Always check AR@100 alongside AP.** Healthy AR with low AP = scoring calibration (give it more epochs); collapsed AR + low AP = real geometry/ID bug.

### What healthy mid-training looks like
At ~5/12 epochs with the backbone still frozen on the full 1,579-class mixture, expect roughly AP ~0.05–0.08, AR@100 ~0.3, AR_large ~0.5+. The mAP step typically arrives after the 80%-of-epochs backbone unfreeze. Confirmed pattern on the May 14–15 resumed run: AP=0.057, AR_100=0.338 at epoch 5.0.

If mAP **plateaus then gently regresses** in this band across 3+ consecutive epoch-end evals (e.g. epoch 6→9 going 0.073→0.068→0.059→0.060), the unfreeze is the first thing to check — not the model. Confirm via `grep BackboneUnfreezeCallback logs/train.log` for either `unfroze last 2 blocks (...)` (good) or `transformer blocks not found; skipping` (path bug — see the `_blocks()` note above).

### Where training output actually lands
When W&B is active, HF Trainer's `{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` log lines are captured by W&B's stdout hook and land in `wandb/latest-run/files/output.log` — **not** in your shell-redirected `logs/train.log`. The redirected file still receives `[transformers] ...` notices and your own `logger.*(...)` calls (including the `BackboneUnfreezeCallback` log line and `PeriodicCOCOEvalCallback` mAP printouts), but grepping `logs/train.log` for `'loss'` mid-run will return zero matches even when training is healthy. Tail `wandb/latest-run/files/output.log` for live loss/grad_norm.

### Resume-specific gotchas
**Stale `state.best_metric` silently throws away the resumed training.** `trainer_state.json` persists `best_metric` + `best_model_checkpoint`. If you change `metric_for_best_model` or flip `greater_is_better` between runs (e.g. `eval_loss`+false → `eval_val/coco/map`+true), `_determine_best_metric` compares the new value to a leftover from the *old* metric — the comparison is nonsensical, the best-tracker never updates, and `load_best_model_at_end=True` quietly reverts to the *pre-resume* checkpoint at train end. Before any such switch, null both fields in the resume checkpoint:
```python
import json, pathlib
p = pathlib.Path('runs/dinov3_rtdetr/checkpoint-<N>/trainer_state.json')
s = json.loads(p.read_text()); s['best_metric'] = s['best_model_checkpoint'] = None
p.write_text(json.dumps(s, indent=2))
```

**`BackboneUnfreezeCallback` on a late resume only trains the tail.** `on_epoch_begin` checks `state.epoch >= frac * num_train_epochs`. Resuming at `state.epoch=9.0` with frac=0.8 and `num_train_epochs=12` (threshold 9.6) means unfreeze fires at the *next* epoch boundary (state.epoch=10.0), leaving exactly `num_train_epochs - ceil(frac*num_train_epochs)` epochs unfrozen — at the cosine-tail LR. For a meaningful unfreeze on a resume, either lower `RAPTOR_UNFREEZE_BACKBONE_FRAC` so it fires immediately, or bump `RAPTOR_TRAIN_EPOCHS` to extend the remaining schedule.

**Optimizer param-group count must match on resume.** The callback appends a 3rd `param_group` for backbone params when it fires. If a checkpoint was saved *after* the unfreeze, its `optimizer.pt` has 3 groups; the fresh optimizer HF creates at resume (backbone re-frozen at setup) has 2 → `Optimizer.load_state_dict` raises `ValueError`. If you hit this, either resume from a checkpoint saved *before* the unfreeze, or pre-unfreeze the same blocks at setup time so the fresh optimizer has 3 groups too.

**`RAPTOR_ACCELERATE_NUM_PROCESSES` is a no-op under `python train/...`.** Only `accelerate launch` reads it. Setting it on a single-GPU `python` run does nothing — the dataloader path uses `RAPTOR_DATALOADER_NUM_WORKERS`.

## Architecture

### Training (`train/train_dinov3_rtdetr_ov.py`)
- `CocoDetDataset` — COCO-format loader. Builds `cat_ids_sorted` / `cat_id_to_idx` / `idx_to_cat_id` at init. `__getitem__` filters degenerate boxes (`w<=0` or `h<=0`) and unknown categories, processor-resizes, then square-pads to `Config.IMAGE_SIZE.longest_edge` and rescales boxes by `(sx, sy)`. Label tensors are **not squeezed** — single-annotation images would collapse `(1,4)`→`(4,)` and break box-axis indexing.
- `build_idmaps()` — dense `id2label = {idx: c["name"] for idx, c in enumerate(sorted(categories, key=id))}`. Must match the dataset's dense space.
- `DinoV3FPNBackbone` — ViTDet-style adapter: DINOv3's single stride-16 output → strides 8/16/32 via ConvTranspose2d 2×, 1×1, stride-2 3×3. Built once and swapped into `model.model.backbone` after RT-DETR is constructed with a placeholder ResNet for sizing. Patch tokens sliced as `last[:, -N:, :]` to skip register tokens at the front.
- `OVHead` — linear `decoder_hidden → SigLIP embed` + learned CLIP-style `logit_scale`.
- `RAPTORHungarianMatcher` — local matcher (focal cost + L1 + GIoU, via `scipy.optimize.linear_sum_assignment`). Local copy because `RTDetrHungarianMatcher` is not import-stable across `transformers` releases.
- `ModelWithOV` — wraps `RTDetrForObjectDetection`. Forces `output_hidden_states=True` in training; runs the matcher over `(logits, pred_boxes)` vs GT to assign queries → GT class IDs, then uses those as the multi-hot focal-BCE target for the OV head (NOT the closed-set head's own argmax — that's self-distillation). `text_embeds` is a non-persistent buffer so it tracks device/dtype. Exposes `num_parameters(only_trainable, exclude_embeddings)` because HF's `WandbCallback` calls it. Forward is `**batch` (kwargs-only), defeating HF's `find_labels()` — `TrainingArguments.label_names=["labels"]` must be set explicitly or `eval_loss` is silently never emitted. **OV-head loss fires only when `self.training=True`**, so `train_loss = RT-DETR + OV`, `eval_loss = RT-DETR only`; they're not directly comparable and `metric_for_best_model=eval_loss` selects on closed-set loss alone.
- `build_text_embeddings()` — uses `SiglipTextModel.pooler_output` directly (batched) rather than `SiglipModel.get_text_features` (now returns `BaseModelOutputWithPooling`, not a tensor). Halves memory by not loading the vision tower.
- `compute_image_weights_from_json()` — inverse-frequency class-aware sampler (β=0.5 or 1.0).
- HF `Trainer` with **bf16** (fp16 destabilizes RT-DETR's focal/VFL), gradient accumulation, cosine LR. W&B gated on `RAPTOR_WANDB_PROJECT_ENABLED`.
- `torch.multiprocessing.set_sharing_strategy("file_system")` at module top — the default `file_descriptor` strategy burns 2 FDs per shared tensor and dies with EMFILE under `num_workers≥8` + `pin_memory=True` within seconds. Pair with `ulimit -n 1048576` and keep several GB free in `/dev/shm`.
- Dataloader workers decoupled from DDP world-size via `RAPTOR_DATALOADER_NUM_WORKERS` (default 8). Sharing `RAPTOR_ACCELERATE_NUM_PROCESSES` for both multiplies fanout (world_size × num_workers) and triggers EMFILE under `pin_memory`. `persistent_workers=True` on the train loader so workers aren't torn down each epoch.

### Smoke test (`train/smoke_test.py`)
**Always run before a long run.** Validates: dataset/processor build, model assembly, DINOv3 fully frozen, FPN stride-8/16/32 shapes, gradients reach OV head + FPN but not the body, model can overfit a single batch >50% in 300 steps (best single predictor of "training will learn"), eval forward is finite, sampler weights non-uniform. Fails fast on first failed assertion.

### Sanity dry-run on subsample
`data/subsample_mixture.py` emits `instances_train_sub5k.json` / `instances_val_sub1k.json` (greedy category-coverage + random fill), preserving all 1,579 categories so model shape matches production. Pair with `RAPTOR_PATHS_{TRAIN,VAL}_JSON` + distinct `RAPTOR_PATHS_MODEL_DIR` + `RAPTOR_TRAIN_AUTO_RESUME=false` to exercise the full train→eval→callback→artifact loop in ~1 hour. Smoke test checks static correctness; this checks the dynamic loop (checkpoint save/load, mid-run unfreeze, W&B artifact upload).

### W&B callbacks (`train/wandb_callbacks.py`)
Attached automatically when `RAPTOR_WANDB_PROJECT_ENABLED`:
- `PeriodicCOCOEvalCallback` — full COCO mAP every epoch. Hooks `on_evaluate` (NOT `on_epoch_end`) so it can mutate the `metrics` dict that `Trainer._determine_best_metric` reads immediately after. Pushes flat `val/coco/map`, `map_50`, `map_75`, `map_small/medium/large`, `ar_100`, `ap_rare/common/frequent` to W&B for dashboards, AND injects the same keys with `eval_` prefix into `metrics` so `metric_for_best_model="eval_val/coco/map"` resolves. Translates dense indices → original COCO `category_id` via `val_ds.idx_to_cat_id` before `coco_gt.loadRes`. **If you ever switch the hook back to `on_epoch_end`, the Trainer never sees the COCO keys and the next run dies with `KeyError: 'eval_val/coco/map'` ~24 h in — the symptom the May 18 run hit.**
- `SamplePredictionsCallback` — every 2 epochs, inference on a fixed pool of 8 val images, annotated PIL images to W&B.
- `BackboneUnfreezeCallback` — at 80% of `num_train_epochs`, unfreezes last 2 DINOv3 blocks at 0.1× LR by adding them to the existing optimizer. `_blocks()` searches multiple attribute paths because HF's `DINOv3ViTModel` names its encoder `model` (not `encoder`) — the actual blocks live at `body.model.layer`, so the path list must include `model.layer` / `model.layers`. A path mismatch fails *open* with `WARNING transformer blocks not found; skipping`, sets `_unfrozen=True`, and the backbone stays frozen for the entire run; symptom is mAP plateaus in the frozen-backbone band (AP~0.06–0.08, AR_100~0.3, AR_large~0.55) and then *gently regresses* epoch-over-epoch as cosine LR decays on a frozen feature extractor. A correct unfreeze logs `unfroze last 2 blocks (34 tensors, lr=...)` — 34 = 17 trainable tensors/block × 2 blocks for ViT-B. Expect a brief **loss bump (~0.5–1.0) for ~1–2 hours** after unfreeze fires while the backbone re-adapts; not a regression.
- `EndOfRunArtifactsCallback` — on train end, uploads `final/` as a versioned Artifact plus tables for mAP-history, top/bottom-50 per-class AP, AP-by-frequency-slice, top-30 confused pairs. `_name_for(orig_cid)` translates pycocotools' original-id keys back through `cat_id_to_idx` → `id2label`.

### Production inference (`serve/predictor.py`)
Two-stage `Predictor.predict()`:
1. **Closed-set** — RT-DETR head → boxes + labels above `CLOSED_SCORE_THRESH=0.30` + per-label NMS. Uses `_letterbox_pad` + `_post_process_padded` matching the training input pipeline. The OV path crops `image` with these boxes before feeding CLIP, so a squashed box silently poisons every OV label.
2. **Open-vocab** — global image-CLIP similarity shortlists the lexicon to top-`TOPK_CLIP_LEXICON=200`; each box crop is embedded and matched against text embeddings above `OPEN_SCORE_THRESH=0.28` cosine sim.
3. Both paths merged and deduped via `IOU_MERGE_THRESH=0.55`.

`ClipHelper` — OpenCLIP `ViT-L-14` wrapper. Must be instantiated with `model_name=Config.CLIP_MODEL, pretrained=Config.CLIP_SOURCE`; missing either is a runtime `TypeError` exercised only when `use_openclip=True` (the default).

### Evaluation (`eval/`)
- `eval_and_infer.py` — runs inference on val, computes COCO mAP@0.5:0.95 + LVIS APr/APc/APf, saves PR plots. Uses local `_letterbox_pad` + `_post_process_padded` to match training input. Translates dense → original `category_id` via `idx_to_cat_id` before pycocotools.
- `zero_shot_prompt.py` — demo OV inference on single images with user-supplied prompts.

### Data pipeline (`data/`)
- `download_and_prepare.py` — downloads COCO, LVIS, OpenImages-V7 via FiftyOne.
- `convert_and_merge.py` — converts OpenImages→COCO, LVIS→COCO, unifies category space. Output: `datasets/mixture/{images,annotations}/`.

## Tunable env knobs (training)

| Env var | Default | Purpose |
|---------|---------|---------|
| `RAPTOR_TRAIN_BF16` | `true` | bf16 over fp16. Auto-falls-back if GPU lacks bf16. fp16 destabilizes RT-DETR's focal/VFL. |
| `RAPTOR_TRAIN_METRIC_FOR_BEST` | `eval_loss` | Best-checkpoint metric. `eval_val/coco/*` keys are injected into the Trainer's metrics dict by `PeriodicCOCOEvalCallback.on_evaluate` and only exist when `RAPTOR_WANDB_PROJECT_ENABLED=true`. `train()` fail-fasts at startup if the chosen metric isn't reachable; do NOT switch this to a mAP key without W&B on. |
| `RAPTOR_TRAIN_GREATER_IS_BETTER` | `false` | `true` for mAP, `false` for loss. |
| `RAPTOR_TRAIN_SAVE_TOTAL_LIMIT` | `3` | Max checkpoints on disk. |
| `RAPTOR_EVAL_BATCH_SIZE` | `8` | COCO-eval inference batch size. |
| `RAPTOR_EVAL_SCORE_THRESH` | `0.05` | Min detection score in COCO eval. |
| `RAPTOR_WANDB_SAMPLE_PREDS_EVERY` | `2` | Epoch cadence for sample-prediction viz. |
| `RAPTOR_WANDB_NUM_SAMPLE_PREDS` | `8` | Fixed val images visualized per pass. |
| `RAPTOR_UNFREEZE_BACKBONE_FRAC` | `0.8` | Epoch fraction after which last N DINOv3 blocks unfreeze. `>=1.0` disables. |
| `RAPTOR_UNFREEZE_BACKBONE_BLOCKS` | `2` | Last-N blocks to unfreeze. |
| `RAPTOR_UNFREEZE_BACKBONE_LR_MULT` | `0.1` | LR mult for unfrozen backbone params (vs `RAPTOR_TRAIN_LEARNING_RATE`). |
| `RAPTOR_DATALOADER_NUM_WORKERS` | `Config.NUM_WORKERS` (8) | DataLoader workers. **Decoupled from `RAPTOR_ACCELERATE_NUM_PROCESSES`** — sharing causes EMFILE under `pin_memory`. |
| `RAPTOR_PATHS_TRAIN_JSON` | `Config.TRAIN_JSON` | Override train JSON without editing config. Applied in `main()` after `load_env_from_json()` (Config attrs evaluate at import). |
| `RAPTOR_PATHS_VAL_JSON` | `Config.VAL_JSON` | Same for val. |
| `RAPTOR_PATHS_MODEL_DIR` | `runs/dinov3_rtdetr` | Checkpoints, `final/`, W&B artifacts. Override for sanity runs. |
| `RAPTOR_TRAIN_AUTO_RESUME` | `true` | Auto-resume from latest `checkpoint-*` in `RAPTOR_PATHS_MODEL_DIR`. `false` for sanity / clean restart. |
