# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAPTOR is a real-time open-vocabulary object detection system combining:
- **DINOv3 backbone** for dense feature extraction (frozen during training)
- **RT-DETR detection head** for NMS-free closed-set detection
- **OV Head** (linear projection + focal BCE) for open-vocabulary classification via CLIP/SigLIP text-image matching
- **Mixed dataset training** on LVIS + OpenImages-V7 unified into COCO format

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.2" torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
huggingface-cli login  # Required: accept DINOv3 license gate
```

## Common Commands

| Task | Command |
|------|---------|
| Download + prepare datasets | `python data/download_and_prepare.py` then `python data/convert_and_merge.py` |
| Pre-training smoke test (always run first) | `python train/smoke_test.py --config-file config.json` |
| Train | `python train/train_dinov3_rtdetr_ov.py --config-file config.json` |
| Evaluate (COCO + LVIS metrics) | `python eval/eval_and_infer.py --config-file config.json` |
| Zero-shot single image | `python eval/zero_shot_prompt.py --config-file config.json --test-image <path> --test-tags "<labels>"` |
| Production inference | `python serve/infer_cli.py --image <path> --lexicon resources/open_vocab_lexicon.txt --prompts "<labels>"` |
| Focal loss ablation sweep | `python train/run_ablation.py --config-file config.json` |

## Architecture

### Configuration System
`config.json` → `common/env.py` flattens nested JSON into env vars (e.g., `RAPTOR_TRAIN_BATCH_SIZE`) → `common/config.py` `Config` dataclass reads them. All scripts call `load_env_from_json()` as the first step.

**Shell env wins over JSON by default** (12-factor convention). `load_env_from_json(..., override=False)` skips keys already present in `os.environ`, so `export RAPTOR_TRAIN_BATCH_SIZE=48` from a launch script overrides whatever the JSON file says without editing it. Pass `override=True` only if you want the file to win.

### Training (`train/train_dinov3_rtdetr_ov.py`)
- `CocoDetDataset`: COCO-format dataset loader with multi-root image paths. Builds `cat_ids_sorted`, `cat_id_to_idx`, and `idx_to_cat_id` at init so raw COCO category IDs (1-indexed, sparse, up to ~1579) are remapped to a dense 0-indexed space `[0, K)` matching the model's classifier slots. `__getitem__` filters degenerate boxes (`w<=0` or `h<=0`) and unknown categories, applies the processor (letterbox resize), then **pads to a square** (`Config.IMAGE_SIZE.longest_edge`) and rescales boxes by `(sx, sy)` so all samples in a batch stack cleanly. Label tensors are NOT squeezed — single-annotation images would otherwise collapse `(1,4)` → `(4,)` and break box-axis indexing.
- `build_idmaps()`: returns dense 0-indexed `id2label = {idx: c["name"] for idx, c in enumerate(sorted(categories, key=id))}` and the inverse `label2id`. Must match the dataset's dense space — out-of-range class labels surface as deferred CUDA `IndexKernel.cu:93: index out of bounds` asserts inside the matcher's focal cost.
- `DinoV3FPNBackbone`: ViTDet-style multi-scale adapter that turns DINOv3 ViT's single stride-16 output into three feature maps at strides 8/16/32 for RT-DETR's HybridEncoder (ConvTranspose2d 2× / 1×1 / stride-2 3×3). Built once and swapped into `model.model.backbone` after RT-DETR is constructed with a placeholder ResNet for sizing. Patch tokens are sliced as `last[:, -N:, :]` so register tokens at the front are excluded.
- `OVHead`: Linear projection from RT-DETR decoder hidden states → SigLIP embedding space, with a learned CLIP-style `logit_scale`.
- `RAPTORHungarianMatcher`: Self-contained matcher (focal class cost + L1 box + GIoU, solved via `scipy.optimize.linear_sum_assignment`) keyed off `RTDetrConfig` cost weights. Written locally because `RTDetrHungarianMatcher` has moved across `transformers` releases and is not import-stable.
- `ModelWithOV`: Wraps `RTDetrForObjectDetection`. Forces `output_hidden_states=True` during training, runs `RAPTORHungarianMatcher` over `(logits, pred_boxes)` vs GT to assign queries → GT class IDs, and uses those as the multi-hot focal-BCE target for the OV head (NOT the closed-set head's own argmax — that would be self-distillation). `text_embeds` is registered as a non-persistent buffer so it tracks device/dtype. Defines an explicit `num_parameters(only_trainable, exclude_embeddings)` method because HF's `WandbCallback` calls `model.num_parameters()` which `nn.Module` doesn't provide.
- `build_text_embeddings()`: Encodes class names with `google/siglip-so400m-patch14-384` via `SiglipTextModel.pooler_output` (batched), then frees the text encoder. `SiglipModel.get_text_features` returns `BaseModelOutputWithPooling` (not a tensor) in current `transformers` releases, so we use `SiglipTextModel` directly — also halves memory because we don't load the vision tower.
- `compute_image_weights_from_json()`: Inverse-frequency class-aware sampler (β=0.5 or 1.0).
- Training uses HF `Trainer` with bf16 (preferred over fp16 — RT-DETR's focal/VFL is unstable in fp16), gradient accumulation, cosine LR; W&B is gated on `RAPTOR_WANDB_PROJECT_ENABLED`.

### Pre-training smoke test (`train/smoke_test.py`)
**Always run before launching a long training run.** Validates: dataset/processor build, model assembly, DINOv3 body fully frozen, FPN produces correct stride-8/16/32 shapes, gradients reach the OV head and FPN but NOT the body, the model can overfit a single batch >50% in 300 steps (the single best predictor of "training will learn"), eval forward pass is finite, sampler weights are non-uniform. Fails fast and loud — first failed assertion exits non-zero.

### W&B logging callbacks (`train/wandb_callbacks.py`)
Four `TrainerCallback` classes attached automatically when `RAPTOR_WANDB_PROJECT_ENABLED` is on:
- `PeriodicCOCOEvalCallback` — full COCO mAP every epoch on val (`val/coco/map`, `map_50`, `map_75`, `map_small/medium/large`, `ap_rare/common/frequent`). Translates dense class indices back to original COCO category IDs via `val_ds.idx_to_cat_id` before submitting to `coco_gt.loadRes` — pycocotools matches on the original ID space. Stores history for end-of-run plots.
- `SamplePredictionsCallback` — every 2 epochs, runs inference on a fixed pool of 8 val images and logs annotated PIL images to W&B for qualitative inspection.
- `BackboneUnfreezeCallback` — at 80% of `num_train_epochs`, unfreezes the last 2 DINOv3 blocks at 0.1× LR by adding them to the existing optimizer.
- `EndOfRunArtifactsCallback` — on train end, uploads the `final/` model directory as a versioned W&B Artifact, plus tables for mAP-history, top/bottom-50 per-class AP, AP-by-frequency-slice, and top-30 most-confused class pairs. Per-class AP keys come back from pycocotools in original-COCO-id space; `_name_for(orig_cid)` translates them back through `cat_id_to_idx` → `id2label` to produce human-readable labels.

### Class-label space (gotcha)
Three places must agree on what a class index means:
1. **Model's classifier output**: dense `[0, K)` slots, where `K = num_labels`.
2. **Loss/matcher input**: every `class_labels` tensor coming out of `__getitem__` must be in the same dense `[0, K)` space.
3. **pycocotools eval input**: needs the **original** COCO category IDs (`category_id` in the JSON). The dataset stores `cat_id_to_idx` and `idx_to_cat_id` so callbacks can translate dense → original on the way out.

A mismatch shows up as `IndexKernel.cu:93: index out of bounds` from somewhere deep inside the matcher (often inside `generalized_box_iou`'s box-validity check, because CUDA errors are async/deferred). Set `CUDA_LAUNCH_BLOCKING=1` to get a non-deferred trace at the real failing line.

### Tunable env knobs (training)
| Env var | Default | What it controls |
|---------|---------|------------------|
| `RAPTOR_TRAIN_BF16` | `true` | Use bf16 instead of fp16 (auto-falls-back to fp16 if GPU lacks bf16). RT-DETR's focal/VFL is unstable in fp16. |
| `RAPTOR_TRAIN_METRIC_FOR_BEST` | `eval_loss` | Metric driving best-checkpoint selection. Set to `eval_val/coco/map` once mAP is logged. |
| `RAPTOR_TRAIN_GREATER_IS_BETTER` | `false` | Pair with the metric — `true` for mAP, `false` for loss. |
| `RAPTOR_TRAIN_SAVE_TOTAL_LIMIT` | `3` | Keep at most this many checkpoints on disk. |
| `RAPTOR_EVAL_BATCH_SIZE` | `8` | Inference batch size for the periodic COCO eval. |
| `RAPTOR_EVAL_SCORE_THRESH` | `0.05` | Min score for a detection to be counted in COCO eval. |
| `RAPTOR_WANDB_SAMPLE_PREDS_EVERY` | `2` | Epoch cadence for sample-prediction visualizations. |
| `RAPTOR_WANDB_NUM_SAMPLE_PREDS` | `8` | How many fixed val images to visualize each time. |
| `RAPTOR_UNFREEZE_BACKBONE_FRAC` | `0.8` | Fraction of total epochs after which to unfreeze the last N DINOv3 blocks. Set `>=1.0` to disable. |
| `RAPTOR_UNFREEZE_BACKBONE_BLOCKS` | `2` | Number of last transformer blocks to unfreeze. |
| `RAPTOR_UNFREEZE_BACKBONE_LR_MULT` | `0.1` | LR multiplier for unfrozen backbone params (relative to `RAPTOR_TRAIN_LEARNING_RATE`). |

### Data Pipeline (`data/`)
- `download_and_prepare.py`: Downloads COCO, LVIS, OpenImages-V7 via FiftyOne
- `convert_and_merge.py`: Converts OpenImages→COCO format, LVIS→COCO format, unifies category space
- Output lands in `datasets/mixture/{images,annotations}/`

### Production Inference (`serve/predictor.py`)
Two-stage detection via `Predictor.predict()`:
1. **Closed-set**: RT-DETR head → boxes + labels filtered by `CLOSED_SCORE_THRESH` (0.30) + per-label NMS
2. **Open-vocab**: Global image-CLIP similarity shortlists lexicon to top-200, then each box crop is embedded and matched against text embeddings above `OPEN_SCORE_THRESH` (0.28 cosine sim)
- `ClipHelper`: OpenCLIP `ViT-L-14` wrapper for text/image encoding
- Results from both paths are merged and deduplicated

### Key Thresholds (`serve/predictor.py`)
- `CLOSED_SCORE_THRESH = 0.30`
- `OPEN_SCORE_THRESH = 0.28`
- `IOU_MERGE_THRESH = 0.55`
- `TOPK_CLIP_LEXICON = 200`

### Evaluation (`eval/`)
- `eval_and_infer.py`: Runs inference on val set, computes COCO mAP@0.5:0.95 + LVIS APr/APc/APf, saves PR plots
- `zero_shot_prompt.py`: Demo script for OV inference on single images with user-supplied prompts
