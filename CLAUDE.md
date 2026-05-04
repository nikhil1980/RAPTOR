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
| Train | `python train/train_dinov3_rtdetr_ov.py --config-file config.json` |
| Evaluate (COCO + LVIS metrics) | `python eval/eval_and_infer.py --config-file config.json` |
| Zero-shot single image | `python eval/zero_shot_prompt.py --config-file config.json --test-image <path> --test-tags "<labels>"` |
| Production inference | `python serve/infer_cli.py --image <path> --lexicon resources/open_vocab_lexicon.txt --prompts "<labels>"` |
| Focal loss ablation sweep | `python train/run_ablation.py --config-file config.json` |

## Architecture

### Configuration System
`config.json` → `common/env.py` flattens nested JSON into env vars (e.g., `RAPTOR_TRAIN_BATCH_SIZE`) → `common/config.py` `Config` dataclass reads them. All scripts call `load_env_from_json()` as the first step.

### Training (`train/train_dinov3_rtdetr_ov.py`)
- `CocoDetDataset`: COCO-format dataset loader with multi-root image paths
- `OVHead`: Linear projection from RT-DETR decoder hidden states → SigLIP embedding space
- `ModelWithOV`: Wraps `RTDetrForObjectDetection`, adds OV focal BCE loss during training
- `build_text_embeddings()`: Encodes class names with `google/siglip-so400m-patch14-384` (one-time)
- `compute_image_weights_from_json()`: Inverse-frequency class-aware sampler (β=0.5 or 1.0)
- Training uses HF `Trainer` with fp16, gradient accumulation (4 steps), cosine LR, W&B logging

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
