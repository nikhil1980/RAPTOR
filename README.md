# RAPTOR

<ins>**R**</ins>eal-time <ins>**A**</ins>ll-<ins>**P**</ins>urpose <ins>**T**</ins>ransformer For <ins>**O**</ins>pen Vocabulary <ins>**R**</ins>ecognition or **RAPTOR** is a production-ready object detection stack that pairs a DINOv3 backbone with an RT-DETR detection head, trained on a LVIS + OpenImages-V7 mixture for closed-set performance, and augmented with a text-conditioned open-vocabulary head for discovering any user supplied label. It tackles long-tail imbalance via focal/varifocal losses and class-aware sampling. Furthermore, it ships with evaluation (COCO + LVIS APr/APc/APf), PR (Precision-Recall) plots, an ablation grid over ($\alpha$, $\gamma$), and a simple production inference pipeline that works on closed $\bigcup$ open labels per image.

To briefly summarize, RAPTOR is a NMS-free, real-time detection with DINOv3 features, open-vocabulary prompts, and long-tail robustness.

## ✨ Features ##

RAPTOR has following features:

### 1. Backbone ###
DINOv3 (teacher–student Semi Supervised Learning or SSL with Gram anchoring) as a Hugging Face Backbone.

### 2. Head ###
It uses RT-DETR head. So, it is a $\int$(hybrid encoder , query selection) and this makes it Non-Maximal-Selection free!

### 3. Mixute of Data (MoD) 
We construct data for training RAPTOR using the following sources:
 * LVIS v1 
 * OpenImages-V7 
 * Custom Data
   
We convert all data points to a unified COCO JSON (bbox-only).

### 4. Vocabulary 
RAPTOR works on both closed (over LVIS data labels) as well as open vocabulary set.  For Open-Vocabulary, we use region-wise CLIP/SigLIP matching on proposals and then couple it with a prompt constructed on top label(s).

### 5. Long-Tail Problem Mitigation
To overcome longtail problem due to skewed distribution of images to a majority of labels, we use ***focal/varifocal classification*** as well as  ***class-aware WeightedRandomSampler***.

### 6. Evaluation Metrics
We use the following evaluation metrics:
 * COCO mAP LVIS APr/APc/APf
 * PR curves
 * Calibration sanity checks

### 7. Ablation
We perform ablation in grid over Focal ($\alpha$, $\gamma$) with LVIS slice metrics.

### 8. Inference 
We perform inference over a a $\int$ closed-set, prompt-based open-set over union of unique labels above thresholds.

## 📦 Repository Structure
```
RAPTOR/
├─ data/
│  ├─ download_and_prepare.py      # COCO imgs, LVIS zips (fixed links), OpenImages via FiftyOne
│  └─ convert_and_merge.py         # Export OI→COCO; LVIS→COCO(bbox); merge to train/val
├─ train/
│  ├─ train_dinov3_rtdetr_ov.py    # Training (RT-DETR+DINOv3), focal/VFL, OV focal head, W&B
│  └─ run_ablation.py              # Ablation grid over (α,γ) → LVIS APr/APc/APf CSV
├─ eval/
│  ├─ eval_and_infer.py            # COCO eval + PR plots; LVIS eval helper
│  └─ zero_shot_prompt.py          # Prompted OV inference on single image
├─ serve/
│  ├─ predictor.py                 # Production predictor: closed + open union per image
│  └─ infer_cli.py                 # CLI wrapper
└─ resources/
   └─ open_vocab_lexicon.txt       # (Optional) Big label list for OV shortlisting
```

## 🚀 How to get Going?
1. Create environment (via venv or Conda). Make sure to accept the DINOv3 model license on the Hub before first download:
```
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.2" torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install "transformers[torch]" git+https://github.com/huggingface/transformers
pip install datasets accelerate evaluate pycocotools lvis wandb pillow tqdm
pip install fiftyone open_clip_torch
huggingface-cli login   # accept gated DINOv3 weights when prompted
```
2. Data Download & preparing mixture
```
# Download COCO images, LVIS zips (fixed URLs), and a manageable OI subset
python data/download_and_prepare.py

# Export OI → COCO; LVIS → COCO(bbox); unify/merge into a single label space
python data/convert_and_merge.py
```
Our data tree:
```
datasets/mixture/
  images/{train2017,val2017}/
  annotations/
    lvis/{lvis_v1_train.json, lvis_v1_val.json}
    coco_merged/
      instances_{lvis,oi}_{train,val}.json
      instances_{train,val}_merged.json   # <- used by training/eval
```

3. Train RAPTOR with focal/VFL + class-aware sampling + OV focal head.
```
export WANDB_PROJECT=<NAME YOUR PROJECT>
python train/train_dinov3_rtdetr_ov.py
```

At the end, you'll get the following:
 * DINOv3 backbone (frozen by default) plugged into RT-DETR
 * Focal/Varifocal classification, low eos weight (downweight background)
 * WeightedRandomSampler using inverse-frequency image weights
 * Optional OV head with focal-BCE on text matches
 * W&B logs with final weights at ```../runs/dinov3_rtdetr/final/``` 
 
4. Evaluate RAPTOR via COCO mAP, LVIS APr/APc/APf, PR plots
```
python eval/eval_and_infer.py
```

This gives the following output:
 * Predictions in file ```coco_detections_val.json``` 
 * PR curves saved per sampled class
 * Optional LVIS breakdown (```APr/APc/APf```) via ```lvis_eval()``` call
   
5. Inference over single image via Open-Vocabulary Prompting:
```
python eval/zero_shot_prompt.py
```

6. Production Inference over union of closed + open labels:
```
python serve/infer_cli.py --image path/to/image.jpg \
  --lexicon resources/open_vocab_lexicon.txt \
  --prompts "mic stand, stage light, ukulele"
```

This returns a JSON with *labels* (unique) and *detections* (boxes + scores + source).

## 🧪 Ablation on Focal ($\alpha$, $\gamma$) vs LVIS APr/APc/APf
We run a short sweep (default 3 epochs per point) and record LVIS slice metrics:
```
python train/run_ablation.py
```
This produces the following excel:
| $\alpha$ | $\gamma$ | AP   | APr  | APc  | APf  |
| :------: | :------: | ---: | ---: | ---: | ---: |
| 0.25     |  2.0     | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| 0.35     |  2.5     | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| 0.50     |  2.0     | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| $\dots$     | $\dots$     | $\dots$ | $\dots$ | $\dots$ | $\dots$ |

The purpose of this grid is to help us choose a setting that boosts rare classes without tanking frequent ones.

## 🔧 Significant Configuration Parameters
1. Image size: Set ```IMAGE_SIZE``` $\in$ [640–896] on the $\min$(ht, wid).
2. Focal/VFL: Set ```focal_loss_alpha```, ```focal_loss_gamma```, ```weight_loss_vfl```  as ```true```.
3. Background: Set ```eos_coefficient``` $\in$ [`1e-4`, `5e-5`]
4. Sampler: Set ```beta``` in ```compute_image_weights_from_json()``` as 0.5 for inverse sqrt and 1.0 for inverse.
5. Backbone: Set ```FREEZE_BACKBONE``` as ```true``` with optionally unfreeze last block for 5–10 epochs.
6. OV Head: Set thresholds (appropriately) ```OPEN_SCORE_THRESH``` in ```serve/predictor.py``` for lexicon + prompts

## 🧠 Reason to choose the combo: DINOv3 + RT-DETR
Besides, trying to learn object detection over an open vocabulary set, here are more plausible reasons for using this combination:

1. **DINOv3** provides stable dense features (thanks to **Gram anchoring**) and optional **text alignment** for zero-shot.
2. **RT-DETR** delivers **real-time** set-prediction without NMS, with a **hybrid encoder** and smarter query init—ideal for latency-sensitive pipelines.
3. Together, they give us a detector that’s **fast**, **tail-aware**, and works on **open-vocabulary**.

## 📊 Metrics
We used the following metrics:

1. COCO mAP (0.50:0.95) on the merged val set
2. LVIS overall AP and APr/APc/APf (rare/common/frequent)
3. PR curves + score calibration (by frequency slice)

## 🛡️ License
1. LVIS and OpenImages-V7 have their own licenses and usage terms—ensure compliance.
2. DINOv3 weights on Hugging Face may require license acceptance.
3. RAPTOR is released under [Apache 2](https://www.apache.org/licenses/LICENSE-2.0).

## 📚 References
We used the following references

1. Carion et al. “End-to-End Object Detection with Transformers (DETR)”
2. Zhao et al. “RT-DETR: DETRs Beat YOLOs on Real-Time Object Detection”
3. Siméoni et al. “DINOv3: Scaling Self-Supervised Learning with a Teacher of Features”
4. Gupta et al. “LVIS: A Dataset for Large Vocabulary Instance Segmentation”
5. Kuznetsova et al. “Open Images Dataset V4/V6/V7”
6. Zhang et al. “VarifocalNet: An IoU-aware Dense Object Detector”

## 🙌 Acknowledgements
This work builds on the incredible open-source efforts around DINO/DINOv2/3, RT-DETR, LVIS, and OpenImages, plus the Hugging Face and FiftyOne ecosystems.

## 📌 Citation

If you use **RAPTOR** in your research, please cite:

```bibtex
@software{Bhargava_RAPTOR_2025,
  author  = {Nikhil Bhargava},
  title   = {RAPTOR: Real-time All-Purpose Transformer for Open-vocabulary Recognition},
  year    = {2025},
  month   = {oct},
  version = {v0.1.0},
  url     = {https://github.com/nikhil1980/RAPTOR/},
  note    = {GitHub repository},
}
```

## 🗺️ Roadmap

- [ ] FastAPI microservice (batch inference, Triton-friendly)
- [ ] Better OV scoring with phrase prompts + synonyms
- [ ] Multi-resolution test-time scaling
- [ ] RFS (repeat-factor sampling) toggle and analysis

      
**RAPTOR** is ready to hunt. Clone, train, prompt, and deploy.
