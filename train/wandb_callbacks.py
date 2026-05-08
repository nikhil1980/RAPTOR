import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from common.logger import get_logger
""" User Modules """

"""
Comprehensive W&B logging callbacks for RAPTOR training.

Pieces:
    1. SamplePredictionsCallback   - every K epochs, run inference on a fixed pool
                                     of val images and log annotated images to W&B.
    2. PeriodicCOCOEvalCallback    - every epoch, run COCO mAP on val with per-class
                                     AP and per-frequency-slice (rare/common/frequent)
                                     metrics. Stores history for end-of-run plots.
    3. BackboneUnfreezeCallback    - at a configured epoch fraction, unfreeze the
                                     last N transformer blocks of DINOv3 with a
                                     reduced LR. Adds them to the existing optimizer.
    4. EndOfRunArtifactsCallback   - on train end, save model dir as a W&B artifact,
                                     log mAP-history table, log per-class AP charts
                                     (top/bottom 50, slice averages), top-K
                                     confusable class pairs.

@author: Nikhil Bhargava
@date: 2026-05-08
@license: Apache-2.0
"""

logger = get_logger(__name__)


# --------- visual helpers ----------
def _draw_boxes_pil(image: Image.Image, boxes_xyxy, scores, labels,
                    id2label: Dict[int, str], max_boxes: int = 20) -> Image.Image:
    """Draw the top-N boxes on a PIL image (in place); returns the image."""
    if len(boxes_xyxy) == 0:
        return image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    palette = ["#00ff66", "#ff8800", "#00ddff", "#ff44dd", "#ffee00", "#ff4444", "#aaff00"]
    order = np.argsort(-np.asarray(scores))[:max_boxes]
    for k, idx in enumerate(order):
        x1, y1, x2, y2 = (int(v) for v in boxes_xyxy[idx])
        color = palette[k % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{id2label.get(int(labels[idx]), str(int(labels[idx])))} {float(scores[idx]):.2f}"
        bb = draw.textbbox((x1, max(y1 - 14, 0)), text, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x1, max(y1 - 14, 0)), text, fill="black", font=font)
    return image


def _build_freq_slices(coco) -> Dict[str, List[int]]:
    """LVIS-style rare/common/frequent splits from per-class instance counts."""
    counts: Dict[int, int] = defaultdict(int)
    for a in coco.dataset.get("annotations", []):
        counts[a["category_id"]] += 1
    rare, common, frequent = [], [], []
    for cid in sorted(coco.cats.keys()):
        c = counts.get(cid, 0)
        if c < 10:
            rare.append(cid)
        elif c < 100:
            common.append(cid)
        else:
            frequent.append(cid)
    return {"rare": rare, "common": common, "frequent": frequent}


def _orig_size_from_label(lab: Dict[str, Any]) -> torch.Tensor:
    """Pull (h, w) from a label dict, tolerating naming variation across HF versions."""
    for key in ("orig_size", "original_size", "image_size"):
        if key in lab:
            return lab[key]
    # Fallback to model input size
    return torch.tensor([640, 640])


# --------- 1. sample predictions ----------
class SamplePredictionsCallback(TrainerCallback):
    """Every K epochs, log annotated predictions on a fixed pool of val images."""

    def __init__(self, val_ds, image_processor, id2label: Dict[int, str],
                 every_n_epochs: int = 2, num_samples: int = 8, score_thresh: float = 0.3):
        self.val_ds = val_ds
        self.image_processor = image_processor
        self.id2label = id2label
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.score_thresh = score_thresh
        # Fix the sample indices once so the same images appear at every checkpoint.
        self._fixed_indices = list(range(min(num_samples, len(val_ds))))
        self._cached_originals: Optional[List[Image.Image]] = None

    def _load_originals(self) -> List[Image.Image]:
        if self._cached_originals is None:
            self._cached_originals = []
            for i in self._fixed_indices:
                iminfo = self.val_ds.images[i]
                p = self.val_ds._resolve_path(iminfo["file_name"])
                self._cached_originals.append(Image.open(p).convert("RGB"))
        return self._cached_originals

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if int(state.epoch) % self.every_n_epochs != 0 or wandb.run is None:
            return
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        wb_imgs = []
        for idx, orig in zip(self._fixed_indices, self._load_originals()):
            sample = self.val_ds[idx]
            pix = sample["pixel_values"].unsqueeze(0).to(device)
            outputs = model(pixel_values=pix)
            tgt_sz = torch.tensor([[orig.height, orig.width]], device=device)
            res = self.image_processor.post_process_object_detection(
                outputs, threshold=self.score_thresh, target_sizes=tgt_sz)[0]
            annotated = _draw_boxes_pil(
                orig.copy(),
                res["boxes"].cpu().numpy(),
                res["scores"].cpu().numpy(),
                res["labels"].cpu().numpy(),
                self.id2label,
            )
            wb_imgs.append(wandb.Image(annotated, caption=f"val[{idx}] epoch={state.epoch:.1f}"))

        wandb.log({"val/sample_predictions": wb_imgs}, step=state.global_step)
        if was_training:
            model.train()


# --------- 2. periodic COCO eval ----------
class PeriodicCOCOEvalCallback(TrainerCallback):
    """
    Run a full COCO mAP evaluation on the val set at the end of every epoch.

    Logs to W&B:
      val/coco/map, map_50, map_75, map_small/medium/large, ar_100,
      val/coco/ap_rare, ap_common, ap_frequent,
      val/coco/eval_secs.

    Stores per-class AP and metric history for the end-of-run callback.
    """

    def __init__(self, val_ds, image_processor, id2label: Dict[int, str],
                 batch_size: int = 8, score_thresh: float = 0.05, num_workers: int = 4):
        from pycocotools.cocoeval import COCOeval  # imported lazily so module imports stay light
        self.COCOeval = COCOeval
        self.val_ds = val_ds
        self.image_processor = image_processor
        self.id2label = id2label
        self.batch_size = batch_size
        self.score_thresh = score_thresh
        self.num_workers = num_workers
        self.coco_gt = val_ds.coco
        self._freq_slices = _build_freq_slices(self.coco_gt)
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def _collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"][0] for b in batch],
        }

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if wandb.run is None:
            return
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
            pin_memory=True,
        )

        all_results: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for batch in loader:
            pix = batch["pixel_values"].to(device, non_blocking=True)
            outputs = model(pixel_values=pix)
            tgt_sz = torch.stack([_orig_size_from_label(lab).to(device) for lab in batch["labels"]])
            res = self.image_processor.post_process_object_detection(
                outputs, threshold=self.score_thresh, target_sizes=tgt_sz)
            for i, dets in enumerate(res):
                img_id = int(batch["labels"][i]["image_id"].item())
                boxes = dets["boxes"].cpu().numpy()
                scores = dets["scores"].cpu().numpy()
                labels = dets["labels"].cpu().numpy()
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b.tolist()
                    all_results.append({
                        "image_id": img_id, "category_id": int(l),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s),
                    })
        elapsed = time.perf_counter() - t0
        logger.info(f"  COCO eval inference at epoch {state.epoch:.1f}: "
                    f"{elapsed:.0f}s, {len(all_results)} predictions")

        if not all_results:
            logger.warning(f"  no predictions above threshold {self.score_thresh}; skipping mAP")
            if was_training:
                model.train()
            return

        coco_dt = self.coco_gt.loadRes(all_results)
        ce = self.COCOeval(self.coco_gt, coco_dt, "bbox")
        ce.evaluate(); ce.accumulate(); ce.summarize()

        s = ce.stats
        metrics = {
            "val/coco/map":       float(s[0]),
            "val/coco/map_50":    float(s[1]),
            "val/coco/map_75":    float(s[2]),
            "val/coco/map_small": float(s[3]),
            "val/coco/map_medium":float(s[4]),
            "val/coco/map_large": float(s[5]),
            "val/coco/ar_100":    float(s[8]),
            "val/coco/eval_secs": float(elapsed),
        }

        # Per-class AP (averaged over IoU thresholds, all areas, max-100 dets)
        per_class_ap: Dict[int, float] = {}
        if hasattr(ce, "eval") and "precision" in ce.eval:
            prec = ce.eval["precision"]  # [T, R, K, A, M]
            cat_ids_sorted = sorted(self.coco_gt.cats.keys())
            for k_idx, cat_id in enumerate(cat_ids_sorted):
                p = prec[:, :, k_idx, 0, -1]
                p = p[p > -1]
                per_class_ap[cat_id] = float(p.mean()) if p.size else float("nan")
            for slice_name, cats in self._freq_slices.items():
                vals = [per_class_ap[c] for c in cats
                        if c in per_class_ap and not np.isnan(per_class_ap[c])]
                if vals:
                    metrics[f"val/coco/ap_{slice_name}"] = float(np.mean(vals))

        wandb.log(metrics, step=state.global_step)
        self.history.append({
            "epoch": float(state.epoch),
            "global_step": int(state.global_step),
            "metrics": metrics,
            "per_class_ap": per_class_ap,
        })

        if was_training:
            model.train()


# --------- 3. backbone unfreeze ----------
class BackboneUnfreezeCallback(TrainerCallback):
    """At a configured fraction of training, unfreeze the last N DINOv3 blocks at lr*mult."""

    def __init__(self, frac_of_total: float = 0.8, num_blocks: int = 2, lr_multiplier: float = 0.1):
        self.frac = frac_of_total
        self.num_blocks = num_blocks
        self.lr_multiplier = lr_multiplier
        self._unfrozen = False

    def on_epoch_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self._unfrozen:
            return
        if state.epoch < self.frac * args.num_train_epochs:
            return

        body = self._body(model)
        if body is None:
            logger.warning("BackboneUnfreezeCallback: backbone body not found; skipping")
            self._unfrozen = True
            return
        blocks = self._blocks(body)
        if not blocks:
            logger.warning("BackboneUnfreezeCallback: transformer blocks not found; skipping")
            self._unfrozen = True
            return

        new_params = []
        for blk in blocks[-self.num_blocks:]:
            for p in blk.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    new_params.append(p)

        if new_params and optimizer is not None:
            backbone_lr = args.learning_rate * self.lr_multiplier
            optimizer.add_param_group({
                "params": new_params,
                "lr": backbone_lr,
                "weight_decay": args.weight_decay,
                "initial_lr": backbone_lr,
            })
            logger.info(f"BackboneUnfreezeCallback: unfroze last {self.num_blocks} blocks "
                        f"({len(new_params)} tensors, lr={backbone_lr:.2e}) at epoch {state.epoch:.1f}")
            if wandb.run is not None:
                wandb.log({
                    "backbone/unfrozen_blocks": self.num_blocks,
                    "backbone/unfrozen_param_tensors": len(new_params),
                    "backbone/unfreeze_epoch": float(state.epoch),
                    "backbone/lr": backbone_lr,
                }, step=state.global_step)
        self._unfrozen = True

    @staticmethod
    def _body(model):
        try:
            inner = model.base if hasattr(model, "base") else model
            return inner.model.backbone.body
        except AttributeError:
            return None

    @staticmethod
    def _blocks(body):
        for path in ("encoder.layer", "encoder.layers", "blocks", "layers"):
            obj = body
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        return None


# --------- 4. end-of-run artifacts ----------
class EndOfRunArtifactsCallback(TrainerCallback):
    """
    On train end:
      - log final model directory as a W&B artifact (versioned)
      - log mAP-history as a W&B Table
      - log top/bottom 50 per-class AP and rare/common/frequent slice means
      - log top-K most-confused class pairs (computed from a final inference pass
        when periodic_eval has run at least once)
    """

    def __init__(self, model_dir: str, periodic_eval: Optional[PeriodicCOCOEvalCallback] = None,
                 confusion_top_k: int = 30, confusion_iou_thresh: float = 0.5):
        self.model_dir = model_dir
        self.periodic_eval = periodic_eval
        self.confusion_top_k = confusion_top_k
        self.confusion_iou_thresh = confusion_iou_thresh

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if wandb.run is None:
            return

        # 1. Model artifact
        save_dir = Path(self.model_dir) / "final"
        try:
            if save_dir.exists():
                art = wandb.Artifact(
                    name=f"raptor-{wandb.run.id}-model",
                    type="model",
                    description=f"Final RAPTOR weights from run {wandb.run.name}",
                )
                art.add_dir(str(save_dir))
                wandb.run.log_artifact(art)
                logger.info(f"Logged W&B model artifact: {art.name}")
        except Exception as e:
            logger.warning(f"Model artifact upload failed: {e}")

        # 2. mAP history table + per-class AP charts
        if self.periodic_eval and self.periodic_eval.history:
            self._log_map_history()
            self._log_per_class_ap()
            self._log_confusion_pairs(model)

    def _log_map_history(self):
        tbl = wandb.Table(columns=["epoch", "map", "map_50", "map_75",
                                   "ap_rare", "ap_common", "ap_frequent"])
        for h in self.periodic_eval.history:
            m = h["metrics"]
            tbl.add_data(
                h["epoch"],
                m.get("val/coco/map", 0.0),
                m.get("val/coco/map_50", 0.0),
                m.get("val/coco/map_75", 0.0),
                m.get("val/coco/ap_rare", 0.0),
                m.get("val/coco/ap_common", 0.0),
                m.get("val/coco/ap_frequent", 0.0),
            )
        wandb.log({"final/map_history": tbl})

    def _log_per_class_ap(self):
        last = self.periodic_eval.history[-1]
        per_class = last["per_class_ap"]
        id2label = self.periodic_eval.id2label
        rows = [(cid, id2label.get(cid, str(cid)), ap)
                for cid, ap in per_class.items() if not np.isnan(ap)]
        rows.sort(key=lambda r: -r[2])
        top = rows[:50]
        bot = rows[-50:][::-1]

        tbl_top = wandb.Table(columns=["class", "AP"])
        for _, name, ap in top:
            tbl_top.add_data(name, ap)
        wandb.log({"final/per_class_ap_top50":
                   wandb.plot.bar(tbl_top, "class", "AP", title="Top 50 classes by AP")})

        tbl_bot = wandb.Table(columns=["class", "AP"])
        for _, name, ap in bot:
            tbl_bot.add_data(name, ap)
        wandb.log({"final/per_class_ap_bottom50":
                   wandb.plot.bar(tbl_bot, "class", "AP", title="Bottom 50 classes by AP")})

        slices = self.periodic_eval._freq_slices
        slice_means = {}
        for name, cats in slices.items():
            vals = [per_class[c] for c in cats
                    if c in per_class and not np.isnan(per_class[c])]
            slice_means[name] = float(np.mean(vals)) if vals else 0.0
        tbl_slice = wandb.Table(columns=["slice", "mean_AP", "n_classes"])
        for name, cats in slices.items():
            tbl_slice.add_data(name, slice_means[name], len(cats))
        wandb.log({"final/ap_by_slice":
                   wandb.plot.bar(tbl_slice, "slice", "mean_AP",
                                  title="AP by frequency slice (rare/common/frequent)")})

    @torch.no_grad()
    def _log_confusion_pairs(self, model):
        """
        Top-K most confusable class pairs:
        For predictions with score > 0.3 that overlap a GT box (IoU > thresh) of a
        DIFFERENT class, count (gt_class, pred_class). Top-K pairs by count.

        Done as a single inference pass over the val set since coco_eval
        doesn't expose this directly.
        """
        ev = self.periodic_eval
        if ev is None:
            return
        device = next(model.parameters()).device
        model.eval()
        loader = DataLoader(ev.val_ds, batch_size=ev.batch_size, shuffle=False,
                            num_workers=ev.num_workers, collate_fn=ev._collate, pin_memory=True)
        confusion: Dict[tuple, int] = defaultdict(int)
        score_thresh = 0.3
        n_pairs = 0
        for batch in loader:
            pix = batch["pixel_values"].to(device, non_blocking=True)
            outputs = model(pixel_values=pix)
            tgt_sz = torch.stack([_orig_size_from_label(lab).to(device) for lab in batch["labels"]])
            res = ev.image_processor.post_process_object_detection(
                outputs, threshold=score_thresh, target_sizes=tgt_sz)
            for i, dets in enumerate(res):
                lab = batch["labels"][i]
                gt_classes = lab.get("class_labels")
                gt_boxes = lab.get("boxes")  # cxcywh normalized; convert to xyxy in resized space
                if gt_classes is None or gt_classes.numel() == 0:
                    continue
                # Convert GT boxes from cxcywh-normalized (target_size) to xyxy in orig image coords.
                # We don't have the resized HxW here, so use orig_size and assume processor preserved aspect.
                orig_h, orig_w = _orig_size_from_label(lab).tolist()
                cx, cy, w, h = gt_boxes.unbind(-1)
                gt_xyxy = torch.stack([
                    (cx - w / 2) * orig_w, (cy - h / 2) * orig_h,
                    (cx + w / 2) * orig_w, (cy + h / 2) * orig_h,
                ], dim=-1)
                pred_xyxy = dets["boxes"].cpu()
                pred_cls = dets["labels"].cpu()
                # IoU [N_pred, N_gt]
                if pred_xyxy.numel() == 0:
                    continue
                ious = self._pairwise_iou_xyxy(pred_xyxy, gt_xyxy)
                # For each pred, find best-matched GT
                best_iou, best_gt = ious.max(dim=1)
                for p_idx, (iou, g_idx) in enumerate(zip(best_iou.tolist(), best_gt.tolist())):
                    if iou < self.confusion_iou_thresh:
                        continue
                    pc = int(pred_cls[p_idx].item())
                    gc = int(gt_classes[g_idx].item())
                    if pc != gc:
                        confusion[(gc, pc)] += 1
                        n_pairs += 1
        if not confusion:
            logger.info("Confusion pairs: no off-diagonal matches above thresh; skipping table")
            return
        top = sorted(confusion.items(), key=lambda kv: -kv[1])[:self.confusion_top_k]
        id2label = ev.id2label
        tbl = wandb.Table(columns=["gt_class", "pred_class", "count"])
        for (gc, pc), n in top:
            tbl.add_data(id2label.get(gc, str(gc)), id2label.get(pc, str(pc)), n)
        wandb.log({"final/top_confused_pairs": tbl})
        logger.info(f"Logged {len(top)} top-confused class pairs (total off-diagonal matches: {n_pairs})")

    @staticmethod
    def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        union = a1[:, None] + a2[None, :] - inter
        return inter / union.clamp(min=1e-6)
