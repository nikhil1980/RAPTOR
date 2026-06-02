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
    5. EMAWeightsCallback          - maintain a shadow exponential-moving-average
                                     of model weights; swap them in for eval and
                                     for checkpoint save so load_best_model_at_end
                                     selects on EMA metrics.

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


def _post_process_padded(outputs, image_processor, threshold: float,
                         orig_sizes_hw, device) -> List[Dict[str, torch.Tensor]]:
    """Post-process detections accounting for the dataset's S×S bottom-right pad.

    `CocoDetDataset` pads every letterboxed image to a fixed S×S square and
    rescales GT boxes so they live in [0,1] of the padded frame. The model
    therefore emits boxes in [0,1] of that same padded frame — NOT of the
    original image. Calling `post_process_object_detection` with `target_sizes
    = orig (h, w)` scales both axes by the original dims, squashing predictions
    along the shorter axis by `min(h,w)/max(h,w)` and tanking mAP for every
    non-square image.

    Fix: pass `target_sizes = (max_dim, max_dim)` per image so post_process
    scales both axes by `max(h, w)`, recovering true original-pixel coords
    inside the valid letterboxed region. Then clip to the image's real bounds
    and drop boxes that collapse to zero area (they fell in the pad).
    """
    max_per = [int(max(int(s[0]), int(s[1]))) for s in orig_sizes_hw]
    max_dims = torch.tensor([[m, m] for m in max_per], dtype=torch.long, device=device)
    res = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=max_dims,
    )
    cleaned: List[Dict[str, torch.Tensor]] = []
    for i, dets in enumerate(res):
        oh = int(orig_sizes_hw[i][0])
        ow = int(orig_sizes_hw[i][1])
        b = dets["boxes"]
        if b.numel() == 0:
            cleaned.append(dets)
            continue
        x1 = b[:, 0].clamp(0, ow)
        y1 = b[:, 1].clamp(0, oh)
        x2 = b[:, 2].clamp(0, ow)
        y2 = b[:, 3].clamp(0, oh)
        keep = (x2 > x1) & (y2 > y1)
        cleaned.append({
            "boxes":  torch.stack([x1, y1, x2, y2], dim=1)[keep],
            "scores": dets["scores"][keep],
            "labels": dets["labels"][keep],
        })
    return cleaned


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
                try:
                    if p is None:
                        raise FileNotFoundError(iminfo["file_name"])
                    self._cached_originals.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    logger.warning(
                        f"SamplePredictionsCallback: skipping unreadable val image "
                        f"idx={i} ({iminfo['file_name']}): {e}"
                    )
                    self._cached_originals.append(Image.new("RGB", (640, 640), color=(0, 0, 0)))
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
            res = _post_process_padded(
                outputs, self.image_processor, self.score_thresh,
                [torch.tensor([orig.height, orig.width])], device,
            )[0]
            annotated = _draw_boxes_pil(
                orig.copy(),
                res["boxes"].cpu().numpy(),
                res["scores"].cpu().numpy(),
                res["labels"].cpu().numpy(),
                self.id2label,
            )
            wb_imgs.append(wandb.Image(annotated, caption=f"val[{idx}] epoch={state.epoch:.1f}"))

        # Step omitted: HF's WandbCallback already committed at state.global_step
        # when training-loss was logged for this step; passing step=N here would
        # trigger "Tried to log to step N < current N+1" and drop the images.
        wandb.log({"val/sample_predictions": wb_imgs})
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
        # COCO mAP only scores the top-100 detections per image (maxDets[-1]=100).
        # Capping here keeps results identical while bounding `all_results` memory:
        # at a low score_thresh on an uncalibrated model RT-DETR passes ~all 300
        # queries/image, so an uncapped 61k-image val set yields ~18M dicts and
        # OOM-kills the process inside `loadRes`/`COCOeval` (May 31 incident).
        self.max_dets_per_image = 100
        self._freq_slices = _build_freq_slices(self.coco_gt)
        self.history: List[Dict[str, Any]] = []
        # Guard against double-runs if on_evaluate fires twice for the same epoch
        # (e.g. eval_strategy changes or end-of-training extra evaluate).
        self._last_epoch_run: int = -1

    @staticmethod
    def _collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"][0] for b in batch],
        }

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Runs inside Trainer.evaluate() so we can inject COCO keys into the
        # Trainer's metrics dict and unblock metric_for_best_model lookup.
        if wandb.run is None:
            return
        # Avoid recomputing the 30+ min COCO eval if evaluate() is invoked twice
        # within the same epoch (HF can do this at end-of-training).
        cur_epoch = int(state.epoch) if state.epoch is not None else -1
        if cur_epoch == self._last_epoch_run:
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
            orig_sizes = [_orig_size_from_label(lab) for lab in batch["labels"]]
            res = _post_process_padded(
                outputs, self.image_processor, self.score_thresh,
                orig_sizes, device,
            )
            for i, dets in enumerate(res):
                img_id = int(batch["labels"][i]["image_id"].item())
                # Keep only the top-K by score per image: COCO mAP scores at most
                # maxDets=100 per image, so this is metric-neutral but caps the
                # size of `all_results` (and the loadRes/COCOeval that follows).
                scores_t = dets["scores"]
                if scores_t.numel() > self.max_dets_per_image:
                    topk = torch.topk(scores_t, self.max_dets_per_image).indices
                    boxes = dets["boxes"][topk].cpu().numpy()
                    scores = scores_t[topk].cpu().numpy()
                    labels = dets["labels"][topk].cpu().numpy()
                else:
                    boxes = dets["boxes"].cpu().numpy()
                    scores = scores_t.cpu().numpy()
                    labels = dets["labels"].cpu().numpy()
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b.tolist()
                    # Map dense model index -> original COCO category_id for COCO eval.
                    orig_cid = self.val_ds.idx_to_cat_id.get(int(l), int(l))
                    all_results.append({
                        "image_id": img_id, "category_id": orig_cid,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s),
                    })
        elapsed = time.perf_counter() - t0
        logger.info(f"  COCO eval inference at epoch {state.epoch:.1f}: "
                    f"{elapsed:.0f}s, {len(all_results)} predictions")

        if not all_results:
            logger.warning(f"  no predictions above threshold {self.score_thresh}; "
                           f"reporting zeroed mAP so the run survives")
            zero_metrics = {
                "val/coco/map": 0.0, "val/coco/map_50": 0.0, "val/coco/map_75": 0.0,
                "val/coco/map_small": 0.0, "val/coco/map_medium": 0.0, "val/coco/map_large": 0.0,
                "val/coco/ar_100": 0.0, "val/coco/eval_secs": float(elapsed),
            }
            # Do NOT pass step=state.global_step: HF's WandbCallback has already
            # committed at that step inside evaluate(), so wandb has advanced its
            # internal pointer and rejects the second write as monotonicity-violating
            # ("step N < current N+1"). Logging without an explicit step lands at
            # the current (post-eval) step — one tick later than eval_loss in the
            # UI, but the data is not silently dropped.
            wandb.log(zero_metrics)
            if metrics is not None:
                for k, v in zero_metrics.items():
                    metrics[f"eval_{k}"] = float(v)
            if was_training:
                model.train()
            return

        coco_dt = self.coco_gt.loadRes(all_results)
        ce = self.COCOeval(self.coco_gt, coco_dt, "bbox")
        ce.evaluate(); ce.accumulate(); ce.summarize()

        s = ce.stats
        flat_metrics = {
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
                    flat_metrics[f"val/coco/ap_{slice_name}"] = float(np.mean(vals))

        # Log flat names to W&B so existing dashboards keep working.
        # Step is intentionally omitted — see comment on the zero-metrics path
        # above. HF's WandbCallback already logged at state.global_step inside
        # evaluate(); reusing that step here drops the data with the warning
        # "Tried to log to step N that is less than the current step N+1".
        wandb.log(flat_metrics)

        # Inject HF-prefixed keys into the Trainer's metrics dict so
        # metric_for_best_model="eval_val/coco/map" can find them in
        # _determine_best_metric (which runs right after on_evaluate).
        if metrics is not None:
            for k, v in flat_metrics.items():
                metrics[f"eval_{k}"] = float(v)

        self.history.append({
            "epoch": float(state.epoch),
            "global_step": int(state.global_step),
            "metrics": flat_metrics,
            "per_class_ap": per_class_ap,
        })
        self._last_epoch_run = cur_epoch

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

        backbone_lr = args.learning_rate * self.lr_multiplier
        n = self.apply_unfreeze(model, optimizer, backbone_lr, args.weight_decay)
        if n:
            logger.info(f"BackboneUnfreezeCallback: unfroze last {self.num_blocks} blocks "
                        f"({n} tensors, lr={backbone_lr:.2e}) at epoch {state.epoch:.1f}")
            if wandb.run is not None:
                wandb.log({
                    "backbone/unfrozen_blocks": self.num_blocks,
                    "backbone/unfrozen_param_tensors": n,
                    "backbone/unfreeze_epoch": float(state.epoch),
                    "backbone/lr": backbone_lr,
                }, step=state.global_step)
        self._unfrozen = True

    def apply_unfreeze(self, model, optimizer, backbone_lr, weight_decay):
        """Unfreeze the last ``num_blocks`` DINOv3 blocks and append a dedicated
        low-LR optimizer param group. Returns the count of newly-trainable
        tensors (0 if the blocks can't be located — fail-open).

        Shared by the scheduled ``on_epoch_begin`` path and the resume-time
        reconciliation in ``LongTailTrainer.create_optimizer``: a checkpoint
        saved after the unfreeze fired carries an extra optimizer param group,
        so the freshly-built optimizer on resume must reproduce it or HF's
        ``_load_optimizer_and_scheduler`` raises a param-group-count ValueError.
        The block/param iteration order here must stay identical to the original
        run — ``optimizer.load_state_dict`` matches saved state to params by
        insertion order, not by name."""
        body = self._body(model)
        if body is None:
            logger.warning("BackboneUnfreezeCallback: backbone body not found; skipping")
            return 0
        blocks = self._blocks(body)
        if not blocks:
            logger.warning("BackboneUnfreezeCallback: transformer blocks not found; skipping")
            return 0

        new_params = []
        for blk in blocks[-self.num_blocks:]:
            for p in blk.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    new_params.append(p)

        if new_params and optimizer is not None:
            optimizer.add_param_group({
                "params": new_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "initial_lr": backbone_lr,
            })
        return len(new_params)

    @staticmethod
    def _body(model):
        try:
            inner = model.base if hasattr(model, "base") else model
            return inner.model.backbone.body
        except AttributeError:
            return None

    @staticmethod
    def _blocks(body):
        for path in ("encoder.layer", "encoder.layers", "model.layer", "model.layers", "blocks", "layers"):
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
        id2label = self.periodic_eval.id2label  # dense 0-indexed
        val_ds = self.periodic_eval.val_ds
        # per_class is keyed by ORIGINAL COCO category IDs (from coco_eval).
        # Look up names via dense index: orig_cid -> dense_idx -> name.
        def _name_for(orig_cid):
            dense = val_ds.cat_id_to_idx.get(orig_cid)
            if dense is None:
                return str(orig_cid)
            return id2label.get(dense, str(orig_cid))
        rows = [(cid, _name_for(cid), ap)
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
            orig_sizes = [_orig_size_from_label(lab) for lab in batch["labels"]]
            res = _post_process_padded(
                outputs, ev.image_processor, score_thresh, orig_sizes, device,
            )
            for i, dets in enumerate(res):
                lab = batch["labels"][i]
                gt_classes = lab.get("class_labels")
                gt_boxes = lab.get("boxes")  # cxcywh normalized in the padded S×S frame
                if gt_classes is None or gt_classes.numel() == 0:
                    continue
                # GT boxes live in [0,1] of the padded S×S frame (see CocoDetDataset
                # bottom-right pad + rescale). Match the prediction post-process by
                # scaling both axes by max(orig_h, orig_w) so predictions and GT live
                # in the same true-original-pixel space; IoU is then meaningful.
                orig_h, orig_w = (int(v) for v in _orig_size_from_label(lab).tolist())
                max_dim = float(max(orig_h, orig_w))
                cx, cy, w, h = gt_boxes.unbind(-1)
                gt_xyxy = torch.stack([
                    (cx - w / 2) * max_dim, (cy - h / 2) * max_dim,
                    (cx + w / 2) * max_dim, (cy + h / 2) * max_dim,
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
        # Both GT and predicted classes are dense 0-indexed here (from class_labels
        # and post_process_object_detection respectively); id2label is dense too.
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


# --------- 5. EMA weights ----------
class EMAWeightsCallback(TrainerCallback):
    """Exponential-moving-average of model weights, swapped in for eval + save.

    Standard SOTA-detector trick: keep a shadow copy of each parameter that decays
    toward the live weights as
        ema_p <- decay * ema_p + (1 - decay) * live_p
    after every optimizer step. Swap the live and EMA weights around evaluation
    (so reported metrics + COCO mAP reflect EMA) and around checkpoint save (so
    `load_best_model_at_end` picks an EMA snapshot).

    Implementation notes:
      - Shadow lives on CPU to save GPU VRAM (~1× model size CPU cost).
      - Tracks ALL params, not just trainable ones. Frozen-param EMA is a no-op
        since live values never change; this also makes the BackboneUnfreeze
        transition automatic — when those params start moving, their EMA buffer
        is initialized to the current (pretrained) value and begins decaying.
      - Buffers (BN stats etc.) are intentionally NOT shadowed; they update via
        their own running-average and forcing them through EMA can drift them.
    """

    def __init__(self, decay: float = 0.9998, warmup_steps: int = 10_000):
        """
        :param decay: long-horizon decay factor (typical 0.9998).
        :param warmup_steps: ramp the effective decay from ~0 up to `decay` over
            this many optimizer steps. Without warmup, the shadow stays anchored
            to the initial random init for thousands of steps (decay^t × init
            term dominates), so evals run on a model that is mostly randomly
            initialized and AP collapses to ~0. Standard FixMatch/MAE pattern:
                d(t) = min(decay, (1+t) / (warmup_steps + 1 + t)).
            At t=0, d≈0 so EMA = live. At t=warmup_steps, d ≈ 0.5. At t≫warmup
            (and t≫1/(1-decay)), d → decay.
        """
        self.decay = decay
        self.warmup_steps = max(1, int(warmup_steps))
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._swapped_in: bool = False
        self._initialized: bool = False

    def _current_decay(self, step: int) -> float:
        # Monotonically increasing toward self.decay; clipped above by it so the
        # asymptote never overshoots the configured long-horizon value.
        ramp = (1.0 + step) / (self.warmup_steps + 1.0 + step)
        return min(self.decay, ramp)

    def _model(self, kwargs) -> Optional[torch.nn.Module]:
        m = kwargs.get("model")
        return m

    def _init_shadow(self, model: torch.nn.Module) -> None:
        self._shadow = {n: p.detach().clone().to("cpu") for n, p in model.named_parameters()}
        self._initialized = True
        logger.info(f"EMAWeightsCallback: initialized shadow ({len(self._shadow)} tensors, "
                    f"decay={self.decay}, warmup_steps={self.warmup_steps})")

    @torch.no_grad()
    def on_train_begin(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        if model is not None and not self._initialized:
            self._init_shadow(model)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        if model is None or not self._initialized or self._swapped_in:
            return
        # Warmup-aware decay: tracks live closely while the model is still moving
        # fast; transitions to the long-horizon decay as training stabilizes.
        d = self._current_decay(int(state.global_step))
        one_minus_d = 1.0 - d
        for n, p in model.named_parameters():
            shadow = self._shadow.get(n)
            if shadow is None:
                # New trainable param (e.g. after BackboneUnfreezeCallback fires).
                self._shadow[n] = p.detach().clone().to("cpu")
                continue
            if shadow.shape != p.shape:
                # Defensive: re-sync if shape changed (shouldn't happen mid-training).
                self._shadow[n] = p.detach().clone().to("cpu")
                continue
            shadow.mul_(d).add_(p.detach().to(shadow.device, dtype=shadow.dtype),
                                alpha=one_minus_d)

    @torch.no_grad()
    def _swap_in(self, model: torch.nn.Module) -> None:
        if self._swapped_in or not self._initialized:
            return
        for n, p in model.named_parameters():
            shadow = self._shadow.get(n)
            if shadow is None or shadow.shape != p.shape:
                continue
            self._backup[n] = p.detach().clone()
            p.data.copy_(shadow.to(p.device, dtype=p.dtype))
        self._swapped_in = True

    @torch.no_grad()
    def _swap_out(self, model: torch.nn.Module) -> None:
        if not self._swapped_in:
            return
        for n, p in model.named_parameters():
            backup = self._backup.get(n)
            if backup is None:
                continue
            p.data.copy_(backup)
        self._backup.clear()
        self._swapped_in = False

    # Eval + save lifecycle. HF Trainer._maybe_log_save_evaluate order is:
    #   on_epoch_end -> _evaluate (which fires on_evaluate) -> _determine_best_metric
    #     -> _save_checkpoint -> on_save
    # We want EMA weights active for: eval_loss computation, the COCO mAP pass,
    # best-metric selection, AND the checkpoint file write. Then restore LIVE
    # before the next training step.
    #   on_epoch_end : swap IN  -> eval/COCO/save all run with EMA weights
    #   on_save      : swap OUT -> next training step optimizes LIVE again
    @torch.no_grad()
    def on_epoch_end(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        if model is not None:
            self._swap_in(model)

    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        # Intentionally a no-op: keep EMA weights swapped in through the save
        # that follows on_evaluate in _maybe_log_save_evaluate, so the saved
        # checkpoint contains EMA weights (which is what _determine_best_metric
        # selected on). Restoration happens in on_save below.
        return

    @torch.no_grad()
    def on_save(self, args, state, control, **kwargs):
        # Fires AFTER _save_checkpoint writes the file. Since on_evaluate no
        # longer swaps out, the file was written with EMA weights. Restore LIVE
        # so the next training step optimizes the real weights. Safe if EMA was
        # never swapped in (mid-epoch save with save_strategy=steps): _swap_out
        # early-returns when self._swapped_in is False.
        model = self._model(kwargs)
        if model is not None:
            self._swap_out(model)

    @torch.no_grad()
    def on_train_end(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        if model is None or not self._initialized:
            return
        # Dump EMA weights to <model_dir>/final-ema/pytorch_model.bin
        try:
            out_dir = Path(args.output_dir) / "final-ema"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Build a CPU state_dict of EMA params + the model's buffers (BN, etc.).
            # Saves all named_buffers — non-persistent ones (e.g. attention masks)
            # are tiny and harmless to round-trip.
            ema_state = dict(self._shadow)
            buffers = {n: b.detach().to("cpu") for n, b in model.named_buffers()}
            ema_state.update(buffers)
            torch.save(ema_state, out_dir / "pytorch_model.bin")
            logger.info(f"EMAWeightsCallback: dumped EMA snapshot to {out_dir}")
        except Exception as e:
            logger.warning(f"EMAWeightsCallback: final-ema dump failed: {e}")
