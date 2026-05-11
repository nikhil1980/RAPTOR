import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mitigate CUDA memory fragmentation on long runs. Must be set before any
# `import torch` path that touches CUDA, hence top-of-file.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import json, math
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch, torch.nn as nn
# Switch torch's shared-memory IPC from FD-passing to /dev/shm-backed files.
# The default `file_descriptor` strategy consumes 2 FDs per shared tensor sent
# across worker -> main, which exhausts the per-process ulimit when num_workers
# is high and pin_memory is on (EMFILE / "unable to open shared memory object").
# `file_system` is unbounded by FD limits at the cost of relying on /dev/shm
# space — fine for our setup since batches are small and freed promptly.
# Must be set before any DataLoader is constructed.
torch.multiprocessing.set_sharing_strategy("file_system")

from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrConfig,
    Trainer,
    TrainingArguments
)
from transformers.utils import logging
from transformers import AutoTokenizer, SiglipTextModel
import wandb
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config, LossConfig
""" User Modules """

"""
Training script for DINOv3 + RTDETR on custom MoD (mixture of data).

@author: Nikhil Bhargava
@date: 2025-10-15
@license: Apache-2.0
@description: This script fine-tunes the RTDETR model with a DINOv3 backbone on a MoD.
@copyright: Copyright 2025 Nikhil Bhargava

To run:

# Open a tmux session, run the process and exit the session (CTRL + B then D)
tmux new -s  raptor   

export WANDB_PROJECT=RAPTOR
export RAPTOR_WANDB_PROJECT="raptor-dinov3-rtdetr"
export RAPTOR_WANDB_RUN_NAME="50ep-bs48-bf16-a6000"
export RAPTOR_TRAIN_BATCH_SIZE=48 
export RAPTOR_TRAIN_VAL_BATCH_SIZE=32 
export RAPTOR_TRAIN_ACCUM_STEPS=1 
export RAPTOR_TRAIN_EPOCHS=2
export RAPTOR_ACCELERATE_NUM_PROCESSES=12
export RAPTOR_EVAL_BATCH_SIZE=32                    
                                                                                                                        
# OPTIONAL: scale LR up slightly to compensate for 3× larger batch.
# Standard sqrt-scaling: 2e-4 × sqrt(3) ≈ 3.5e-4. Or keep conservative at 2e-4.

export RAPTOR_TRAIN_LEARNING_RATE=3e-4      
                                                                                                         
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train/train_dinov3_rtdetr_ov.py --config-file config.json > logs/train.log 2>&1

tmux attach -t raptor 
"""

logger = get_logger(__name__)

logging.set_verbosity_info()


# Get the parent path
BASE_PATH = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# --------- Dataset ----------
class CocoDetDataset(Dataset):
    """
    COCO Detection Dataset
    """
    def __init__(self,
                 ann_json: Path,
                 img_roots: List[Path],
                 processor: RTDetrImageProcessor,
                 id2label: Dict[int,str]):
        """
        Initialize the COCO Detection Dataset

        :param ann_json: Path to COCO annotations JSON.
        :param img_roots: List of directories containing images.
        :param processor: RTDetrImageProcessor for preprocessing images and annotations.
        :param id2label: Mapping from label IDs to label names.

        :return: None
        """
        self.coco = COCO(str(ann_json))
        self.img_ids = self.coco.getImgIds()
        self.img_roots = img_roots
        self.processor = processor
        self.id2label = id2label
        all_images = self.coco.loadImgs(self.img_ids)
        # Pre-filter images whose files cannot be resolved across img_roots so
        # the DataLoader never throws FileNotFoundError mid-epoch (which kills
        # the worker and tears down the run). Surface the count + a few names
        # at startup instead.
        kept, missing = [], []
        for info in all_images:
            if self._resolve_path(info["file_name"]) is not None:
                kept.append(info)
            else:
                missing.append(info["file_name"])
        if missing:
            sample = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            logger.warning(
                f"CocoDetDataset: {len(missing)}/{len(all_images)} images not found "
                f"under img_roots={[str(r) for r in self.img_roots]}; skipping. "
                f"Examples: {sample}{more}"
            )
        self.images = kept
        self.img_ids = [info["id"] for info in kept]
        self._missing_at_runtime: set = set()
        # Dense 0-indexed remap: COCO often uses 1-indexed (or sparse) category IDs,
        # but the model's classifier outputs num_classes logits indexed [0, num_classes).
        # Build both directions; __getitem__ uses cat_id_to_idx, COCO eval uses idx_to_cat_id.
        self.cat_ids_sorted = sorted(self.coco.cats.keys())
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(self.cat_ids_sorted)}
        self.idx_to_cat_id = {idx: cid for idx, cid in enumerate(self.cat_ids_sorted)}

    def _resolve_path(self, file_name: str) -> Optional[Path]:
        """
        Resolve the full path of an image file by searching through the provided image root directories.

        Returns the first matching Path, or None if the file is not found in any
        configured img_root. Callers must handle the None case (we used to return
        a bare Path(file_name) which silently bubbled up to a worker-killing
        FileNotFoundError deep in the DataLoader).

        :param file_name: Name of the image file.

        :return: Full path to the image file, or None if not found.
        :rtype: Optional[Path]
        """
        for root in self.img_roots:
            p = Path(root) / file_name
            if p.exists():
                return p
        return None

    def __len__(self): return len(self.images)

    def __getitem__(self, idx)-> Dict[str, Any]:
        """
        Get a processed item from the dataset.

        1. Retrieves image information and loads the image.
        2. Loads annotations for the image and constructs the target dictionary.
        3. Processes the image and annotations using the RTDetrImageProcessor.
        4. Squeezes unnecessary dimensions from the processed tensors.
        5. Returns a dictionary with keys "pixel_values" and "labels".
        6. The "pixel_values" tensor is of shape (3, H, W) and "labels" is a list of dictionaries.
        7. Each label dictionary contains keys such as "boxes", "labels", "area", and "iscrowd".
        8. The method ensures that the data is in the correct format for training or evaluation.
        9. Handles images in RGB format.
        10. Uses the COCO API to manage annotations and image metadata.
        11. The processor resizes images to the specified size in the configuration.
        12. The method is designed to work with datasets formatted in the COCO style.
        13. The method assumes that the processor is properly configured for object detection tasks.
        14. The method can be used in conjunction with a DataLoader for batching.
        15. The method is compatible with PyTorch tensors and can be used in training loops.
        16. The method is efficient and leverages the capabilities of the RTDetrImageProcessor.
        17. The method is robust to missing or incomplete annotations, defaulting to empty
        annotations if none are found.

        :param idx: Index of the item to retrieve.
        :return: Dictionary containing processed image and labels.
        :rtype: Dict[str, Any]

        """
        iminfo = self.images[idx]
        img_path = self._resolve_path(iminfo["file_name"])
        try:
            if img_path is None:
                raise FileNotFoundError(iminfo["file_name"])
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as e:
            # Defensive: pre-filter at __init__ removes resolvable-at-startup
            # gaps, but a file may still vanish or be corrupt at read time.
            # Log once per filename and fall through to the next sample so the
            # DataLoader worker keeps running instead of tearing down training.
            fname = iminfo["file_name"]
            if fname not in self._missing_at_runtime:
                self._missing_at_runtime.add(fname)
                logger.warning(f"CocoDetDataset: skipping unreadable image idx={idx} ({fname}): {e}")
            return self.__getitem__((idx + 1) % len(self))

        ann_ids = self.coco.getAnnIds(imgIds=iminfo["id"])
        anns = self.coco.loadAnns(ann_ids)
        objects = []
        for a in anns:
            x, y, w, h = a["bbox"]
            # Skip degenerate boxes (zero or negative size — they crash GIoU).
            if w <= 0 or h <= 0:
                continue
            raw_cid = int(a["category_id"])
            # Skip annotations whose category isn't in the categories list, and
            # remap the rest to dense 0-indexed so they're valid model class indices.
            dense_idx = self.cat_id_to_idx.get(raw_cid)
            if dense_idx is None:
                continue
            objects.append({
                "category_id": dense_idx,
                "bbox": [x, y, w, h],
                "area": a.get("area", w * h),
                "iscrowd": a.get("iscrowd", 0),
            })
        target = {"image_id": int(iminfo["id"]), "annotations": objects}

        processed = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
            size=Config.IMAGE_SIZE,
        )
        processed["pixel_values"] = processed["pixel_values"].squeeze(0)
        # Do NOT squeeze label tensors. The processor returns boxes as (N, 4),
        # class_labels as (N,), area as (N,), etc. — already without a batch dim.
        # Squeezing collapsed (1, 4) -> (4,) for single-annotation images and
        # broke downstream `boxes[:, 0]` indexing.
        processed["labels"] = [processed["labels"][0]]

        # Letterbox-style resize keeps aspect ratio but yields variable (H, W).
        # Pad bottom-right to a fixed square so collate can stack the batch,
        # and rescale normalized cxcywh boxes from (H, W) basis to (S, S) basis.
        target_size = Config.IMAGE_SIZE.get(
            "longest_edge",
            Config.IMAGE_SIZE.get("height", 640),
        )
        pv = processed["pixel_values"]  # (3, H, W)
        H, W = pv.shape[-2], pv.shape[-1]
        if H != target_size or W != target_size:
            padded = torch.zeros(pv.shape[0], target_size, target_size, dtype=pv.dtype)
            padded[:, :H, :W] = pv
            processed["pixel_values"] = padded
            lab = dict(processed["labels"][0])
            if "boxes" in lab and lab["boxes"].numel() > 0:
                sx, sy = W / target_size, H / target_size
                boxes = lab["boxes"].clone()
                boxes[:, 0] *= sx  # cx
                boxes[:, 1] *= sy  # cy
                boxes[:, 2] *= sx  # w
                boxes[:, 3] *= sy  # h
                lab["boxes"] = boxes
            processed["labels"] = [lab]
        return processed


def build_idmaps(coco_json: Path) -> Tuple[Dict[int,str], Dict[str,int]]:
    """
    Build ID to label and label to ID mappings from COCO annotations JSON.

    1. Loads the COCO annotations JSON file.
    2. Extracts the categories from the JSON data.
    3. Constructs a dictionary mapping category IDs to category names (id2label).
    4. Constructs a reverse dictionary mapping category names to category IDs (label2id).
    5. Returns both dictionaries.

    :param coco_json: Path to COCO annotations JSON.

    :return: Tuple of two dictionaries: (id2label, label2id).
    :rtype: (Dict[int,str], Dict[str,int])
    """
    js = json.load(open(coco_json))
    # Sort by raw COCO id, then assign dense 0-indexed positions. Must use the
    # same ordering as CocoDetDataset so the model's id2label aligns with the
    # class_labels the dataset produces.
    cats = sorted(js["categories"], key=lambda c: c["id"])
    id2label = {idx: c["name"] for idx, c in enumerate(cats)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


# --------- DINOv3 ViT -> multi-scale (ViTDet-style FPN) ----------
class DinoV3FPNBackbone(nn.Module):
    """
    ViTDet-style multi-scale adapter for DINOv3 ViT.

    DINOv3 ViT outputs a single stride-16 feature map. RT-DETR's HybridEncoder
    needs three feature maps at strides 8 / 16 / 32. We synthesize them from
    the last block's patch tokens with:
        - stride 8:  ConvTranspose2d 2x  (upsample)
        - stride 16: 1x1 Conv            (channel projection)
        - stride 32: stride-2 3x3 Conv   (downsample)

    Exposes the interface RTDetrModel expects from its `backbone`:
        - forward(pixel_values, pixel_mask) -> List[(features, mask)]
        - .intermediate_channel_sizes attribute
    """
    def __init__(self, backbone_name: str, out_channels: List[int], freeze: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.body = AutoModel.from_pretrained(backbone_name)
        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)
            self.body.eval()
        cfg = self.body.config
        self.patch_size = cfg.patch_size
        c_in = cfg.hidden_size
        c8, c16, c32 = out_channels

        self.lat8 = nn.Sequential(
            nn.ConvTranspose2d(c_in, c8, kernel_size=2, stride=2),
            nn.GroupNorm(32, c8),
        )
        self.lat16 = nn.Sequential(
            nn.Conv2d(c_in, c16, kernel_size=1),
            nn.GroupNorm(32, c16),
        )
        self.lat32 = nn.Sequential(
            nn.Conv2d(c_in, c32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, c32),
        )
        self.intermediate_channel_sizes = list(out_channels)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor = None):
        B, _, H, W = pixel_values.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        N = Hp * Wp
        out = self.body(pixel_values=pixel_values)
        last = out.last_hidden_state                    # [B, special+N, C]
        patches = last[:, -N:, :]                       # patch tokens are last
        feat = patches.transpose(1, 2).reshape(B, -1, Hp, Wp)

        f8 = self.lat8(feat)
        f16 = self.lat16(feat)
        f32 = self.lat32(feat)

        if pixel_mask is None:
            pixel_mask = torch.ones((B, H, W), device=pixel_values.device, dtype=torch.bool)
        feats_and_masks = []
        for f in (f8, f16, f32):
            m = nn.functional.interpolate(
                pixel_mask[None].float(), size=f.shape[-2:], mode="nearest"
            ).to(torch.bool)[0]
            feats_and_masks.append((f, m))
        return feats_and_masks


# --------- OV head + focal BCE ----------
class OVHead(nn.Module):
    """
    Open-Vocabulary Head for RTDETR that projects model features to text embedding space.
    It uses focal BCE loss for training to solve the long-tail problem.
    """
    def __init__(self, hidden_dim: int, text_dim: int = 768):
        """
        Initialize the OVHead.

        1. Initializes a linear projection layer to map from hidden_dim to text_dim.
        2. Initializes a learnable logit scale parameter for scaling the logits.
        3. The projection layer does not use a bias term.
        4. The logit scale is initialized to log(1/0.07).
        5. The OVHead is designed to be used in conjunction with a text encoder that
        produces embeddings of dimension text_dim.
        6. The OVHead can be integrated into a larger model for open-vocabulary object

        :param hidden_dim: Hidden dimension of the RTDETR model.
        :param text_dim: Dimension of the text embeddings (default: 768).

        :return: None
        """
        super().__init__()
        self.proj = nn.Linear(hidden_dim, text_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))

    def forward(self,
                box_embeds: torch.Tensor,
                text_embeds: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass to compute similarity logits between box embeddings and text embeddings.

        1. Projects the box embeddings to the text embedding space using the linear projection layer.
        2. Normalizes both the projected box embeddings and the text embeddings to unit length.
        3. Computes the similarity logits using the dot product between the normalized box embeddings and text embeddings.
        4. Scales the logits by the exponential of the logit scale parameter.
        5. Returns the similarity logits (shape: [batch_size, num_queries, num_classes]).
        6. The method is designed to handle batches of box embeddings and text embeddings.
        7. The method uses torch.einsum for efficient computation of the dot product.
        8. The method is designed to be used in the context of open-vocabulary object detection.
        9. The method assumes that the input embeddings are in the correct shape and format.
        10. The method can be used during both training and inference.
        11. The method is compatible with PyTorch's autograd for backpropagation.
        12. The method can handle batches of data for efficient processing.
        13. The method is robust to different batch sizes and number of queries.
        14. The method is optimized for performance using normalization and scaling techniques.

        :param box_embeds: Box embeddings from the RTDETR model (shape: [batch_size, num_queries, hidden_dim]).
        :param text_embeds: Text embeddings (shape: [num_classes, text_dim]).

        :return: Similarity logits (shape: [batch_size, num_queries, num_classes]).
        :rtype: torch.Tensor
        """
        be = nn.functional.normalize(self.proj(box_embeds), dim=-1)
        te = nn.functional.normalize(text_embeds, dim=-1)
        return torch.einsum("bqc,tc->bqt", be, te) * self.logit_scale.exp()


def binary_focal_with_logits(logits: torch.Tensor,
                             targets: torch.Tensor,
                             alpha: float = 0.25,
                             gamma: float = 2.0,
                             reduction: str = "mean"
                             ) -> torch.Tensor:
    """
    Compute binary focal loss with logits.

    1. Computes the binary cross-entropy loss with logits without reduction.
    2. Applies the sigmoid function to the logits to obtain probabilities.
    3. Computes the modulating factor based on the predicted probabilities and targets.
    4. Computes the alpha factor based on the targets.
    5. Combines the cross-entropy loss, modulating factor, and alpha factor to compute the focal loss.
    6. Applies the specified reduction method (mean, sum, or none) to the loss.
    7. Returns the computed focal loss.
    8. The function is designed to handle binary classification tasks.

    :param logits: Predicted  logits. Shape: (batch_size, num_classes)
    :param targets: Binary ground truth labels. Shape: (batch_size, num_classes)
    :param alpha: Weighting factor for the rare class (default: 0.25).
    :param gamma: Weighting factor for hard examples (default: 2.0).
    :param reduction: Reduction method to apply to the output: "mean", "sum", or "none" (default: "mean").

    :return: Focal loss value.
    :rtype: torch.Tensor
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p*targets + (1 - p)*(1 - targets)
    alpha_t = alpha*targets + (1 - alpha)*(1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss


def build_text_embeddings(class_names: List[str], device) -> torch.Tensor:
    """
    Build text embeddings for class names using a pretrained text encoder.

    1. Loads a pretrained text encoder and tokenizer.
    2. For each class name, constructs a prompt "a photo of a {class_name}".
    3. Tokenizes the prompt and passes it through the text encoder to obtain embeddings.
    4. Collects the embeddings for all class names into a single tensor.
    5. Returns the tensor of text embeddings (shape: [num_classes, embedding_dim]).
    6. The method uses the specified device (e.g., GPU) for computation.
    7. The method is designed to work with a specific text encoder model.
    8. The method handles batching of text inputs for efficiency.
    9. The method ensures that the text embeddings are in the correct format for use in the OV head.

    :param class_names: List of class names.
    :param device: Device to load the model onto (e.g., "cuda" or "cpu").

    :return: Tensor of text embeddings (shape: [num_classes, embedding_dim]).
    :rtype: torch.Tensor
    """
    tok = AutoTokenizer.from_pretrained(Config.TEXT_ENCODER)
    txtm = SiglipTextModel.from_pretrained(Config.TEXT_ENCODER).to(device).eval()
    prompts = [f"a photo of a {name}" for name in class_names]
    with torch.no_grad():
        inputs = tok(prompts, return_tensors="pt", padding="max_length", truncation=True).to(device)
        out = txtm(**inputs)
        # SigLIP pools the last (EOS-equivalent) token; fall back to it if the
        # checkpoint exposes no pooler_output.
        embs = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, -1]
    embs = embs.detach().clone()
    del txtm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embs


# --------- Self-contained Hungarian matcher ----------
# Avoids depending on transformers.models.rt_detr internals — the matcher
# class has moved across HF releases (modeling_rt_detr -> loss_rt_detr -> ...).
def _box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def _pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Both inputs in xyxy. Returns GIoU matrix [N, M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    lt_e = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_e = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc = (rb_e - lt_e).clamp(min=0).prod(-1)
    return iou - (enc - union) / enc.clamp(min=1e-6)


class RAPTORHungarianMatcher(nn.Module):
    """
    Hungarian matcher for RT-DETR style outputs (focal classification cost
    + L1 box cost + GIoU box cost). Boxes are normalized cxcywh.
    """
    def __init__(self,
                 class_cost: float = 2.0,
                 bbox_cost: float = 5.0,
                 giou_cost: float = 2.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        bs, nq = outputs["logits"].shape[:2]
        device = outputs["pred_boxes"].device
        # Build the full [B*Q, sumNt] cost matrix on CPU. linear_sum_assignment
        # runs on CPU anyway; keeping the transient class/bbox/giou tensors
        # off-GPU avoids hundreds-of-MB spikes that previously OOM'd long runs
        # whose reserved cache had fragmented.
        out_prob = outputs["logits"].flatten(0, 1).sigmoid().float().cpu()    # [B*Q, K]
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float().cpu()           # [B*Q, 4]

        sizes = [int(t["class_labels"].numel()) for t in targets]
        if sum(sizes) == 0:
            empty = (torch.empty(0, dtype=torch.long, device=device),
                     torch.empty(0, dtype=torch.long, device=device))
            return [empty for _ in range(bs)]

        tgt_ids = torch.cat([t["class_labels"] for t in targets]).long().cpu()
        tgt_bbox = torch.cat([t["boxes"] for t in targets]).float().cpu()

        eps = 1e-8
        neg = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + eps).log())
        pos = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + eps).log())
        class_cost = pos[:, tgt_ids] - neg[:, tgt_ids]              # [B*Q, sumNt]

        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1)             # [B*Q, sumNt]
        giou_cost = -_pairwise_giou(_box_cxcywh_to_xyxy(out_bbox),
                                    _box_cxcywh_to_xyxy(tgt_bbox))   # [B*Q, sumNt]

        C = (self.class_cost * class_cost
             + self.bbox_cost * bbox_cost
             + self.giou_cost * giou_cost)
        C = C.view(bs, nq, -1)

        indices = []
        offset = 0
        for b, n_tgt in enumerate(sizes):
            if n_tgt == 0:
                indices.append((torch.empty(0, dtype=torch.long, device=device),
                                torch.empty(0, dtype=torch.long, device=device)))
                continue
            cost_b = C[b, :, offset:offset + n_tgt].numpy()
            q_idx, t_idx = linear_sum_assignment(cost_b)
            indices.append((torch.as_tensor(q_idx, dtype=torch.long, device=device),
                            torch.as_tensor(t_idx, dtype=torch.long, device=device)))
            offset += n_tgt
        return indices


#--------- Model wrapper with OV head and loss ----------
class ModelWithOV(nn.Module):
    """
    Model wrapper to add OV head and loss to RTDETR.
    """
    def __init__(self,
                 base: RTDetrForObjectDetection,
                 ov_head: OVHead,
                 text_embeds: torch.Tensor
                 ):
        """
        Initialize the ModelWithOV.

        :param base: Base RTDETR model.
        :param ov_head: OV head for open-vocabulary detection.
        :param text_embeds: Text embeddings for class names.

        :return: None
        """
        super().__init__()
        self.base = base
        self.ov_head = ov_head
        self.register_buffer("text_embeds", text_embeds, persistent=False)
        # Self-contained Hungarian matcher (HF moved RTDetrHungarianMatcher across
        # releases; we don't depend on its location). Cost weights mirror RT-DETR
        # defaults, falling back if the config doesn't expose them.
        cfg = base.config
        # Expose config so HF Trainer's W&B integration can log model metadata
        # (it does `model.config.to_dict()` to serialize architecture into the run).
        self.config = cfg
        self.matcher = RAPTORHungarianMatcher(
            class_cost=getattr(cfg, "matcher_class_cost", 2.0),
            bbox_cost=getattr(cfg, "matcher_bbox_cost", 5.0),
            giou_cost=getattr(cfg, "matcher_giou_cost", 2.0),
            alpha=getattr(cfg, "matcher_alpha", 0.25),
            gamma=getattr(cfg, "matcher_gamma", 2.0),
        )

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        HuggingFace's WandbCallback calls model.num_parameters() to log model size.
        nn.Module doesn't define this — only PreTrainedModel does — so we provide
        the same signature here.
        """
        params = self.parameters()
        if only_trainable:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, **batch):
        """
        Forward pass with OV head and loss.

        OV target construction (replaces prior self-distillation):
            1. Run RT-DETR Hungarian matcher on (logits, pred_boxes) vs GT.
            2. For each matched query, write a 1.0 at the GT class index.
            3. Unmatched queries stay all-zero — focal BCE handles imbalance,
               so we do NOT mask them out (RT-DETR has no background class
               under focal loss).

        :param batch: Input batch containing pixel values and labels.
        :return: Model outputs with added OV loss if in training mode.
        """
        if self.training and self.ov_head is not None and self.text_embeds is not None:
            outputs = self.base(**batch, output_hidden_states=True)
            dec = getattr(outputs, "decoder_hidden_states", None)
            labels = batch.get("labels", None)
            if dec is not None and len(dec) > 0 and labels is not None:
                dec = dec[-1]                                       # [B,Q,C]
                ov_logits = self.ov_head(dec, self.text_embeds)     # [B,Q,T]
                B, Q, T = ov_logits.shape

                with torch.no_grad():
                    indices = self.matcher(
                        {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes},
                        labels,
                    )

                target = torch.zeros((B, Q, T), device=ov_logits.device, dtype=ov_logits.dtype)
                for b_idx, (q_idx, t_idx) in enumerate(indices):
                    if q_idx.numel() == 0:
                        continue
                    gt_classes = labels[b_idx]["class_labels"][t_idx].long().clamp_max(T - 1)
                    target[b_idx, q_idx.long(), gt_classes] = 1.0

                ov_loss = binary_focal_with_logits(
                    ov_logits, target,
                    alpha=LossConfig.ov_alpha,
                    gamma=LossConfig.ov_gamma,
                    reduction="mean",
                )
                outputs.loss = outputs.loss + LossConfig.ov_weight * ov_loss
        else:
            outputs = self.base(**batch)
        return outputs


# --------- Build processor / datasets ----------
def build_processor_and_datasets():
    """
    Build the image processor and datasets for training and validation.

    :return: None
    """
    id2label, label2id = build_idmaps(Path(os.path.join(BASE_PATH, Config.TRAIN_JSON)))
    image_processor = RTDetrImageProcessor.from_pretrained(Config.RTDETR_IMAGE_PROCESSOR)
    image_processor.size = Config.IMAGE_SIZE

    # Make absolute paths for img_dirs
    img_dirs = list()
    for  dir in Config.TRAIN_IMG_DIRS:
        path = os.path.join(BASE_PATH, dir)
        img_dirs.append(path)

    train_ds = CocoDetDataset(Path(os.path.join(BASE_PATH, Config.TRAIN_JSON)),
                              img_dirs, #0th entry is train and 1st entry is val
                              image_processor,
                              id2label)

    # Make absolute paths for img_dirs
    img_dirs = list()
    for  dir in Config.VAL_IMG_DIRS:
        path = os.path.join(BASE_PATH, dir)
        img_dirs.append(path)

    val_ds   = CocoDetDataset(Path(os.path.join(BASE_PATH, Config.VAL_JSON)),
                              img_dirs,
                              image_processor,
                              id2label)

    return image_processor, train_ds, val_ds, id2label, label2id


# --------- Class-aware sampling (image weights) ----------
def compute_image_weights_from_json(ann_json: Path, beta: float = 1.0)-> Dict[int, float]:
    """
    Compute image weights for class-aware sampling from COCO annotations JSON.

    :param ann_json: Path to COCO annotations JSON.
    :param beta: Weighting factor for class frequency adjustment
    (default: 1.0 for inverse frequency).

    beta=1 -> inverse frequency;
    beta=0.5 -> inverse sqrt; beta in [0.5,1] is common.

    :return: Dictionary mapping class ID to its image weights.
    :rtype: Dict[int, float]
    """
    js = json.load(open(ann_json))
    counts = defaultdict(int)
    img2cats = defaultdict(set)
    for a in js["annotations"]:
        counts[a["category_id"]] += 1
        img2cats[a["image_id"]].add(a["category_id"])
    # per-class weight
    class_w = {k: (1.0 / (v + 1e-6))**beta for k, v in counts.items()}
    # per-image weight = mean of class weights present in image
    img_w = {}
    for im in js["images"]:
        imcats = img2cats.get(im["id"], set())
        if not imcats:
            img_w[im["id"]] = 1.0
        else:
            img_w[im["id"]] = sum(class_w[c] for c in imcats) / max(1, len(imcats))
    return img_w


def build_weight_vector_for_dataset(dataset: CocoDetDataset, img_w: Dict[int, float]) -> torch.Tensor:
    """
    Assign weights to each image in the dataset based on precomputed image weights.

    1. Iterates through each image in the dataset.
    2. Retrieves the weight for each image using its ID from the provided img_w dictionary
    3. If an image ID is not found in img_w, a default weight of 1.0 is assigned.
    4. Collects all weights into a list and converts it to a PyTorch tensor
    5. Returns the tensor of weights.

    :param dataset: CocoDetDataset instance.
    :param img_w: Dictionary mapping image IDs to their weights.

    :return: Tensor of weights corresponding to each image in the dataset.
    :rtype: torch.Tensor
    """
    # Align weights in dataset order
    w = []
    for iminfo in dataset.images:
        w.append(float(img_w.get(iminfo["id"], 1.0)))
    return torch.tensor(w, dtype=torch.double)



# --------- Model ----------
def build_model(num_labels: int,
                id2label: Dict[int,str],
                label2id: Dict[str,int],
                device="cuda"
                )-> nn.Module:
    """
    Build the RTDETR model with DINOv3 backbone and optional OV head.

    :param num_labels: Number of labels/classes.
    :param id2label: Mapping from label IDs to label names.
    :param label2id: Mapping from label names to label IDs.
    :param device: Device to load the model onto (default: "cuda").

    :return: Configured RTDETR model (with OV head if specified).
    :rtype: nn.Module

    1. Configures the RTDETR model with specified parameters including focal loss and VFL settings.
    2. Initializes the model with a DINOv3 backbone and freezes the backbone if specified.
    3. If USE_OV_HEAD is True, builds text embeddings and an OV head, wrapping the model accordingly.
    4. Returns the final model ready for training or evaluation.
    5. If USE_OV_HEAD is False, returns the RTDETR model without the OV head.
    6. The model is moved to the specified device (e.g., GPU if available).
    7. The function uses global configuration parameters defined in the Config class.
    8. The OV head is configured with specific loss parameters for open-vocabulary detection.
    """
    # Build RT-DETR with a placeholder ResNet backbone purely to size the
    # encoder's input projections. We then swap in our DINOv3 + FPN adapter,
    # matching the channel sizes the placeholder reported.
    cfg = RTDetrConfig(
        num_labels=num_labels,
        id2label={i: id2label[i] for i in id2label},
        label2id=label2id,
        use_pretrained_backbone=False,                   # weights come from DINOv3 below
        backbone="resnet50",                             # placeholder for channel sizing
        backbone_kwargs={"out_indices": (1, 2, 3)},
        freeze_backbone_batch_norms=True,

        # ---- long-tail / focal knobs ----
        use_focal_loss=True,
        focal_loss_alpha=Config.FOREGROUND_ALPHA,
        focal_loss_gamma=Config.FOCAL_GAMMA,
        weight_loss_vfl=Config.VFL_WEIGHT,               # 1.0 keeps VFL on; 0.0 for pure focal
        eos_coefficient=Config.EOS_COEF,                 # downweight "no object"
    )
    model = RTDetrForObjectDetection(cfg)

    # Replace the placeholder backbone with DINOv3 ViT + ViTDet-style FPN.
    fpn_channels = list(model.model.backbone.intermediate_channel_sizes)
    model.model.backbone = DinoV3FPNBackbone(
        Config.PRETRAINED_BACKBONE,
        fpn_channels,
        freeze=Config.FREEZE_BACKBONE,
    )

    # Manual activation checkpointing on RT-DETR decoder layers.
    # RTDetrForObjectDetection has `_supports_gradient_checkpointing = False`,
    # so HF's `gradient_checkpointing_enable()` raises ValueError. The decoder's
    # cross-attention (300 queries × multi-scale encoder tokens × 6 layers) is
    # the dominant activation cost; wrapping each layer's forward in
    # torch.utils.checkpoint recomputes those activations on backward instead
    # of storing them — ~25-35% memory at ~15% step-time cost.
    if os.getenv("RAPTOR_TRAIN_GRADIENT_CHECKPOINTING", "true").lower() in ("true", "1", "yes"):
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _wrap_with_checkpoint(layer: nn.Module) -> None:
            _orig = layer.forward
            def _checkpointed_forward(*args, **kwargs):
                return _ckpt(_orig, *args, use_reentrant=False, **kwargs)
            layer.forward = _checkpointed_forward

        n_wrapped = 0
        decoder = getattr(getattr(model, "model", None), "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            for _layer in decoder.layers:
                _wrap_with_checkpoint(_layer)
                n_wrapped += 1
        logger.info(f"---> Gradient checkpointing wrapped {n_wrapped} decoder layers <---")

    if Config.USE_OV_HEAD:
        text_names = [id2label[i] for i in sorted(id2label)]
        text_embeds = build_text_embeddings(text_names, device=device)
        ov_head = OVHead(hidden_dim=model.config.d_model, text_dim=text_embeds.shape[-1])
        wrap = ModelWithOV(model, ov_head, text_embeds)
        return wrap
    return model


# --------- Collator ----------
def collate_fn(batch)-> Dict[str, Any]:
    """
    Collate function to combine a list of samples into a batch.

    :param batch: List of samples, where each sample is a dictionary with keys "pixel_values" and "labels".
    1. Stacks the "pixel_values" tensors from each sample into a single tensor.
    2. Collects the "labels" from each sample into a list.
    3. Returns a dictionary with the batched "pixel_values" and the list of

    :return: A dictionary with keys "pixel_values" and "labels".
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"][0] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# --------- Trainer with WeightedRandomSampler ----------
class LongTailTrainer(Trainer):
    """
    Trainer with WeightedRandomSampler
    """
    def __init__(self,
                 image_processor: RTDetrImageProcessor,
                 train_ds: Dataset,
                 *args,
                 **kwargs):
        """
        Custom Trainer to handle long-tail distributions using WeightedRandomSampler.

        1. Initializes the base Trainer class with provided arguments.
        2. Computes sample weights based on the training dataset to address class imbalance.
        3. Sets up a WeightedRandomSampler for the training dataloader to ensure balanced sampling.
        4. Overrides the get_train_dataloader method to use the custom sampler.
        5. Overrides the compute_loss method to handle the model's output format.

        :param image_processor: Instance of RTDetrImageProcessor for preprocessing images.
        :param train_ds: Training dataset.
        :param args: Additional positional arguments for the Trainer.
        :param kwargs: Additional keyword arguments for the Trainer.

        :return: None
        """
        super().__init__(*args, **kwargs)
        self.image_processor = image_processor
        self.train_ds = train_ds
        # compute weights once
        img_w = compute_image_weights_from_json(Path(os.path.join(BASE_PATH, str(Config.TRAIN_JSON))),
                                                beta=0.8)
        self.sample_weights = build_weight_vector_for_dataset(train_ds, img_w)

    def get_train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.train_ds),  # one epoch in expectation
            replacement=True
        )
        # Dataloader workers are intentionally decoupled from DDP world-size:
        # using RAPTOR_ACCELERATE_NUM_PROCESSES here multiplies fanout (world_size
        # ranks each spawning that many workers) and triggers EMFILE under
        # pin_memory. Use a dedicated knob with a conservative default.
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=int(os.getenv("RAPTOR_DATALOADER_NUM_WORKERS",
                                      str(Config.NUM_WORKERS))),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs)-> torch.Tensor:
        """
        Compute loss for the model.

        1. Calls the model with the provided inputs to get the outputs.
        2. Extracts the loss from the model's outputs, handling both attribute and dictionary formats.
        3. Returns the loss, and optionally the outputs if return_outputs is True.
            - If return_outputs is True, returns a tuple of (loss, outputs).
            - If return_outputs is False, returns only the loss.

        The method is designed to be flexible with different model output formats.

        :param model: Model to compute loss for.
        :param inputs: Inputs to the model.
        :param return_outputs: Whether to return the model outputs along with the loss.

        :return: Computed loss, and optionally the model outputs.
        :rtype: torch.Tensor or Tuple[torch.Tensor, Any]
        """
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss



def train():
    """
    Train the RTDETR model with DINOv3 backbone on a mixture of datasets.

    :return: None
    """
    from train.wandb_callbacks import (
        SamplePredictionsCallback,
        PeriodicCOCOEvalCallback,
        BackboneUnfreezeCallback,
        EndOfRunArtifactsCallback,
    )

    wandb_enabled = os.getenv("RAPTOR_WANDB_PROJECT_ENABLED", "true").lower() in ("true", "1", "yes")
    if wandb_enabled:
        wandb.init(
            project=os.getenv("RAPTOR_WANDB_PROJECT", "dinov3_lvis_oi_rtdetr"),
            name=os.getenv("RAPTOR_WANDB_RUN_NAME", "focal_vfl_classaware"),
            config={
                "backbone":               Config.PRETRAINED_BACKBONE,
                "text_encoder":           Config.TEXT_ENCODER,
                "image_size":             Config.IMAGE_SIZE,
                "freeze_backbone":        Config.FREEZE_BACKBONE,
                "use_ov_head":            Config.USE_OV_HEAD,
                "focal_alpha":            Config.FOREGROUND_ALPHA,
                "focal_gamma":            Config.FOCAL_GAMMA,
                "vfl_weight":             Config.VFL_WEIGHT,
                "eos_coef":               Config.EOS_COEF,
                "ov_alpha":               LossConfig.ov_alpha,
                "ov_gamma":               LossConfig.ov_gamma,
                "ov_weight":              LossConfig.ov_weight,
            },
            tags=["dinov3", "rt-detr", "ov-head", "long-tail"],
        )

    image_processor, train_ds, val_ds, id2label, label2id = build_processor_and_datasets()
    model = build_model(len(id2label),
                        id2label,
                        {v: k for k, v in id2label.items()},
                        device=Config.DEVICE
                        )
    model_dir = os.path.join(BASE_PATH, os.getenv("RAPTOR_PATHS_MODEL_DIR", "runs/dinov3_rtdetr"))

    # Prefer bf16 over fp16 — RT-DETR's focal/VFL is known to NaN in fp16.
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_supported and os.getenv("RAPTOR_TRAIN_BF16", "true").lower() in ("true", "1", "yes")
    use_fp16 = (not use_bf16) and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=int(os.getenv("RAPTOR_TRAIN_BATCH_SIZE", "2")),
        per_device_eval_batch_size=int(os.getenv("RAPTOR_TRAIN_VAL_BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.getenv("RAPTOR_TRAIN_ACCUM_STEPS", "4")),
        learning_rate=float(os.getenv("RAPTOR_TRAIN_LEARNING_RATE", "2e-4")),
        num_train_epochs=int(os.getenv("RAPTOR_TRAIN_EPOCHS", "50")),
        lr_scheduler_type=os.getenv("RAPTOR_TRAIN_LR_SCHEDULER_TYPE", "cosine"),
        warmup_ratio=float(os.getenv("RAPTOR_TRAIN_WARMUP_RATIO", "0.05")),
        weight_decay=float(os.getenv("RAPTOR_TRAIN_WEIGHT_DECAY", "0.05")),
        logging_steps=int(os.getenv("RAPTOR_TRAIN_LOGGING_STEPS", "25")),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(os.getenv("RAPTOR_TRAIN_SAVE_TOTAL_LIMIT", "3")),
        load_best_model_at_end=True,
        metric_for_best_model=os.getenv("RAPTOR_TRAIN_METRIC_FOR_BEST", "eval_loss"),
        greater_is_better=os.getenv("RAPTOR_TRAIN_GREATER_IS_BETTER", "false").lower() == "true",
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=float(os.getenv("RAPTOR_TRAIN_MAX_GRAD_NORM", "1.0")),
        dataloader_num_workers=int(os.getenv("RAPTOR_DATALOADER_NUM_WORKERS",
                                             str(Config.NUM_WORKERS))),
        # ModelWithOV.forward uses **batch (kwargs-only), so HF's find_labels()
        # cannot auto-detect "labels" from the signature and label_names defaults
        # to []. That makes has_labels=False in prediction_step, the loss branch
        # is skipped, and eval_loss is never emitted -> metric_for_best lookup
        # fails. Set explicitly so HF knows which inputs key carries the targets.
        label_names=["labels"],
        report_to=["wandb"] if wandb_enabled else [],
        run_name=os.getenv("RAPTOR_WANDB_RUN_NAME", "focal_vfl_classaware"),
    )
    logger.info(f"---> precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'} <---")

    # Build callbacks (only attach W&B-dependent ones if wandb is on).
    callbacks = []
    if wandb_enabled:
        periodic_eval = PeriodicCOCOEvalCallback(
            val_ds=val_ds,
            image_processor=image_processor,
            id2label=id2label,
            batch_size=int(os.getenv("RAPTOR_EVAL_BATCH_SIZE", "8")),
            score_thresh=float(os.getenv("RAPTOR_EVAL_SCORE_THRESH", "0.05")),
            num_workers=int(os.getenv("RAPTOR_DATALOADER_NUM_WORKERS",
                                      str(Config.NUM_WORKERS))),
        )
        sample_preds = SamplePredictionsCallback(
            val_ds=val_ds,
            image_processor=image_processor,
            id2label=id2label,
            every_n_epochs=int(os.getenv("RAPTOR_WANDB_SAMPLE_PREDS_EVERY", "2")),
            num_samples=int(os.getenv("RAPTOR_WANDB_NUM_SAMPLE_PREDS", "8")),
        )
        end_artifacts = EndOfRunArtifactsCallback(
            model_dir=model_dir,
            periodic_eval=periodic_eval,
        )
        callbacks.extend([periodic_eval, sample_preds, end_artifacts])

    # Backbone unfreeze is independent of W&B — always include if Config.FREEZE_BACKBONE.
    if Config.FREEZE_BACKBONE and float(os.getenv("RAPTOR_UNFREEZE_BACKBONE_FRAC", "0.8")) < 1.0:
        callbacks.append(BackboneUnfreezeCallback(
            frac_of_total=float(os.getenv("RAPTOR_UNFREEZE_BACKBONE_FRAC", "0.8")),
            num_blocks=int(os.getenv("RAPTOR_UNFREEZE_BACKBONE_BLOCKS", "2")),
            lr_multiplier=float(os.getenv("RAPTOR_UNFREEZE_BACKBONE_LR_MULT", "0.1")),
        ))

    trainer = LongTailTrainer(
        image_processor=image_processor,
        train_ds=train_ds,
        model=model,
        args=args,
        data_collator=collate_fn,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )

    # Auto-resume from the latest checkpoint in `model_dir` if one exists.
    # Set RAPTOR_TRAIN_AUTO_RESUME=false to force a fresh run.
    auto_resume = os.getenv("RAPTOR_TRAIN_AUTO_RESUME", "true").lower() in ("true", "1", "yes")
    ckpts = glob.glob(os.path.join(model_dir, "checkpoint-*")) if auto_resume else []
    resume = True if ckpts else None
    if resume:
        logger.info(f"---> RESUMING FROM LATEST CHECKPOINT IN {model_dir} <---")
    trainer.train(resume_from_checkpoint=resume)
    save_dir = os.path.join(model_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    if wandb_enabled:
        wandb.finish()

def main():
    """
    Main function to run the workflow.

    :return:
    """
    parser = build_myargparser()
    args = parser.parse_args()

    # 1. Load configuration from JSON file if provided
    if args.config_file:
        # Get the parent path
        par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct the full path to the config file
        config_file_path = os.path.join(par_path, str(args.config_file))

        # Check if the file exists before attempting to open
        if os.path.exists(config_file_path):
            try:
                # 1. Load the environment variables
                load_env_from_json(config_file_path)
                Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT")
                # Env overrides for annotation JSONs let sanity-check runs point at
                # a subsampled mixture without editing common/config.py. Class attrs
                # are evaluated at import time (before load_env_from_json), so we
                # apply the override here, mirroring the Config.ROOT pattern above.
                Config.TRAIN_JSON = os.getenv("RAPTOR_PATHS_TRAIN_JSON", Config.TRAIN_JSON)
                Config.VAL_JSON   = os.getenv("RAPTOR_PATHS_VAL_JSON",   Config.VAL_JSON)
                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")
                logger.debug(f"---> TRAIN_JSON={Config.TRAIN_JSON}  VAL_JSON={Config.VAL_JSON} <---")

                # 2. Start Training
                start_time = time.perf_counter()
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_TRAIN")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")
                train()
                end_time = time.perf_counter()
                logger.debug(f"---> DINOv3 WITH RTDETR OV HEAD TRAINED IN TIME {(end_time - start_time) / 60 * 60: .3f} HOURS <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")

            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_TRAIN")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

if __name__ == '__main__':
    main()
