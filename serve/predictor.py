# serve/predictor.py
# Unified closed + open vocabulary object detection for production


import os, json, math, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """

"""
Script to run unified closed + open vocabulary object detection using a pre-trained

@author: Nikhil Bhargava
@date: 2025-10-18
@license: Apache-2.0
@description: This script performs unified closed + open vocabulary object detection using a pre-trained RTDETR model. It first detects objects using the closed-set head and then refines and expands the detections using an open-vocabulary approach with CLIP.
 - Closed-set: RT-DETR head w/ DINOv3 backbone (on trained weights)
 - Open-vocab: region-wise CLIP/SigLIP text matching on detector's proposals
 - Returns union of unique labels above thresholds (with boxes/scores if requested)

@copyright: Copyright 2025 Nikhil Bhargava
"""

# Optional: OpenCLIP for robust image<->text
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


logger = get_logger(__name__)

# --------------------------------
# Helpers: boxes, IoU, NMS, utils
# --------------------------------
def xyxy_to_xywh(box: List[float])-> List[float]:
    """
    Convert [x1,y1,x2,y2] to [x,y,w,h] format on each box

    :param box: Bounding box in xyxy format
    :return: Bounding box in xywh format
    :rtype: List[float]
    """
    x1,y1,x2,y2 = box
    return [x1, y1, x2-x1, y2-y1]

def wh_to_xyxy(box: List[float])-> List[float]:
    """
    Convert [x,y,w,h] to [x1,y1,x2,y2] format on each box

    :param box: Bounding box in xywh format
    :return: Bounding box in xyxy format
    :rtype: List[float]
    """
    x,y,w,h = box
    return [x, y, x+w, y+h]

def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Vanilla IoU between two boxes

    :param a: Bonding box a
    :type a: np.ndarray
    :param b: Bounding box b
    :type b: np.ndarray

    :return: IoU value
    :rtype: float
    """
    # a,b: [4] in xyxy
    x_a= max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])

    # 1. Find intersection
    inter = max(0, x_b-x_a) * max(0, y_b-y_a)

    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])

    # 2. Find union
    union = area_a + area_b - inter + 1e-6
    return inter/union

def nms(dets: List[Dict], iou_thresh=0.5) -> List[Dict]:
    """
    Non-maximum suppression on detections by label

    :param dets: Detections list to process
    :type dets: List[Dict]
    :param iou_thresh: IoU threshold for suppression
    :type iou_thresh: float

    :return: Filtered detections after NMS
    :rtype: List[Dict]
    """
    out = []
    by_label: Dict[str, List[Dict]] = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)

    for lab, group in by_label.items():
        group = sorted(group, key=lambda x: x["score"], reverse=True)
        keep = []
        for g in group:
            if all(box_iou(np.array(g["box"]), np.array(k["box"])) < iou_thresh for k in keep):
                keep.append(g)
        out.extend(keep)
    return out

def load_lexicon(path: Optional[str]) -> List[str]:
    """
    Load lexicon from file or return fallback

    :param path: Path to lexicon file
    :type path: Optional[str]
    :return: List of labels
    :rtype: List[str]
    """
    if path and Path(path).exists():
        with open(path, "r") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        return labels
    return Config.FALLBACK_LEXICON

# --------------------------------
# OpenCLIP wrapper
# --------------------------------
class ClipHelper:
    def __init__(self,
                 device: str,
                 model_name: str,
                 pretrained: str
                 ):
        """
        Initialize CLIP model and tokenizer.

        :param device: Device to run the model on
        :param model_name: Model name for CLIP
        :param pretrained: Pretrained weights to use
        :type device: str
        :type model_name: str
        :type pretrained: str

        :return: None
        """
        if not _HAS_OPENCLIP:
            raise RuntimeError("open_clip_torch not installed. pip install open_clip_torch")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode list of texts to embeddings.

        :param texts: List of text strings
        :type texts: List[str]

        :return: Text embeddings tensor
        :rtype: torch.Tensor
        """
        toks = self.tokenizer(texts)
        return F.normalize(self.model.encode_text(toks.to(self.device)), dim=-1)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to embedding.

        :param image: Input image given as PIL Image
        :type image: Image.Image
        :return: Image embedding tensor
        :rtype: torch.Tensor
        """
        imt = self.preprocess(image).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_image(imt), dim=-1).squeeze(0)

    @torch.no_grad()
    def crop_embed(self, image: Image.Image, box_xyxy: List[float]) -> torch.Tensor:
        """
        Crop image region and encode to embedding.

        :param image: Input image given as PIL Image Object
        :type image: Image.Image
        :param box_xyxy: Bounding box in xyxy format

        :return: Image region embedding tensor
        :rtype: torch.Tensor
        """
        x1,y1,x2,y2 = [int(v) for v in box_xyxy]
        crop = image.crop((max(0,x1), max(0,y1), max(1,x2), max(1,y2)))
        return self.encode_image(crop)


# --------------------------------
# Main Predictor
# --------------------------------
class Predictor:
    """
    Unified closed + open vocabulary object detection predictor.
    """
    def __init__(
        self,
        model_dir: str,
        lexicon_path: Optional[str],
        device: str = Config.DEVICE,
        use_openclip: bool = True
        ):
        """
        Initialize the Predictor with model and lexicon.

        :param model_dir: Model directory path
        :param lexicon_path: Lexicon file path
        :param device: Computation device
        :param use_openclip: Flag to use OpenCLIP for text-image matching

        :return: None
        """
        self.device = device
        self.model = RTDetrForObjectDetection.from_pretrained(model_dir).to(device).eval()
        self.processor = RTDetrImageProcessor.from_pretrained(Config.RTDETR_IMAGE_PROCESSOR)

        # id2label from config (string keys)
        id2label = getattr(self.model.config, "id2label", None)
        if id2label is None or len(id2label) == 0:
            # fallback to a default mapping
            raise ValueError("---> Model config missing id2label; make sure you saved with labels <---")

        # Normalize to int keys
        self.id2label = {int(k): v for k, v in id2label.items()}

        self.closed_thresh = Config.CLOSED_SCORE_THRESH
        self.open_thresh   = Config.OPEN_SCORE_THRESH

        self.lexicon_all = load_lexicon(lexicon_path)
        self.use_openclip = use_openclip and _HAS_OPENCLIP
        self.clip = ClipHelper(device=Config.DEVICE) if self.use_openclip else None

        # Cache text embeddings for lexicon
        self._lex_text_emb: Optional[torch.Tensor] = None
        if self.clip:
            self._lex_text_emb = self.clip.encode_text([f"a photo of a {t}" for t in self.lexicon_all])

    @torch.no_grad()
    def _closed_set_detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Run closed-set detection using RTDETR head.

        :param image: Input image given as PIL Image Object
        :type image: Image.Image
        :return: List of closed-set detections
        :rtype: List[Dict[str, Any]]
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # RTDetrImageProcessor -> postprocess
        target_sizes = torch.tensor([(image.height, image.width)]).to(self.device)
        det = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]

        # Convert to records
        out = []
        for score, label_id, box in zip(det["scores"].tolist(), det["labels"].tolist(), det["boxes"].tolist()):
            lab = self.id2label.get(int(label_id), str(label_id))
            rec = {"label": lab, "score": float(score), "box": [float(x) for x in box], "source": "closed"}
            out.append(rec)

        # Filter by score
        out = [d for d in out if d["score"] >= self.closed_thresh]

        # NMS per label
        out = nms(out, Config.IOU_MERGE_THRESH)

        # return output
        return out

    @torch.no_grad()
    def _shortlist_lexicon(self, image: Image.Image) -> Tuple[List[str], torch.Tensor]:
        """
        Use global image-text similarity to shortlist labels from a large lexicon.
        Returns (labels_list, embeddings_subset)

        :param image: Input image given as PIL Image Object
        :type image: Image.Image

        :return: Tuple of shortlisted labels and their embeddings
        :rtype: Tuple[List[str], torch.Tensor]
        """
        if not self.clip or self._lex_text_emb is None:
            # No CLIP: return entire lexicon (can be slower)
            labs = self.lexicon_all
            return labs, None

        img_emb = self.clip.encode_image(image)                       # [D]
        sims = (img_emb @ self._lex_text_emb.T).float().cpu().numpy() # [L]
        idx = np.argsort(-sims)[:Config.TOPK_CLIP_LEXICON]
        labs = [self.lexicon_all[i] for i in idx]
        embs = self._lex_text_emb[idx]                                # [K,D]
        return labs, embs

    @torch.no_grad()
    def _open_vocab_on_proposals(
        self,
        image: Image.Image,
        proposals: List[Dict[str, Any]],
        deny_labels: List[str],
        provided_prompts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        For each proposal box, crop and match against text:
        - if prompts provided: use them
        - else: shortlist from lexicon via global image-text sim
        Then pick best non-denied label above OPEN_SCORE_THRESH.

        :param image: Input image given as PIL Image Object
        :param proposals: List of proposal detections
        :param deny_labels: List of labels to deny (already found closed-set)
        :param provided_prompts: Optional list of provided prompts

        :return: List of open-vocab detections
        :rtype: List[Dict[str, Any]]
        """
        if len(proposals) == 0:
            return []

        # Build candidate label bank
        if provided_prompts and len(provided_prompts) > 0:
            cand_labels = [p.strip() for p in provided_prompts if p.strip()]
            if self.clip:
                cand_embs = self.clip.encode_text([f"a photo of a {t}" for t in cand_labels])
            else:
                cand_embs = None
        else:
            cand_labels, cand_embs = self._shortlist_lexicon(image)

        # Remove already found closed labels
        closed_set = set([c.lower() for c in deny_labels])
        cand = [(l, i) for i, l in enumerate(cand_labels) if l.lower() not in closed_set]
        if len(cand) == 0:
            return []
        cand_labels = [l for (l, _) in cand]
        if self.clip and cand_embs is not None:
            cand_embs = cand_embs[[i for (_, i) in cand]]  # subset
        elif self.clip:
            cand_embs = self.clip.encode_text([f"a photo of a {t}" for t in cand_labels])

        # Score each proposal crop -> best label
        ov_dets = []
        for prop in proposals:
            crop_emb = self.clip.crop_embed(image, prop["box"]) if self.clip else None
            if crop_emb is None:
                continue
            sims = (crop_emb @ cand_embs.T).float()  # [K]
            topk = min(Config.MAX_CAND_PER_BOX, sims.shape[-1])
            vals, idx = torch.topk(sims, k=topk)
            best_val = float(vals[0].item())
            best_lab = cand_labels[int(idx[0].item())]
            if best_val >= self.open_thresh:
                rec = {
                    "label": best_lab,
                    "score": best_val,
                    "box": prop["box"],
                    "source": "open"
                }
                ov_dets.append(rec)

        # NMS within open labels
        ov_dets = nms(ov_dets, Config.IOU_MERGE_THRESH)
        return ov_dets

    def predict(
        self,
        image_path: str,
        general_prompts: Optional[List[str]] = None,
        return_boxes: bool = True
    ) -> Dict[str, Any]:
        """
        Run both paths and return unique label set + (optionally) deduped detections.

        :param image_path: Path to input image
        :param general_prompts: Optional list of prompts for open-vocab detection
        :param return_boxes: Flag to return boxes and scores along with labels

        :return: Dictionary with unique labels and optional detections
        :rtype: Dict[str, Any]
        """
        image = Image.open(image_path).convert("RGB")

        # 1) Closed-set detection
        closed = self._closed_set_detect(image)

        # 2) Open-vocab using proposals (we use the same proposals = closed detections' boxes)
        deny = [d["label"] for d in closed]
        open_extra = self._open_vocab_on_proposals(image, closed, deny_labels=deny, provided_prompts=general_prompts)

        # 3) Merge & dedupe across sources: prefer closed label if same label overlaps
        merged = closed[:]
        for o in open_extra:
            # If overlaps strongly with a closed det of same label, keep the higher-score one
            keep = True
            for c in merged:
                if c["label"].lower() == o["label"].lower():
                    if box_iou(np.array(c["box"]), np.array(o["box"])) >= Config.IOU_MERGE_THRESH:
                        if o["score"] > c["score"]:
                            c.update(o)  # replace
                        keep = False
                        break
            if keep:
                merged.append(o)

        # final label set
        unique_labels = sorted(set([m["label"] for m in merged]))

        out = {"labels": unique_labels}
        if return_boxes:
            out["detections"] = merged
        return out




