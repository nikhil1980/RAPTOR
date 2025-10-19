import os
import time
import json, re
from pathlib import Path
import torch
from dataclasses import dataclass
""" System Modules """

@dataclass
class Config:
    """
    Configuration class for training DINOv3 + RTDETR on a mixture of datasets (MoD).
    This class encapsulates all the configuration parameters needed for the training process.
    1. Paths to datasets and annotations
    2. Model parameters including backbone and text encoder
    3. Training parameters such as number of workers, image size, and loss function coefficients
    4. Data augmentation and preprocessing settings
    5. Hyperparameters for loss functions including focal loss and VFL
    6. Flags for freezing backbone and using OV head
    7. Any other relevant settings for training and evaluation

    This class can be easily extended to include more parameters as needed.
    """
    ROOT = "datasets/mixture"
    COCO_ANN = ROOT + "/" + "annotations/coco_merged"
    TRAIN_JSON = COCO_ANN + "/" + "instances_train_merged.json"
    VAL_JSON = COCO_ANN + "/" + "instances_val_merged.json"
    TRAIN_IMG_DIRS = [ROOT + "/" + "images/train2017", COCO_ANN + "/" + "oi_train"] # DON'T CHANGE ORDER
    VAL_IMG_DIRS = [ROOT + "/" + "images/val2017", COCO_ANN + "/" + "oi_val"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PRETRAINED_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # accept license
    RTDETR_IMAGE_PROCESSOR = "PekingU/rtdetr_r50vd"
    TEXT_ENCODER = "google/siglip-so400m-patch14-384"
    CLIP_MODEL =  "ViT-L-14"
    CLIP_SOURCE = "openai"
    IMAGE_SIZE = {"shortest_edge": 640, "longest_edge": 640}
    FREEZE_BACKBONE = True
    USE_OV_HEAD = True

    FOREGROUND_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    EOS_COEF = 5e-5
    VFL_WEIGHT = 1.0  # set 0.0 for *pure* focal

    CLOSED_SCORE_THRESH = 0.30
    OPEN_SCORE_THRESH = 0.28  # similarity threshold for OV label adoption (0-1 cosine)
    IOU_MERGE_THRESH = 0.55  # NMS/merge threshold when deduping
    TOPK_CLIP_LEXICON = 200  # shortlist from a big lexicon via global image embedding
    MAX_CAND_PER_BOX = 40  # re-rank top text candidates per box

    # If you have a big lexicon file (one label per line), put it here:
    DEFAULT_LEXICON_TXT = "resources/open_vocab_lexicon.txt"  # optional
    FALLBACK_LEXICON = [
        # a small fallback so it runs out of the box
        "ukulele", "martini glass", "handbag", "laptop", "skateboard", "microwave", "zebra", "peacock",
        "scooter", "syringe", "passport", "toothbrush", "drone", "wine bottle", "safety helmet", "power drill",
        "traffic sign", "water bottle", "dumbbell", "walkie-talkie", "wheelchair", "suitcase", "tripod", "headphones"
    ]

@dataclass
class LossConfig:
    ov_alpha: float = 0.25
    ov_gamma: float = 2.0
    ov_weight: float = 0.25