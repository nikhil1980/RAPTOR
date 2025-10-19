import os, json, math
import time
from pathlib import Path
from Collections import defaultdict
from typing import Any, Dict, List

import torch, torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrConfig,
    Trainer,
    TrainingArguments
)
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModel
import wandb
from pycocotools.coco import COCO
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
        self.images = self.coco.loadImgs(self.img_ids)

    def _resolve_path(self, file_name: str) -> Path:
        """
        Resolve the full path of an image file by searching through the provided image root directories.

        1. Iterates through each directory in img_roots.
        2. Constructs the full path by joining the directory with the file_name.
        3. Checks if the constructed path exists.
        4. If the path exists, returns it as a Path object.
        5. If the file is not found in any of the directories, returns the file_name as a Path object (may lead to a FileNotFoundError later).

        :param file_name: Name of the image file.

        :return: Full path to the image file as a Path object.
        :rtype: Path
        """
        for root in self.img_roots:
            p = Path(root)/file_name
            if p.exists(): return p

        return Path(file_name)

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
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=iminfo["id"])
        anns = self.coco.loadAnns(ann_ids)
        objects = []
        for a in anns:
            x,y,w,h = a["bbox"]
            objects.append({"category_id": int(a["category_id"]), "bbox": [x,y,w,h], "area": a.get("area", w*h), "iscrowd": a.get("iscrowd",0)})
        target = {"image_id": int(iminfo["id"]), "annotations": objects}

        processed = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
            size=Config.IMAGE_SIZE,
        )
        processed["pixel_values"] = processed["pixel_values"].squeeze(0)
        processed["labels"] = [{k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                                for k,v in processed["labels"][0].items()}]
        return processed


def build_idmaps(coco_json: Path) -> (Dict[int,str], Dict[str,int]):
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
    cats = js["categories"]
    id2label = {c["id"]: c["name"] for c in cats}
    label2id = {v:k for k,v in id2label.items()}
    return id2label, label2id


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
    txtm = AutoModel.from_pretrained(Config.TEXT_ENCODER).to(device).eval()
    outs = []
    with torch.no_grad():
        for name in class_names:
            prompt = f"a photo of a {name}"
            out = txtm(**tok(prompt, return_tensors="pt").to(device))
            emb = out.text_embeds if hasattr(out, "text_embeds") and out.text_embeds is not None else out.last_hidden_state[:,0]
            outs.append(emb[0])
    return torch.stack(outs, dim=0)  # [T, D]


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
        self.text_embeds = text_embeds

    def forward(self, **batch):
        """
        Forward pass with OV head and loss.

        :param batch: Input batch containing pixel values and labels.
        :return: Model outputs with added OV loss if in training mode.
        """
        outputs = self.base(**batch)
        if self.training and self.ov_head is not None and self.text_embeds is not None:
            dec = getattr(outputs, "decoder_hidden_states", None)
            if dec is not None and len(dec) > 0:
                dec = dec[-1]  # [B,Q,C]
                ov_logits = self.ov_head(dec, self.text_embeds)  # [B,Q,T]
                pred = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                hard = pred.argmax(-1)  # [B,Q]
                b, q = hard.shape
                T = ov_logits.shape[-1]
                num_labels = pred.shape[-1]
                mask = hard < (num_labels - 1)
                target = torch.zeros((b, q, T), device=ov_logits.device)
                target.scatter_(-1, hard.unsqueeze(-1).clamp_max(T-1), 1.0)
                ov_loss = binary_focal_with_logits(
                    ov_logits[mask], target[mask],
                    alpha=LossConfig.ov_alpha,
                    gamma=LossConfig.ov_gamma,
                    reduction="mean"
                )
                outputs.loss = outputs.loss + self.loss_cfg.ov_weight * ov_loss
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
    cfg = RTDetrConfig(
        num_labels=num_labels,
        id2label={i: id2label[i] for i in id2label},
        label2id=label2id,
        use_pretrained_backbone=True,
        backbone=Config.PRETRAINED_BACKBONE,             # DINOv3 backbone
        backbone_kwargs={"out_indices": (0,1,2)},
        freeze_backbone_batch_norms=True,

        # ---- long-tail / focal knobs ----
        use_focal_loss=True,
        focal_loss_alpha=Config.FOREGROUND_ALPHA,
        focal_loss_gamma=Config.FOCAL_GAMMA,
        weight_loss_vfl=Config.VFL_WEIGHT,               # 1.0 keeps VFL on; 0.0 for pure focal
        eos_coefficient=Config.EOS_COEF,                  # downweight "no object"
    )
    model = RTDetrForObjectDetection(cfg)
    if Config.FREEZE_BACKBONE:
        for _, p in model.backbone.named_parameters(): p.requires_grad_(False)

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
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=os.getenv("RAPTOR_ACCELERATE_NUM_PROCESSES", 4),
            collate_fn=collate_fn,
            pin_memory=True
        )

    def compute_loss(self, model, inputs, return_outputs=False)-> torch.Tensor:
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
    if os.getenv("RAPTOR_WANDB_PROJECT_ENABLED", "true").lower() in ("true", "1", "yes"):
        wandb.init(project=os.getenv("RAPTOR_WANDB_PROJECT", "dinov3_lvis_oi_rtdetr"),
               name=os.getenv("RAPTOR_WANDB_RUN_NAME", "focal_vfl_classaware"))

    image_processor, train_ds, val_ds, id2label, label2id = build_processor_and_datasets()
    model = build_model(len(id2label),
                        id2label,
                        {v: k for k, v in id2label.items()},
                        device=Config.DEVICE
                        )
    model_dir = os.path.join(BASE_PATH, os.getenv("RAPTOR_PATHS_MODEL_DIR", "runs/dinov3_rtdetr"))
    args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=int(os.getenv("RAPTOR_TRAIN_BATCH_SIZE", 2)),
        per_device_eval_batch_size=int(os.getenv("RAPTOR_TRAIN_VAL_BATCH_SIZE", 2)),
        gradient_accumulation_steps=int(os.getenv("RAPTOR_TRAIN_ACCUM_STEPS", 4)),
        learning_rate=float(os.getenv("RAPTOR_TRAIN_LEARNING_RATE", 2e-4)),
        num_train_epochs=int(os.getenv("RAPTOR_TRAIN_EPOCHS", 50)),
        lr_scheduler_type=os.getenv("RAPTOR_TRAIN_LR_SCHEDULER_TYPE", "cosine"),
        warmup_ratio=float(os.getenv("RAPTOR_TRAIN_WARMUP_RATIO", 0.05)),
        weight_decay=float(os.getenv("RAPTOR_TRAIN_WEIGHT_DECAY", 0.05)),
        logging_steps=int(os.getenv("RAPTOR_TRAIN_LOGGING_STEPS", 50)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=int(os.getenv("RAPTOR_ACCELERATE_NUM_PROCESSES", 4)),
        report_to=["wandb"],
    )

    trainer = LongTailTrainer(
        image_processor=image_processor,
        train_ds=train_ds,
        model=model,
        args=args,
        data_collator=collate_fn,
        eval_dataset=val_ds,
    )
    trainer.train()
    save_dir = os.path.join(model_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
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
                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

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
