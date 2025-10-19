import os, json, numpy as np
from pathlib import Path
from PIL import Image
import torch
import time
from transformers import(RTDetrForObjectDetection,
                         RTDetrImageProcessor,
                         AutoTokenizer,
                         AutoModel
                         )

from typing import List, Dict
from PIL import Image
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """

"""
Script to run zero-shot object detection using DINOv3 + RTDETR with OV head.

@author: Nikhil Bhargava
@date: 2025-10-18
@license: Apache-2.0
@description: This script performs zero-shot object detection using a pre-trained DINOv3 + RTDETR model with an open-vocabulary head. It encodes text prompts and matches them against detected objects in images.
@copyright: Copyright 2025 Nikhil Bhargava
"""

logger = get_logger(__name__)

def encode_text(prompts):
    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    txt = AutoModel.from_pretrained(TEXT_ENCODER).to(device).eval()
    with torch.no_grad():
        outs = txt(**tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device))
        if hasattr(outs, "text_embeds") and outs.text_embeds is not None:
            return torch.nn.functional.normalize(outs.text_embeds, dim=-1)
        return torch.nn.functional.normalize(outs.last_hidden_state[:,0], dim=-1)

def ov_infer(model_dir: str,
             image_path: Path,
             prompts: List[str],
             score_thresh: float=0.25
             )-> List[Dict]:
    """
    Perform zero-shot object detection using DINOv3 + RTDETR with OV head.

    :param model_dir: Model directory
    :param image_path: Path to input image
    :param prompts: text prompts for object detection
    :param score_thresh: threshold for detection scores

    :return: Returns list of detected objects with labels, scores, and bounding boxes
    :rtype: List[Dict]
    """
    # 1. Load the model and processor
    model = RTDetrForObjectDetection.from_pretrained(model_dir).to(Config.DEVICE).eval()
    proc = RTDetrImageProcessor.from_pretrained(Config.RTDETR_IMAGE_PROCESSOR)

    # 2. Encode text prompts
    te = encode_text([f"a photo of a {p}" for p in prompts])  # [T,D]

    # 3. Lad the image and run inference
    im = Image.open(image_path).convert("RGB")
    inputs = proc(images=im, return_tensors="pt").to(Config.DEVICE)
    with torch.inference_mode():
        out = model(**inputs)

    # get decoder features to score vs text
    dec = out.decoder_hidden_states[-1]  # [B,Q,C]

    # small projection: reuse OVHead projection from training if present
    # fallback: l2-normalize decoder and a linear mapping to text dim=768
    ctx = dec.shape[-1]
    proj = torch.nn.Linear(ctx, te.shape[-1], bias=False).to(Config.DEVICE)
    with torch.inference_mode():
        logits = torch.einsum("bqc,tc->bqt", torch.nn.functional.normalize(proj(dec),dim=-1), te)  # [1,Q,T]
        scores, labels = logits.softmax(-1).max(-1)  # pick best text per query

    # Decode boxes to image space:
    det = proc.post_process_object_detection(out, target_sizes=torch.tensor([(im.height, im.width)]).to(device), threshold=0.0)[0]
    final = []
    for i,(b) in enumerate(det["boxes"]):
        if scores[0,i].item() >= score_thresh:
            final.append({"label": prompts[labels[0,i].item()],
                          "score": scores[0,i].item(),
                          "box": [float(x) for x in b.cpu().tolist()]})
    return final

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
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_ZERO_SHOT")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

                # Get the parent path
                par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                # Get the trained model path
                output_dir = os.path.join(par_path, os.getenv("RAPTOR_PATHS_MODEL_DIR", "runs/dinov3_rtdetr"))
                model_dir= os.path.join(output_dir, "final")

                # 2. Call Zero-shot Inference
                start_time = time.perf_counter()

                if args.test_image is None or not os.path.isfile(args.test_image) or args.test_tags is None:
                    logger.error(f"---> PLEASE PROVIDE BOTH --test-image AND --test-tags ARGUMENTS. "
                                 f"SWITCHING TO DEFAULT <---")
                    test_image = os.path.join(par_path, "datasets/mixture/images/val2017/000000039769.jpg")
                    test_tags = ["cat", "dog", "person", "car", "bicycle", "bus", "truck"]
                else:
                    test_image = args.test_image
                    test_tags = args.test_tags.split(",")

                logger.info(f"---> RUNNING ZERO-SHOT INFERENCE ON IMAGE: {test_image} WITH TAGS: {test_tags} <---")

                result = ov_infer(model_dir, test_image, test_tags)

                logger.info(f"---> ZERO-SHOT DETECTION RESULTS: {json.dumps(result, indent=4)} <---")

                end_time = time.perf_counter()
                logger.debug(f"---> DINOv3 WITH RTDETR OV HEAD ZERO-SHOT ON "
                             f"IMAGE: {test_image} WITH TAGS: {test_tags} IN "
                             f"TIME: {(end_time - start_time): .3f} SECONDS <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")

            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_TEST")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

if __name__ == '__main__':
    main()
