import os, json, numpy as np
from pathlib import Path
from PIL import Image
import torch
import time
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# LVIS evaluation (rare/common/frequent)
from lvis import LVIS, LVISEval
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """


"""
Script to run evaluation using DINOv3 + RTDETR with OV head.

@author: Nikhil Bhargava
@date: 2025-10-18
@license: Apache-2.0
@description: This script performs evaluation on COCO and LVIS datasets using a trained DINOv3 + RTDETR model.
@copyright: Copyright 2025 Nikhil Bhargava
"""


logger = get_logger(__name__)


def load_model(model_dir: str) -> Tuple[RTDetrForObjectDetection, RTDetrImageProcessor]:
    """
    Load the RTDETR model and image processor from the specified directory.

    :return: model, processor
    :rtype: Tuple[RTDetrForObjectDetection, RTDetrImageProcessor]
    """
    model = RTDetrForObjectDetection.from_pretrained(model_dir).to(Config.DEVICE).eval()
    proc = RTDetrImageProcessor.from_pretrained(Config.RTDETR_IMAGE_PROCESSOR)
    return model, proc

def resolve_path(fn:str, val_img_dirs: List[str]) -> Path:
    """
    Resolve the full path of an image file by checking multiple directories.

    :param val_img_dirs: List of directories to search for the image
    :param fn: Filename of the image
    :return: Full path to the image file
    :rtype: Path
    """
    for root in val_img_dirs:
        p = Path(root)/fn
        if p.exists(): return p
    return Path(fn)

def infer_images(img_paths: List[str],
                 model_dir: str = None,
                 threshold=0.3,
                 text_prompts: List[str]=None) -> List[Tuple[str, Dict]]:
    """
    Inference on a list of images using the RTDETR model.

    :param img_paths: Path to images
    :param threshold: Threshold for detection
    :param text_prompts: Text prompts for OV head (if used)

    :return: List of tuples containing image path and detection results
    :rtype: List[Tuple[str, Dict]]
    """
    model, proc = load_model(model_dir)
    outs = []
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        inputs = proc(images=im, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_object_detection(out, target_sizes=torch.tensor([(im.height, im.width)]).to(device), threshold=threshold)[0]
        outs.append((p, res))
    return outs

def coco_eval(model_dir: str, val_img_dirs: List[str], val_json: str = None):
    """
    Evaluate COCO metrics on COCO val using our predictions

    1. Run inference on all val images and save results in COCO json format
    2. Use pycocotools to compute COCO metrics
    3. Plot PR curves for a few classes
    4. Save results to MODEL_DIR

    :param val_img_dirs: List of directories containing validation images
    :param model_dir: Directory where the trained model is saved
    :param val_json: Path to COCO-format validation annotations JSON

    :return: None
    """
    # 1. Load the model and processor
    model, proc = load_model(model_dir)

    # 2. Load COCO val annotations
    coco = COCO(str(val_json))

    # 3. Get image ids and run inference
    img_ids = coco.getImgIds()
    results = []
    for iid in img_ids:
        im_info = coco.loadImgs(iid)[0]

        # Get the full image path
        p = resolve_path(im_info["file_name"], val_img_dirs)

        # Convert the image to RGB and run inference
        im = Image.open(p).convert("RGB")
        inputs = proc(images=im, return_tensors="pt").to(Config.DEVICE)
        with torch.inference.no_grad():
            out = model(**inputs)

        det = proc.post_process_object_detection(out,
                                                 target_sizes=torch.tensor([(im.height, im.width)]).to(Config.DEVICE),
                                                 threshold=0.0)[0]
        # Convert to COCO json
        for s, l, b in zip(det["scores"].cpu().tolist(), det["labels"].cpu().tolist(), det["boxes"].cpu().tolist()):
            x1,y1,x2,y2 = b
            results.append({
                "image_id": iid,
                "category_id": int(l),
                "bbox": [x1, y1, x2-x1, y2-y1], # COCO format: top-left point (x,y), width, height
                "score": float(s),
            })

    # Store the results
    res_file = Path(model_dir)/"coco_detections_val.json"
    json.dump(results, open(res_file,"w"))
    coco_dt = coco.loadRes(str(res_file))
    coco_eval_handle = COCOeval(coco, coco_dt, "bbox")

    # Run evaluation
    coco_eval_handle.evaluate(); coco_eval_handle.accumulate(); coco_eval_handle.summarize()

    # Plot PR curve for a few classes
    # E.eval contains precision array with dims: [TxRxKxAxM]
    precisions = coco_eval_handle.eval["precision"]  # T=10 IoUs

    # PR at IoU=0.5 for first 10 classes
    iou_thr = np.where(np.isclose(coco_eval_handle.params.iouThrs, 0.5))[0][0]
    cls_ids = coco_eval_handle.params.catIds[: min(10, len(coco_eval_handle.params.catIds))]
    for cid in cls_ids:
        # Max recall curve
        pi = precisions[iou_thr, :, coco_eval_handle.params.catIds.index(cid), 0, -1]
        plt.figure()
        plt.plot(np.linspace(0,1,len(pi)), pi)
        plt.title(f"PR curve @ IoU=0.5 for label: {cid}")
        plt.xlabel("Recall [0-1]"); plt.ylabel("Precision [0-1")

        # Save the plot
        plt.savefig(Path(model_dir)/f"pr_cat{cid}.png", dpi=320)
        plt.close()

def lvis_eval(model_dir,
              lvis_val_json="datasets/mixture/annotations/lvis/lvis_v1_val.json") -> None:
    """
    LVIS evaluation on LVIS val using our predictions
    :param model_dir: Directory where the trained model is saved
    :param lvis_val_json: Path to LVIS-format validation annotations JSON

    :return: None
    :raises: Following exceptions:
    - ValueError: If lvis_val_json is invalid
    - FileNotFoundError: If predictions file does not exist
    """
    # Evaluate LVIS metrics on LVIS val using our predictions (we’ll reuse the COCO-format prediction file)
    resfile = Path(model_dir)/"coco_detections_val.json"
    if not resfile.exists():
        raise FileNotFoundError("Run coco_eval() first to write predictions")

    try:
        # Get the LVIS ground truth
        lvis = LVIS(lvis_val_json)

        # Load our COCO-format predictions
        preds = json.load(open(resfile))

        # Map category ids if needed (ensure IDs align to LVIS val IDs); if merged mapping changed, build a remap here.
        lvis_results = []
        for d in preds:
            lvis_results.append({
                "image_id": d["image_id"],
                "category_id": d["category_id"],  # assumes aligned IDs for LVIS images subset
                "bbox": d["bbox"], "score": d["score"]
            })
        pred_path = Path(model_dir)/"lvis_preds.json"
        json.dump(lvis_results, open(pred_path,"w"))
        lvis_dt = lvis.load_res(str(pred_path))
        le = LVISEval(lvis, lvis_dt, iou_type="bbox")
        le.run(); le.print_results()  # reports AP, APr/APc/APf

    except ValueError as e:
        logger.exception(f"---> ERROR: INVALID LVIS JSON: {e} <---")

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
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_TEST")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

                # Get the parent path
                par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                # Get the final merged COCO annotation output path
                coco_out = os.path.join(par_path, str(Config.COCO_ANN))
                Path(coco_out).mkdir(parents=True, exist_ok=True)
                val_json = os.path.join(coco_out, "instances_val_merged.json")

                # Make absolute paths for img_dirs
                val_img_dirs = list()
                for dir in Config.VAL_IMG_DIRS:
                    path = os.path.join(par_path, dir)
                    val_img_dirs.append(path)

                # Get the trained model path
                output_dir = os.path.join(par_path, os.getenv("RAPTOR_PATHS_MODEL_DIR", "runs/dinov3_rtdetr"))
                model_dir= os.path.join(output_dir, "final")


                # 2. Start Evaluation
                start_time = time.perf_counter()

                # A. COCO Evaluation
                coco_eval(model_dir, val_img_dirs, val_json)

                # B. LVIS Evaluation
                lvis_eval(model_dir)

                # C. Custom Evaluation (if any)
                # custom_eval(model_dir)

                end_time = time.perf_counter()
                logger.debug(f"---> DINOv3 WITH RTDETR OV HEAD EVALUATED ON "
                             f"DATASETS: {os.getenv("RAPTOR_DATASETS")} IN "
                             f"TIME: {(end_time - start_time) / 60 * 60: .3f} HOURS <---")

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
