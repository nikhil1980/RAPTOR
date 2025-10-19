import itertools, json, csv, os
import time
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from pycocotools.coco import COCO
from transformers import RTDetrImageProcessor
from PIL import Image
import torch, json
import torch
from transformers import TrainingArguments
from lvis import LVIS, LVISEval
""" System Modules """

import train.train_dinov3_rtdetr_ov as mod
from eval.eval_and_infer import coco_eval, load_model, resolve_path
from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """

logger = get_logger(__name__)



def lvis_eval_from_coco_json(pred_file: Path,
                             lvis_val_json: Path
                             )->Dict[str, float]:
    """
    Evaluate LVIS metrics from COCO-format predictions.
    :param pred_file: Path to predictions in COCO format
    :param lvis_val_json: Path to LVIS validation JSON annotations

    :return: Dictionary with AP, APr, APc, APf metrics
    :rtype: Dict[str, float]
    """
    lvis = LVIS(str(lvis_val_json))
    preds = json.load(open(pred_file))

    # assumes consistent id mapping for LVIS subset; remap if you changed mapping
    dt = lvis.load_res(preds)
    ev = LVISEval(lvis, dt, iou_type="bbox")
    ev.run()

    # Collect headline metrics:
    s = ev.get_results()
    # s contains AP, APr, APc, APf keys after print_results(); fallback parse:
    return {
        "AP":  s.get("AP", None),
        "APr": s.get("APr", None),
        "APc": s.get("APc", None),
        "APf": s.get("APf", None),
    }

def ablate(ablation_dir: Path,
           val_img_dirs:List[Path],
           val_json: json,
           alpha: float,
           gamma: float,
           epochs: int = 5,
           name_suffix: str = "") -> Tuple[Dict[str, float], Path]:
    """
    Ablate over specified alpha and gamma values.

    :param ablation_dir: Ablation Directory
    :param val_img_dirs: Validation Image Directories
    :param val_json: Validation JSON
    :param alpha: Focal Loss alpha
    :param gamma: Focal Loss gamma
    :param epochs: Number of training epochs
    :param name_suffix: Suffix for the run name

    :return: Tuple of (metrics dict, output directory path)
    :rtype: Tuple[Dict[str, float], Path]
    """
    # Rebuild model with new focal settings by monkey-patching module constants
    mod.FOREGROUND_ALPHA = alpha
    mod.FOCAL_GAMMA = gamma

    image_processor, train_ds, val_ds, id2label, label2id = mod.build_processor_and_datasets()
    model = mod.build_model(len(id2label), id2label, {v: k for k, v in id2label.items()},
                            device=("cuda" if torch.cuda.is_available() else "cpu"))

    tag = f"a{alpha}_g{gamma}{name_suffix}"
    out_dir = ablation_dir / f"{tag}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.05,
        logging_steps=50,
        evaluation_strategy="no",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to=[],
    )
    trainer = mod.LongTailTrainer(
        image_processor=image_processor,
        train_ds=train_ds,
        model=model,
        args=args,
        data_collator=mod.collate_fn,
        eval_dataset=val_ds,
    )
    trainer.train()

    # Write COCO-format predictions on val using eval script logic (lightweight)
    proc = RTDetrImageProcessor.from_pretrained(Config.RTDETR_IMAGE_PROCESSOR)
    model_loaded = type(model).from_pretrained(outdir / "final").to(Config.DEVICE).eval()

    # Reuse model saved at end
    model.save_pretrained(outdir / "final")

    # Emit predictions file into outdir/final/
    def _resolve(fn):
        for root in val_img_dirs:
            p = Path(root) / fn
            if p.exists(): return p
        return Path(fn)

    coco = COCO(str(val_json))
    img_ids = coco.getImgIds()
    results = []

    for iid in img_ids[:5000]:  # cap for speed; adjust/remove for full
        im_info = coco.loadImgs(iid)[0]
        p = _resolve(im_info["file_name"])
        im = Image.open(p).convert("RGB")
        inputs = proc(images=im, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            out = model_loaded(**inputs)
        det = proc.post_process_object_detection(out,
                                                 target_sizes=torch.tensor([(im.height, im.width)]).to(device),
                                                 threshold=0.0)[0]
        for s, l, b in zip(det["scores"].cpu().tolist(), det["labels"].cpu().tolist(), det["boxes"].cpu().tolist()):
            x1, y1, x2, y2 = b
            results.append(
                {"image_id": iid, "category_id": int(l), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})

    resfile = out_dir / "final" / "coco_detections_val.json"
    json.dump(results, open(resfile, "w"))

    # LVIS APr/APc/APf
    par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lvis_val_json = os.path.join(par_path, "datasets/mixture/annotations/lvis/lvis_v1_val.json")

    lvis_metrics = lvis_eval_from_coco_json(resfile, Path(lvis_val_json))
    return lvis_metrics, out_dir



def run_ablation(ablation_dir: Path, val_img_dirs: List[Path], val_json: json) -> None:
    """
    Run ablation over specified alpha and gamma values.
    :param ablation_dir: Ablation directory path
    :param val_img_dirs: List of validation image directories
    :param val_json: Validation JSON annotations

    :return: None
    """

    alphas = [0.25, 0.35, 0.5]
    gammas = [1.5, 2.0, 2.5]
    rows = []
    for a, g in itertools.product(alphas, gammas):
        m, _ = ablate(ablation_dir, val_img_dirs, val_json, g, epochs=3)
        logger.debug(f"---> [ABL] a={a}, g={g} -> {m} <---")
        rows.append({"alpha": a, "gamma": g, **m})

    # Save CSV summary
    with open(ablation_dir / "ablation_lvis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "gamma", "AP", "APr", "APc", "APf"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"---> [DONE] Wrote {ablation_dir / 'ablation_lvis.csv'} <---")


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

                # 2. Setup Arguments
                start_time = time.perf_counter()
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_ABLATION")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

                ablation_dir = os.path.join(par_path,
                                            os.getenv("RAPTOR_PATHS_ABLATION_DIR", "runs/ablation"))
                os.makedirs(ablation_dir, exist_ok=True)

                # Get the final merged COCO annotation output path
                coco_out = os.path.join(par_path, str(Config.COCO_ANN))
                Path(coco_out).mkdir(parents=True, exist_ok=True)
                val_json = os.path.join(coco_out, "instances_val_merged.json")

                # Make absolute paths for img_dirs
                val_img_dirs = list()
                for dir in Config.VAL_IMG_DIRS:
                    path = os.path.join(par_path, dir)
                    val_img_dirs.append(path)

                # 3. Start Ablation
                run_ablation(ablation_dir, val_img_dirs, val_json)

                end_time = time.perf_counter()
                logger.debug(f"---> ABLATION DONE OVER DINOv3 WITH RTDETR OV HEAD IN TIME {(end_time - start_time) / 60 * 60: .3f} HOURS <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")

            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_ABLATION")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")


if __name__ == '__main__':
    main()
