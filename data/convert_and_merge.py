import os
import time
import json
import re
from pathlib import Path
import fiftyone as fo
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.config import Config
from common.myargparser import build_myargparser
""" User Modules """

""""
Basic Script to convert and merge datasets.

@author: Nikhil Bhargava
@date: 2025-10-15
@license: Apache-2.0
@description: This script converts datasets to COCO format and merges them if needed.
@copyright: Copyright 2025 Nikhil Bhargava
"""

logger = get_logger(__name__)


def export_openimages_to_coco(split: str ,
                              out_images_dir: Path,
                              out_json: json):
    """
    Export FiftyOne OpenImages dataset to COCO format.

    :param split: "train" or "val"
    :param out_images_dir: directory to save images
    :param out_json: path to save COCO annotations JSON

    :return: None
    """
    ds_name = f"openimages-v7-{split}-det"
    assert ds_name in fo.list_datasets(), f"---> FiftyOne dataset {ds_name} not found. Run download first!!! <---"

    ds = fo.load_dataset(ds_name)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    ds.export(
        export_dir=str(out_images_dir),
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=str(out_json),
        label_field="detections",
        export_media="symlink",
    )


def lvis_to_coco_det(lvis_json: json, out_json: json):
    """
    Convert LVIS annotations to COCO detection format.

    :param lvis_json: path to LVIS annotations JSON
    :param out_json: path to save COCO annotations JSON

    :return: None
    """
    lvis = json.load(open(lvis_json))
    images = lvis["images"]
    categories = lvis["categories"]
    anns = []
    for a in lvis["annotations"]:
        if a.get("bbox"):
            x,y,w,h = a["bbox"]
            anns.append({
                "id": a["id"],
                "image_id": a["image_id"],
                "category_id": a["category_id"],
                "bbox": [x,y,w,h],
                "iscrowd": a.get("iscrowd", 0),
                "area": a.get("area", w*h),
            })
    coco = {"images": images, "annotations": anns, "categories": categories}
    json.dump(coco, open(out_json,"w"))

def unify_and_merge(json_paths: json, out_json: json):
    """
    Merge multiple COCO-format JSONs into one, unifying category space.

    :param json_paths: list of paths to COCO annotations JSONs
    :param out_json: location to save merged COCO annotations JSON

    :return: None
    """
    def norm(name: str) -> str:
        """
        Normalize category name for matching.

        :param name: category name

        :return: normalized name
        """
        return re.sub(r"\s+"," ", name.strip().lower())

    merged = {"images": [],
              "annotations": [],
              "categories": []
              }

    cat_name_to_id = {}
    next_img_id, next_ann_id, next_cat_id = 1, 1, 1

    # build unified category space
    for jp in json_paths:
        js = json.load(open(jp))
        for c in js["categories"]:
            nm = norm(c["name"])
            if nm not in cat_name_to_id:
                cat_name_to_id[nm] = next_cat_id
                merged["categories"].append({"id": next_cat_id, "name": c["name"]})
                next_cat_id += 1

    for jp in json_paths:
        js = json.load(open(jp))
        cat_by_id = {c["id"]: c["name"] for c in js["categories"]}

        img_id_map = {}

        # Build new image and annotation entries with remapped IDs
        for im in js["images"]:
            new_id = next_img_id; next_img_id += 1
            img_id_map[im["id"]] = new_id
            merged["images"].append({
                "id": new_id,
                "file_name": im["file_name"],
                "height": im.get("height"),
                "width": im.get("width"),
            })

        for an in js["annotations"]:
            cname = cat_by_id[an["category_id"]]
            cid = cat_name_to_id[norm(cname)]
            merged["annotations"].append({
                "id": next_ann_id,
                "image_id": img_id_map[an["image_id"]],
                "category_id": cid,
                "bbox": an["bbox"],
                "iscrowd": an.get("iscrowd",0),
                "area": an.get("area", an["bbox"][2]*an["bbox"][3]),
            })
            next_ann_id += 1

    json.dump(merged, open(out_json,"w"))
    logger.debug(f"---> [MERGE] {out_json} -> IMAGES={len(merged['images'])}, "
                 f"ANNOTATIONS={len(merged['annotations'])}, "
                 f"CLASSES={len(merged['categories'])} <---")


def main():
    """
    Main function to run the application.

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
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_MERGE_DATA")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")
                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

                # 2. Export and Merge datasets to COCO format
                base_path = os.path.join(par_path, Config.ROOT)
                Path(base_path).mkdir(parents=True, exist_ok=True)

                image_path = os.path.join(base_path, "images")
                Path(image_path).mkdir(parents=True, exist_ok=True)

                annotation_path = os.path.join(base_path, "annotations")
                Path(annotation_path).mkdir(parents=True, exist_ok=True)

                start_time = time.perf_counter()
                coco_out = os.path.join(par_path, str(Config.COCO_ANN))
                Path(coco_out).mkdir(parents=True, exist_ok=True)

                # A. OPENIMAGESv7 --> COCO for both TRAIN and VAL
                coco_merged_train = os.path.join(par_path, str(Config.TRAIN_IMG_DIRS[1]))
                Path(coco_merged_train).mkdir(parents=True, exist_ok=True) # 1 is for Merged COCO Train Anno

                coco_merged_val = os.path.join(par_path, str(Config.VAL_IMG_DIRS[1]))
                Path(coco_merged_val).mkdir(parents=True, exist_ok=True)   # 1 is for Merged COCO Val Anno

                export_openimages_to_coco("train",
                                          Path(coco_merged_train),
                                          os.path.join(coco_out, "instances_oi_train.json"))

                export_openimages_to_coco("validation",
                                          Path(coco_merged_val),
                                          os.path.join(coco_out, "instances_oi_val.json"))

                # B. LVIS (bbox-only) to COCO
                lvis_train = annotation_path + "/" + "lvis" + "/" + "lvis_v1_train.json"
                lvis_val = annotation_path + "/" + "lvis" + "/" + "lvis_v1_val.json"
                lvis_to_coco_det(lvis_train, os.path.join(coco_out, "instances_lvis_train.json"))
                lvis_to_coco_det(lvis_val, os.path.join(coco_out, "instances_lvis_val.json"))

                # C. MERGE
                unify_and_merge(
                    [os.path.join(coco_out, "instances_lvis_train.json"),
                     os.path.join(coco_out, "instances_oi_train.json")],
                    os.path.join(coco_out, "instances_train_merged.json")
                )
                unify_and_merge(
                    [os.path.join(coco_out, "instances_lvis_val.json"),
                     os.path.join(coco_out, "instances_oi_val.json")],
                    os.path.join(coco_out, "instances_val_merged.json")
                )

                end_time = time.perf_counter()
                logger.debug(f"---> DATASETS CONVERSION/EXPORT TO COCO JSON AND "
                             f"MERGE DONE IN TIME {(end_time - start_time) / 60 * 60: .3f} HOURS <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")

            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_MERGE_DATA")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

if __name__ == '__main__':
    main()
