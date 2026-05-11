
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time
import json
import re
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm
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

To run:
tmux new-session -s convert
python data/convert_and_merge.py --config-file config.json >logs/convert.log 2>&1
tmux attach -t convert

tail -f logs/convert.log
"""

logger = get_logger(__name__)


def export_openimages_to_coco(split: str,
                              oi_root: str,
                              out_images_dir: Path,
                              out_json: str):
    """
    Convert OpenImages CSV annotations to COCO format directly (no FiftyOne dependency).
    Reads images already on disk, creates symlinks in out_images_dir, writes COCO JSON to out_json.

    :param split: "train" or "validation"
    :param oi_root: root of the OpenImages dataset directory
    :param out_images_dir: directory where image symlinks will be created
    :param out_json: path to write COCO annotations JSON

    :return: None
    """
    data_dir = os.path.join(oi_root, "data", split)
    labels_dir = os.path.join(oi_root, "labels")

    # 1. Build LabelName -> category mapping
    class_csv = os.path.join(labels_dir, "class-descriptions-boxable.csv")
    label_to_name: dict = {}
    with open(class_csv, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                label_to_name[row[0].strip()] = row[1].strip()

    sorted_labels = sorted(label_to_name.items())
    categories = [{"id": i, "name": name} for i, (_, name) in enumerate(sorted_labels, 1)]
    cat_id = {label: i for i, (label, _) in enumerate(sorted_labels, 1)}

    # 2. Scan on-disk images and read dimensions from JPEG headers (no full decode)
    logger.info(f"---> [OI-COCO] Scanning {data_dir} for images <---")
    out_images_dir.mkdir(parents=True, exist_ok=True)
    images = []
    img_id_map: dict = {}
    img_num = 1

    if os.path.isdir(data_dir):
        entry_names = os.listdir(data_dir)
        for entry_name in tqdm(entry_names, desc=f"[OI-COCO] {split} scan", unit="img"):
            if not entry_name.lower().endswith(".jpg"):
                continue
            stem = os.path.splitext(entry_name)[0]
            entry_path = os.path.join(data_dir, entry_name)
            try:
                with PILImage.open(entry_path) as img:
                    w, h = img.size
            except Exception:
                continue
            images.append({"id": img_num, "file_name": entry_name, "width": w, "height": h})
            img_id_map[stem] = (img_num, w, h)
            link = out_images_dir / entry_name
            if not link.exists():
                os.symlink(os.path.abspath(entry_path), str(link))
            img_num += 1

    logger.info(f"---> [OI-COCO] {split}: {len(images)} images found <---")

    # 3. Parse detections CSV and convert normalised coords to absolute pixels
    det_csv = os.path.join(labels_dir, "detections", f"{split}-annotations-bbox.csv")
    annotations = []
    ann_id = 1

    if os.path.exists(det_csv) and images:
        with open(det_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"[OI-COCO] {split} bbox", unit="row"):
                iid = row["ImageID"]
                if iid not in img_id_map:
                    continue
                label = row["LabelName"]
                if label not in cat_id:
                    continue
                im_num, w, h = img_id_map[iid]
                xmin = float(row["XMin"]) * w
                ymin = float(row["YMin"]) * h
                bw   = (float(row["XMax"]) - float(row["XMin"])) * w
                bh   = (float(row["YMax"]) - float(row["YMin"])) * h
                annotations.append({
                    "id": ann_id,
                    "image_id": im_num,
                    "category_id": cat_id[label],
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": int(row.get("IsGroupOf", 0)),
                })
                ann_id += 1

    logger.info(f"---> [OI-COCO] {split}: {len(annotations)} annotations <---")
    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(out_json, "w") as f:
        json.dump(coco, f)
    logger.debug(f"---> [OI-COCO] Written {out_json} <---")


def lvis_to_coco_det(lvis_json: str,
                     src_image_dirs: list,
                     out_images_dir: Path,
                     out_json: str):
    """
    Convert LVIS annotations to COCO detection format.

    LVIS-v1's val set contains some rare-class images sourced from COCO train2017
    (not val2017), so the actual file may live in either split's directory. For each
    image, the function searches `src_image_dirs` in order, skips images with no
    on-disk file (and their dangling annotations), and symlinks resolved images
    into `out_images_dir` so the merged dataset has a single image root.

    :param lvis_json: path to LVIS annotations JSON
    :param src_image_dirs: directories (in priority order) to search for each image file
    :param out_images_dir: directory where image symlinks will be created
    :param out_json: path to save COCO annotations JSON

    :return: None
    """
    lvis = json.load(open(lvis_json))
    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # LVIS images lack file_name; derive it from coco_url or zero-padded id
    images = []
    kept_image_ids = set()
    skipped_missing = 0
    for im in tqdm(lvis["images"], desc=f"[LVIS] {os.path.basename(lvis_json)} images", unit="img"):
        url = im.get("coco_url", "")
        file_name = os.path.basename(url) if url else f"{im['id']:012d}.jpg"

        # Locate the actual image file on disk (LVIS-v1 val pulls some files from train2017)
        src_path = None
        for d in src_image_dirs:
            candidate = os.path.join(d, file_name)
            if os.path.isfile(candidate):
                src_path = candidate
                break
        if src_path is None:
            skipped_missing += 1
            continue

        # Symlink into the merged image dir so all merged samples share one root
        link = out_images_dir / file_name
        if not (link.is_symlink() or link.exists()):
            try:
                os.symlink(os.path.abspath(src_path), str(link))
            except FileExistsError:
                pass

        images.append({
            "id": im["id"],
            "file_name": file_name,
            "height": im.get("height"),
            "width": im.get("width"),
        })
        kept_image_ids.add(im["id"])

    if skipped_missing:
        logger.warning(f"---> [LVIS] {os.path.basename(lvis_json)}: skipped "
                       f"{skipped_missing} images with no source file on disk <---")

    categories = lvis["categories"]
    anns = []
    for a in tqdm(lvis["annotations"], desc=f"[LVIS] {os.path.basename(lvis_json)} anns", unit="ann"):
        if a["image_id"] not in kept_image_ids:
            continue
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
    logger.info(f"---> [LVIS] {os.path.basename(lvis_json)}: kept "
                f"{len(images)} images, {len(anns)} annotations <---")

def verify_merged_against_roots(merged_json: str, image_roots: list, split: str):
    """
    Walk every image in `merged_json` and confirm it resolves under one of `image_roots`.
    Logs a warning with a sample of unresolved files. Returns the unresolved count.

    :param merged_json: path to merged COCO JSON
    :param image_roots: directories the dataset will search at training time
    :param split: human label for logs

    :return: number of images that did not resolve
    """
    js = json.load(open(merged_json))
    missing = []
    for im in js["images"]:
        fn = im["file_name"]
        if not any(os.path.isfile(os.path.join(r, fn)) for r in image_roots):
            missing.append(fn)
    if missing:
        sample = ", ".join(missing[:5])
        logger.error(f"---> [VERIFY] {split}: {len(missing)} of {len(js['images'])} "
                     f"images do not resolve under {image_roots}. Sample: {sample} <---")
    else:
        logger.info(f"---> [VERIFY] {split}: all {len(js['images'])} images resolve <---")
    return len(missing)


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
        for c in tqdm(js["categories"], desc=f"[MERGE] cats {os.path.basename(jp)}", unit="cat"):
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
        for im in tqdm(js["images"], desc=f"[MERGE] imgs {os.path.basename(jp)}", unit="img"):
            new_id = next_img_id; next_img_id += 1
            img_id_map[im["id"]] = new_id
            merged["images"].append({
                "id": new_id,
                "file_name": im["file_name"],
                "height": im.get("height"),
                "width": im.get("width"),
            })

        for an in tqdm(js.get("annotations", []), desc=f"[MERGE] anns {os.path.basename(jp)}", unit="ann"):
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
                oi_root = os.path.join(base_path, "open-images-v7")

                coco_merged_train = os.path.join(par_path, str(Config.TRAIN_IMG_DIRS[1]))
                Path(coco_merged_train).mkdir(parents=True, exist_ok=True) # 1 is for Merged COCO Train Anno

                coco_merged_val = os.path.join(par_path, str(Config.VAL_IMG_DIRS[1]))
                Path(coco_merged_val).mkdir(parents=True, exist_ok=True)   # 1 is for Merged COCO Val Anno

                export_openimages_to_coco("train",
                                          oi_root,
                                          Path(coco_merged_train),
                                          os.path.join(coco_out, "instances_oi_train.json"))

                export_openimages_to_coco("validation",
                                          oi_root,
                                          Path(coco_merged_val),
                                          os.path.join(coco_out, "instances_oi_val.json"))

                # B. LVIS (bbox-only) to COCO
                #    LVIS-v1 references COCO 2017 images. Most lvis_train files live in
                #    train2017/, most lvis_val in val2017/, but a handful of rare-class
                #    val images live in train2017/ — search both directories per image.
                coco_train_dir = os.path.join(par_path, str(Config.TRAIN_IMG_DIRS[0]))
                coco_val_dir   = os.path.join(par_path, str(Config.VAL_IMG_DIRS[0]))

                lvis_train = annotation_path + "/" + "lvis" + "/" + "lvis_v1_train.json"
                lvis_val = annotation_path + "/" + "lvis" + "/" + "lvis_v1_val.json"
                lvis_to_coco_det(lvis_train,
                                 [coco_train_dir, coco_val_dir],
                                 Path(coco_merged_train),
                                 os.path.join(coco_out, "instances_lvis_train.json"))
                lvis_to_coco_det(lvis_val,
                                 [coco_val_dir, coco_train_dir],
                                 Path(coco_merged_val),
                                 os.path.join(coco_out, "instances_lvis_val.json"))

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

                # D. Verify every merged image resolves under the configured roots
                train_roots = [os.path.join(par_path, str(d)) for d in Config.TRAIN_IMG_DIRS]
                val_roots   = [os.path.join(par_path, str(d)) for d in Config.VAL_IMG_DIRS]
                verify_merged_against_roots(
                    os.path.join(coco_out, "instances_train_merged.json"),
                    train_roots, "train")
                verify_merged_against_roots(
                    os.path.join(coco_out, "instances_val_merged.json"),
                    val_roots, "val")

                end_time = time.perf_counter()
                logger.debug(f"---> DATASETS CONVERSION/EXPORT TO COCO JSON AND "
                             f"MERGE DONE IN TIME {(end_time - start_time) / (60 * 60): .3f} HOURS <---")

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
