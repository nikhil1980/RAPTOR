import os, json, zipfile, urllib.request
import time
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config
""" User Modules """

"""
Basic Script to download and prepare datasets.

@author: Nikhil Bhargava
@date: 2025-10-15
@license: Apache-2.0
@description: This script downloads datasets from specified URLs and prepares them for use.
@copyright: Copyright 2025 Nikhil Bhargava
"""

logger = get_logger(__name__)


# ---------- LVIS (uses COCO-2017 images) ----------
def download_coco_2017_images(dst: Path):
    # COCO official zips
    coco_urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017":   "http://images.cocodataset.org/zips/val2017.zip",
    }
    for split, url in coco_urls.items():
        zpath = dst/f"{split}.zip"
        if not zpath.exists():
            print(f"[COCO] downloading {split} ...")
            urllib.request.urlretrieve(url, zpath)
        outdir = dst/split
        outdir.mkdir(exist_ok=True)
        if not any(outdir.iterdir()):
            logger.debug(f"[COCO] extracting {zpath} ...")
            with zipfile.ZipFile(zpath,"r") as zf: zf.extractall(dst)

def download_lvis_annotations(dst: Path):
    """
    Download LVIS annotations (v1) from official site.

    :param dst: destination directory to save annotations
    :return: None
    """
    # From official LVIS site / fbaipublicfiles mirrors (v1)
    files = {
        "lvis_v1_train.json": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json",
        "lvis_v1_val.json":   "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json",
    }
    (dst/ "lvis").mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        out = dst/"lvis"/name
        if not out.exists():
            logger.debug(f"[LVIS] downloading {name} ...")
            urllib.request.urlretrieve(url, out)

# ---------- OpenImages V7 via FiftyOne ----------
def download_openimages(split="train", label_types=("detections",), max_samples=None, classes=None):
    """
    Creates a FiftyOne dataset on disk under ~/.fiftyone and caches media.
    We'll export to COCO later into datasets/mixture/.
    """
    logger.debug(f"---> [OpenImages] Loading {split} with label_types={label_types}, max_samples={max_samples}, classes={classes}" <---)
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=list(label_types),
        classes=classes,            # None = all 600 "boxable" classes
        max_samples=max_samples,    # keep this small at first!
        shuffle=True,
        seed=42,
        dataset_name=f"openimages-v7-{split}-det"
    )
    return ds


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
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_DOWNLOAD_DATA")}] "
                            f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

                logger.debug(f"---> Configuration loaded from: {args.config_file} <---")

                # 2. Download and prepare datasets as per config option
                base_path = Path(os.path.join(par_path, Config.ROOT))
                base_path.mkdir(parents=True, exist_ok=True)

                image_path = os.path.join(base_path, "images")
                Path(image_path).mkdir(parents=True, exist_ok=True)

                annotation_path = os.path.join(base_path, "annotations")
                Path(annotation_path).mkdir(parents=True, exist_ok=True)

                start_time = time.perf_counter()

                # 1) LVIS annotation over COCO images (I use 2017 set)
                download_coco_2017_images(Path(image_path))  # makes images/train2017, images/val2017
                download_lvis_annotations(Path(annotation_path))  # puts annotations/lvis/*.json
                end_time = time.perf_counter()

                logger.debug(f"---> LVIS DOWNLOADED IN TIME: {(end_time - start_time)/ 60 * 60: .3f} HOURS <---")

                # 2) OpenImages — start with a small sample size to test the pipeline
                #    move to (None) for full dataset, or set 'classes=[...]' to focus on categories
                for split in ("train", "validation"):
                    download_openimages(split=split, label_types=("detections",), max_samples=5000)

                end_time = time.perf_counter()
                logger.debug(f"---> OPENIMAGESv7 DOWNLOADED IN TIME: {(end_time - start_time) / 60 * 60: .3f} HOURS <---")

                logger.debug("---> DATA DOWNLOADED. CONVERT/EXPORT TO COCO JSON AND MERGE <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")
            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_DOWNLOAD_DATA")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---")

if __name__ == '__main__':
    main()