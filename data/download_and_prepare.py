import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json, zipfile, urllib.request
import time
from pathlib import Path

import logging.config, os

os.environ["FIFTYONE_LOGGING_LEVEL"] = "WARNING"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,   # don't nuke everything globally
    "handlers": {
        "null": {"class": "logging.NullHandler"},
        "stderr-warning": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
        },
    },
    "root": {"handlers": ["stderr-warning"], "level": "WARNING"},
    "loggers": {
        # AWS stack
        "botocore":           {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.loaders":   {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.hooks":     {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.endpoint":  {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.utils":     {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.client":    {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "boto3":              {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "s3transfer":         {"handlers": ["null"], "level": "ERROR", "propagate": False},
        # HTTP + FS
        "urllib3":            {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "fsspec":             {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "s3fs":               {"handlers": ["null"], "level": "ERROR", "propagate": False},
        # Mongo
        "pymongo":            {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.serverSelection": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.connection": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.command":    {"handlers": ["null"], "level": "ERROR", "propagate": False},
        # FiftyOne
        "fiftyone":           {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "fiftyone.core":      {"handlers": ["null"], "level": "ERROR", "propagate": False},
    },
}
logging.config.dictConfig(LOGGING)

import fiftyone as fo
import fiftyone.zoo as foz
from requests.adapters import HTTPAdapter, Retry

import warnings; warnings.filterwarnings("ignore")

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


def download_progress_hook(blocknum, blocksize, totalsize):
    """
    Reporthook for urllib.request.urlretrieve to display download progress.
    
    :param blocknum: number of blocks transferred so far
    :param blocksize: size of each block (in bytes)
    :param totalsize: total size of the file (in bytes)
    
    :return: None
    """
    read_data = blocknum * blocksize
    if totalsize > 0:
        percentage = min(100, int(read_data * 100 / totalsize))
        print(f"\rDownloading: {percentage}%", end="")
    else:
        print(f"\rDownloading: {read_data} bytes", end="")
        
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
            logger.debug(f"---> [COCO] downloading {split} ...<---")
            urllib.request.urlretrieve(url, zpath, reporthook=download_progress_hook)
        outdir = dst/split
        outdir.mkdir(exist_ok=True)
        if not any(outdir.iterdir()):
            logger.debug(f"--> [COCO] extracting {zpath} ... <---")
            with zipfile.ZipFile(zpath,"r") as zf: zf.extractall(dst)
        os.remove(zpath)  # clean up zip file
        
    logger.debug(f"---> [COCO] 2017 images ready at {dst} <---")
    

def _http_session():
    s = requests.Session()
    # retry on common transient errors + 403/429
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    # Some CDNs/S3 endpoints return 403 without a UA
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    return s

def _stream_download(url: str, out_path: Path, chunk=1 << 20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _http_session().get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        with open(out_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    downloaded += len(part)
                    if total:
                        pct = 100.0 * downloaded / total
                        print(f"\r[DL] {out_path.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="")
        print()

def _unzip_if_needed(zpath: Path):
    target = zpath.with_suffix("")  # .zip -> no suffix (the .json)
    if target.exists():
        return target
    print(f"[UNZIP] {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(zpath.parent)
    return target

def download_lvis_annotations(dst_dir: Path):
    """
    Downloads LVIS v1 annotations (zipped JSONs) with retries, UA header,
    and mirrors. Places files under: <dst_dir>/lvis/{lvis_v1_train.json,lvis_v1_val.json}
    """
    (dst_dir / "lvis").mkdir(parents=True, exist_ok=True)

    # Primary (S3) endpoints — use these first
    primary = {
        "lvis_v1_train.json.zip": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_val.json.zip":   "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    }

    # Fallback (CDN) — sometimes OK, sometimes 403 behind corporate egress
    fallback = {
        "lvis_v1_train.json.zip": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_val.json.zip":   "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    }

    # Optional: a third mirror (community references occasionally use it)
    # If your network blocks both primary & fallback, comment these out and use Kaggle/HF manually.
    fallback2 = {
        "lvis_v1_train.json.zip": "https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_val.json.zip":   "https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    }

    for name in ["lvis_v1_train.json.zip", "lvis_v1_val.json.zip"]:
        zpath = dst_dir / "lvis" / name
        json_out = zpath.with_suffix("")
        if json_out.exists():
            print(f"[SKIP] {json_out.name} exists")
            continue

        # try primary -> fallback -> fallback2
        tried = []
        for urlmap in (primary, fallback, fallback2):
            url = urlmap[name]
            try:
                print(f"[DL] {url}")
                _stream_download(url, zpath)
                break
            except Exception as e:
                tried.append((url, str(e)))
                print(f"[WARN] Failed: {url} ({e})")
                continue
        else:
            # All attempts failed
            raise RuntimeError(
                "LVIS download failed.\n"
                + "\n".join([f"- {u} -> {err}" for u, err in tried])
                + "\nIf you are behind a corporate proxy, set HTTPS_PROXY and try again, "
                  "or download the zips manually and place them in <dst>/lvis/."
            )

        _unzip_if_needed(zpath)
        os.remove(zpath)  # clean up zip file

    print("[OK] LVIS v1 annotations ready.")


# ---------- OpenImages V7 via FiftyOne ----------
def download_openimages(split="train", 
                        label_types=("detections",), 
                        max_samples=None, 
                        classes=None)-> fo.Dataset:
    """
    Creates a FiftyOne dataset on disk under ~/.fiftyone and caches media.
    We'll export to COCO later into datasets/mixture/.
    """
    logger.debug(f"---> [OpenImages] Loading {split} with label_types={label_types}, max_samples={max_samples}, classes={classes} <---")
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
                logger.info(f"---> Configuration loaded from: {args.config_file} <---")

                Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT", str("datasets/mixture"))
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv('RAPTOR_WORKFLOW_DOWNLOAD_DATA')}] "
                            f"FOR [MODULE: {os.getenv('RAPTOR_PROJECT_NAME')}] <---")

          
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

                logger.debug(f"---> COCO + LVIS DOWNLOADED IN TIME: {(end_time - start_time)/ (60 * 60): .3f} HOURS <---")

                # 2) OpenImages — start with a small sample size to test the pipeline
                #    move to (None) for full dataset, or set 'classes=[...]' to focus on categories
                for split in ("train", "validation"):
                    download_openimages(split=split, label_types=("detections",), max_samples=None)

                end_time = time.perf_counter()
                logger.debug(f"---> OPENIMAGESv7 DOWNLOADED IN TIME: {(end_time - start_time) / (60 * 60): .3f} HOURS <---")

                logger.debug("---> DATA DOWNLOADED. CONVERT/EXPORT TO COCO JSON AND MERGE <---")

            except json.JSONDecodeError:
                logger.exception(f"---> ERROR: INVALID JSON in {args.config_file} <---")
            except Exception as e:
                logger.exception(f"---> ERROR READING CONFIGURATION: {e} <---")
        else:
            logger.info(f"---> CONFIGURATION FILE: '{args.config_file}' NOT FOUND AT PATH: {config_file_path}")
    else:
        logger.info(f"---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")

    logger.info(f"""---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_DOWNLOAD_DATA")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---""")

if __name__ == '__main__':
    main()