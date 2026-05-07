import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import tempfile, json, shutil, zipfile, urllib.request, tarfile
import pandas as pd
import csv
from typing import Dict, Optional, Set, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import requests
import time
from pathlib import Path
from tqdm import tqdm
import logging.config

os.environ["FIFTYONE_LOGGING_LEVEL"] = "WARNING"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "null": {"class": "logging.NullHandler"},
        "stderr-warning": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
        },
    },
    "root": {"handlers": ["stderr-warning"], "level": "WARNING"},
    "loggers": {
        "botocore":           {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.loaders":   {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.hooks":     {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.endpoint":  {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.utils":     {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "botocore.client":    {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "boto3":              {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "s3transfer":         {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "urllib3":            {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "fsspec":             {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "s3fs":               {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo":            {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.serverSelection": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.connection": {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "pymongo.command":    {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "fiftyone":           {"handlers": ["null"], "level": "ERROR", "propagate": False},
        "fiftyone.core":      {"handlers": ["null"], "level": "ERROR", "propagate": False},
    },
}
logging.config.dictConfig(LOGGING)

os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://127.0.0.1:27100/fiftyone"
os.environ.pop("FIFTYONE_DATABASE_DIR", None)

import fiftyone as fo
import fiftyone.utils.openimages as fouo
from fiftyone.core.odm.database import get_db_client

from requests.adapters import HTTPAdapter, Retry

import warnings; warnings.filterwarnings("ignore")

""" System Modules """
from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config


"""
@author: Nikhil Bhargava
@date: October 31, 2025
@description: This module contains functions to download and prepare datasets such as COCO, LVIS, and OpenImages V7.
@copyright: Nikhil Bhargava

To run download_and_prepare.py
-----------------------------
# 1) Choose a place for Mongo’s data/logs
export FO_MONGO_DIR=/raid/home/nikhilb/fo_mongo
mkdir -p "$FO_MONGO_DIR"

# 2) Stop any stray mongod and stale locks
pkill -f mongod || true
rm -f "$FO_MONGO_DIR/mongod.lock"

# 3) Give the process enough file descriptors (important)
ulimit -n 64000

# 4) Start mongod yourself on a fixed port (27100), in the background
#    Resolve the bundled mongod from whichever env (conda, venv, pyenv) is active:
MONGOD_BIN="$(python -c "import os, fiftyone; print(os.path.join(os.path.dirname(fiftyone.__file__), 'db', 'bin', 'mongod'))")"
"$MONGOD_BIN" \
  --dbpath "$FO_MONGO_DIR" \
  --logpath "$FO_MONGO_DIR/mongo.log" \
  --bind_ip 127.0.0.1 \
  --port 27100 \
  --nounixsocket \
  --wiredTigerCacheSizeGB 4 \
  --fork

python data/download_and_prepare.py --config-file config.json > main_log.log 2>&1 &
tail -f main_log.log

"""


logger = get_logger(__name__)


#------ Utility Functions ------
def download_progress_hook(blocknum, blocksize, totalsize):
    """
    Report hook for urllib.request.urlretrieve to show download progress.
    :param blocknum: Number of blocks transferred so far
    :param blocksize: Size of each block (in bytes)
    :param totalsize: Total size of the file (in bytes). May be -1 if unknown.  
    
    :return: None
    """
    read_data = blocknum * blocksize
    if totalsize > 0:
        percentage = min(100, int(read_data * 100 / totalsize))
        print(f"\rDownloading: {percentage}%", end="")
    else:
        print(f"\rDownloading: {read_data} bytes", end="")


# ---------- COCO ----------
def _download_with_aria2(url: str, out_path: Path) -> bool:
    if not shutil.which("aria2c"):
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--max-tries=0",           # retry forever
        "--retry-wait=5",
        "--timeout=60",
        "--dir", str(out_path.parent),
        "--out", out_path.name,
        url,
    ]
    print(f"[aria2c] {url}")
    ret = subprocess.run(cmd).returncode
    return ret == 0


def download_coco_2017_images(dst: Path):
    """
    Downloads and extracts COCO 2017 train and val images to the specified directory.
    :param dst: Path to the directory where images will be downloaded and extracted.
    :return: None
    """
    coco_urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017":   "http://images.cocodataset.org/zips/val2017.zip",
    }
    for split, url in coco_urls.items():
        zpath = dst / f"{split}.zip"
        outdir = dst / split
        outdir.mkdir(exist_ok=True)
        if any(outdir.iterdir()):
            logger.debug(f"---> [COCO] {split} already extracted, skipping <---")
            if zpath.exists():
                os.remove(zpath)
            continue
        logger.debug(f"---> [COCO] downloading {split} ...<---")
        if not _download_with_aria2(url, zpath):
            # fallback: resumable Python download
            _stream_download(url, zpath)
        logger.debug(f"--> [COCO] extracting {zpath} ... <---")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dst)
        os.remove(zpath)
    logger.debug(f"---> [COCO] 2017 images ready at {dst} <---")


def _http_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    return s


def _stream_download(url: str, out_path: Path, chunk=1 << 20, max_retries=20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        offset = out_path.stat().st_size if out_path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        try:
            with _http_session().get(url, stream=True, timeout=(30, 300), headers=headers) as r:
                if r.status_code == 416:
                    print(f"[DL] {out_path.name}: already complete")
                    return
                r.raise_for_status()
                total_remaining = int(r.headers.get("Content-Length", "0") or "0")
                total = offset + total_remaining
                downloaded = offset
                mode = "ab" if offset > 0 else "wb"
                with open(out_path, mode) as f:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
                            f.flush()
                            downloaded += len(part)
                            if total:
                                pct = 100.0 * downloaded / total
                                print(f"\r[DL] {out_path.name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
            return
        except Exception as e:
            wait = min(2 ** attempt, 120)
            print(f"\n[RETRY {attempt+1}/{max_retries}] {out_path.name} @ {offset/1e6:.0f} MB — {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries")



def _unzip_if_needed(zpath: Path):
    """
    Unzips a zip file if the target file does not already exist.
    
    
    :param zpath: Path to the zip file.
    :return: Path to the unzipped target file.
    """

    target = zpath.with_suffix("")
    if target.exists():
        return target
    print(f"[UNZIP] {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(zpath.parent)
    return target

def download_lvis_annotations(dst_dir: Path):
    """
    Download LVIS v1 annotations to the specified directory.
    
    :param dst_dir: Path to the directory where annotations will be downloaded.
    :return: None
    
    """
    (dst_dir / "lvis").mkdir(parents=True, exist_ok=True)
    primary = {
        "lvis_v1_train.json.zip": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_val.json.zip":   "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    }
    fallback = {
        "lvis_v1_train.json.zip": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
        "lvis_v1_val.json.zip":   "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
    }
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
            raise RuntimeError(
                "LVIS download failed.\n"
                + "\n".join([f"- {u} -> {err}" for u, err in tried])
            )
        _unzip_if_needed(zpath)
        os.remove(zpath)
    print("[OK] LVIS v1 annotations ready.")


# -----------------------------------
# ---------- OpenImages V7 ----------
# -----------------------------------

OI_URLS: Dict[str, str] = {
    "train_ids":       "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation_ids":  "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test_ids":        "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    "train_boxes":     "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
    "validation_boxes":"https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv",
    "test_boxes":      "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv",
    "classes_boxable": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
}

def _ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists.
    :param path: The directory path to ensure.
    :type path: str
    
    :return: None
    """
    os.makedirs(path, exist_ok=True)
    

def _download(url: str, dst: str) -> None:
    """
    Downloads a file from a URL to a specified destination with resume support.
    
    :param url: The URL to download from.
    :type url: str
    :param dst: The destination file path.
    :type dst: str
    
    :return: None
    """
    _ensure_dir(os.path.dirname(dst))
    headers = {}
    mode = "wb"
    if os.path.exists(dst):
        cur = os.path.getsize(dst)
        headers["Range"] = f"bytes={cur}-"
        mode = "ab"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(dst, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)



def _ensure_split_ids_csv(root: str, split: str) -> str:
    """
    Downloads official split ImageID list with rotation using the exact
    filenames the V7 ecosystem expects, e.g.:
      metadata/validation/validation-images-with-rotation.csv
    Returns the full path to that file.
    """
    meta_dir = os.path.join(root, "metadata", split)
    _ensure_dir(meta_dir)
    # use official name (not "image_ids.csv")
    out_csv = os.path.join(meta_dir, f"{split}-images-with-rotation.csv")
    if not os.path.exists(out_csv):
        key = f"{split}_ids"
        if key not in OI_URLS:
            raise ValueError(f"Unsupported split '{split}'")
        logger.info(f"[Meta] Downloading image IDs for {split} → {out_csv}")
        _download(OI_URLS[key], out_csv)
    return out_csv

def _read_ids(csv_path: str) -> Set[str]:
    """
    Read ImageIDs from a CSV file.
    
    :param csv_path: Path to the CSV file.
    :param csv_path: str
    
    :return: A set of ImageIDs.
    :rtype: Set[str]    
    """
    ids = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        idx = 0
        if header and "ImageID" in header:
            idx = header.index("ImageID")
        else:
            if header and len(header) > 0 and len(header[0]) == 16:
                ids.add(header[0])
        for row in reader:
            if not row:
                continue
            ids.add(row[idx])
    return ids



def _cleanup_bad_media(split: str, dirpath: str) -> int:
    """
    Cleanup zero-byte files in the specified directory.
    
    :param split: The data split (e.g., 'train', 'validation', 'test').
    :type split: str
    :param dirpath: The directory path to clean up.
    :type dirpath: str
    
    :return: The number of files removed.
    :rtype: int
    """
    touched = 0
    if not os.path.isdir(dirpath):
        return 0
    for entry in tqdm(os.scandir(dirpath), desc=f"Cleanup: {split} ", unit="img"):
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_size == 0:
                os.remove(entry.path)
                touched += 1
        except Exception:
            pass
    return touched






def _resolve_oi_dataset_type():
    """
    Function to resolve the appropriate OpenImages dataset type from FiftyOne.
    
    :return: The OpenImages dataset type class.
    :rtype: type
    """
    try:
        from fiftyone.types.dataset_types import OpenImagesV7Dataset as OIType
        return OIType
    except Exception:
        from fiftyone.types.dataset_types import OpenImagesDataset as OIType
        return OIType






def _find_missing_ids(split: str, dir_split: str, idset: Set[str]) -> List[str]:
    """
    Recursively scan dir_split and treat any *.jpg|*.jpeg|*.png as present
    regardless of accidental nesting from tar extraction.
    
    :param split: The data split (e.g., 'train', 'validation', 'test').
    :type split: str
    :param dir_split: The directory path for the split.
    :type dir_split: str
    :param idset: A set of ImageIDs for the specified split.
    :type idset: Set[str]
    
    :return: A sorted list of missing ImageIDs.
    :rtype: List[str]
    """
    have = set()
    if os.path.isdir(dir_split):
        for root_dir, _, files in os.walk(dir_split):
            for n in files:
                stem, ext = os.path.splitext(n)
                if ext.lower() in (".jpg", ".jpeg", ".png"):
                    have.add(stem)
    return sorted(list(idset - have))



def _http_download_one(url_tmpl: str, 
                       dir_split: str, 
                       image_id: str, 
                       timeout=40
                       ) -> bool:
    """
    Function to download a single image via HTTP.
    
    :param url_tmpl: The URL template with a placeholder for the image ID.
    :type url_tmpl: str
    
    :param dir_split: The directory to save the downloaded image.
    :type dir_split: str
    
    :param image_id: The image ID to download.
    :type image_id: str
    
    :param timeout: The timeout for the HTTP request.
    :type timeout: int
    
    :return: True if the image was downloaded, False otherwise.
    :rtype: bool
    """
    dst = os.path.join(dir_split, f"{image_id}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return False
    url = url_tmpl.format(id=image_id)
    os.makedirs(dir_split, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == 0:
                os.remove(dst)
        except Exception:
            pass
        return False
    
    
    
# Official OpenImages image URL templates
OI_IMG_URL = {
    "train":      "https://storage.googleapis.com/openimages/2018_04/train/{id}.jpg",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/{id}.jpg",
    "test":       "https://storage.googleapis.com/openimages/2018_04/test/{id}.jpg",
}


# ---------- AWS CVDF tar helpers for OPENIMAGES v7 ----------
S3_TARS = {
    "validation": "s3://open-images-dataset/tar/validation.tar.gz",
    "test":       "s3://open-images-dataset/tar/test.tar.gz",
}

# Train is sharded into 16 tarballs, one per first hex digit of the image ID.
# Each ~10-15 GB compressed; ~108 K images per shard.
OI_TRAIN_TARS = [
    f"s3://open-images-dataset/tar/train_{d}.tar.gz"
    for d in "0123456789abcdef"
]


def _download_missing_images_manual(split: str, 
                                    dir_split: str, 
                                    missing_ids: List[str], 
                                    workers: int = 32
                                    ) -> int:
    """
    Download missing images manually via HTTP.
    
    :param split: The data split (e.g., 'train', 'validation', 'test').
    :type split: str
    :param dir_split: The directory path for the split.
    :type dir_split: str
    :param missing_ids: A list of missing ImageIDs to download.
    :type missing_ids: List[str]
    :param workers: The number of concurrent download workers.
    :type workers: int
    
    :return: The number of images downloaded.
    :rtype: int
    """
    url_tmpl = OI_IMG_URL[split]
    fetch = partial(_http_download_one, url_tmpl, dir_split)
    done = 0
    if not missing_ids:
        return 0
    logger.info(f"---> [IMAGES] Missing count for {split}: {len(missing_ids)} — downloading only those <---")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, iid) for iid in missing_ids]
        for fut in as_completed(futs):
            if fut.result():
                done += 1
    return done






def _run(cmd, 
         cwd=None, 
         check=True
         ) -> str:
    """
    Run a command as a subprocess and capture its output.
    Used for awscli commands.
    
    :param cmd: The command to run as a list of arguments.
    :type cmd: List[str]
    :param cwd: The working directory to run the command in.
    :type cwd: Optional[str]
    :param check: Whether to raise an error if the command fails.
    :type check: bool
    
    :return: The standard output of the command.
    :rtype: str
    """
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout



def _has_awscli() -> bool:
    """
    Check if the AWS CLI is installed.
    
    :return: True if AWS CLI is installed, False otherwise.
    :rtype: bool
    """
    return shutil.which("aws") is not None



def _tar_extract_strip1(tar_path: str, dst_dir: str):
    """
    Extract tar.gz while stripping the first path component so files in
    'validation/...jpg' land directly under dst_dir, not dst_dir/validation/.
    """
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            m.name = parts[1] if len(parts) == 2 else parts[0]
            if m.name and not m.name.endswith("/"):
                members.append(m)
        tf.extractall(dst_dir, members=members)
        
        
def _fetch_split_tar_to(split: str, split_dir: str):
    """
    Download split tar.gz from CVDF (no auth) and extract into data/{split}/
    with the top-level 'split/' folder stripped.
    Uses aria2c (16 connections) when available, then curl, then aws-cli.
    """
    url = S3_TARS[split]
    https_url = url.replace("s3://open-images-dataset/", "https://open-images-dataset.s3.amazonaws.com/")
    tmp_tar = os.path.join(tempfile.gettempdir(), f"openimages_{split}.tar.gz")
    os.makedirs(split_dir, exist_ok=True)

    if _has_awscli():
        _run(["aws", "s3", "--no-sign-request", "cp", url, tmp_tar])
    elif not _download_with_aria2(https_url, Path(tmp_tar)):
        _run(["curl", "-L", "-C", "-", "-o", tmp_tar, https_url])

    _tar_extract_strip1(tmp_tar, split_dir)
    os.remove(tmp_tar)


def _fetch_train_tars_parallel(split_dir: str, tar_workers: int = 4) -> None:
    """
    Download and extract all 16 OI train tarballs in parallel.
    Each shard covers images whose ID starts with one hex digit (0-f).
    Uses aria2c (16 connections/file) when available, otherwise curl.
    tar_workers controls how many tarballs are downloaded simultaneously.
    """
    def _one_shard(s3_url: str) -> None:
        fname  = os.path.basename(s3_url)              # train_0.tar.gz
        https  = s3_url.replace("s3://open-images-dataset/",
                                "https://open-images-dataset.s3.amazonaws.com/")
        tmp    = os.path.join(tempfile.gettempdir(), fname)
        digit  = fname[len("train_"):-len(".tar.gz")]  # "0" … "f"
        logger.info(f"---> [OI-TRAIN] Starting shard train_{digit} <---")
        try:
            ok = False

            # 1. Try aws-cli (shows --no-progress so output stays clean in logs)
            if _has_awscli():
                try:
                    _run(["aws", "s3", "--no-sign-request", "--no-progress",
                          "cp", s3_url, tmp])
                    ok = True
                except Exception as aws_err:
                    logger.warning(f"---> [OI-TRAIN] aws-cli failed for train_{digit} "
                                   f"({aws_err}), falling back to aria2c <---")

            # 2. Fallback: aria2c (log-friendly: no escape codes, 60 s summaries)
            if not ok and shutil.which("aria2c"):
                cmd = [
                    "aria2c",
                    "--continue=true",
                    "--max-connection-per-server=16",
                    "--split=16",
                    "--min-split-size=1M",
                    "--max-tries=5",
                    "--retry-wait=10",
                    "--timeout=120",
                    "--console-log-level=warn",
                    "--summary-interval=60",
                    "--dir", tempfile.gettempdir(),
                    "--out", fname,
                    https,
                ]
                ret = subprocess.run(cmd).returncode
                ok = (ret == 0)

            # 3. Last resort: curl with resume support
            if not ok:
                _run(["curl", "-L", "-C", "-", "-o", tmp, https])
                ok = True

            logger.info(f"---> [OI-TRAIN] Extracting shard train_{digit} <---")
            _tar_extract_strip1(tmp, split_dir)
            logger.info(f"---> [OI-TRAIN] Shard train_{digit} done <---")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    logger.info(f"---> [OI-TRAIN] Downloading {len(OI_TRAIN_TARS)} train shards "
                f"({tar_workers} parallel) via CVDF <---")
    failed = []
    with ThreadPoolExecutor(max_workers=tar_workers) as ex:
        futs = {ex.submit(_one_shard, u): u for u in OI_TRAIN_TARS}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                url = futs[fut]
                logger.error(f"---> [OI-TRAIN] FAILED {url}: {e} <---")
                failed.append(url)
    if failed:
        logger.warning(f"---> [OI-TRAIN] {len(failed)} shards failed — "
                       "remaining images will be fetched individually <---")
    

def _ensure_validation_labels(root: str):
    """
    Download official validation detections CSV + class map to expected paths:
      labels/detections/validation-annotations-bbox.csv
      labels/class-descriptions-boxable.csv
    """
    labels_dir = os.path.join(root, "labels")
    det_dir = os.path.join(labels_dir, "detections")
    _ensure_dir(det_dir)

    det_csv = os.path.join(det_dir, "validation-annotations-bbox.csv")
    if not os.path.exists(det_csv):
        logger.info(f"[Labels] Downloading {det_csv}")
        _download(OI_URLS["validation_boxes"], det_csv)

    cls_csv = os.path.join(labels_dir, "class-descriptions-boxable.csv")
    if not os.path.exists(cls_csv):
        logger.info(f"[Labels] Downloading {cls_csv}")
        _download(OI_URLS["classes_boxable"], cls_csv)


def _ensure_train_labels(root: str):
    """
    Download official train detections CSV + class map to expected paths:
      labels/detections/train-annotations-bbox.csv  (~2 GB)
      labels/class-descriptions-boxable.csv
    """
    labels_dir = os.path.join(root, "labels")
    det_dir = os.path.join(labels_dir, "detections")
    _ensure_dir(det_dir)

    det_csv = os.path.join(det_dir, "train-annotations-bbox.csv")
    if not os.path.exists(det_csv):
        logger.info("[Labels] Downloading train-annotations-bbox.csv (~2 GB) …")
        _download(OI_URLS["train_boxes"], det_csv)

    cls_csv = os.path.join(labels_dir, "class-descriptions-boxable.csv")
    if not os.path.exists(cls_csv):
        logger.info("[Labels] Downloading class-descriptions-boxable.csv …")
        _download(OI_URLS["classes_boxable"], cls_csv)
        
        


def _flatten_split_dir(split_dir: str, split: str) -> int:
    """
    If tar created data/<split>/<split>/..., move files up one level.
    """
    nested = os.path.join(split_dir, split)
    moved = 0
    if os.path.isdir(nested):
        for root_dir, _, files in os.walk(nested):
            rel = os.path.relpath(root_dir, nested)
            dst_root = os.path.join(split_dir, rel) if rel != "." else split_dir
            os.makedirs(dst_root, exist_ok=True)
            for fn in files:
                src = os.path.join(root_dir, fn)
                dst = os.path.join(dst_root, fn)
                if not os.path.exists(dst):
                    os.replace(src, dst)
                    moved += 1
        # cleanup dirs
        for root_dir, dirs, _ in os.walk(nested, topdown=False):
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root_dir, d))
                except OSError:
                    pass
        try:
            os.rmdir(nested)
        except OSError:
            pass
        logger.info(f"[Flatten] Moved {moved} files up from {nested}")
    return moved


# ---------- OpenImages V7 via FiftyOne ----------
def download_openimages(split="train",
                        label_types=("detections",),
                        max_samples=None,
                        classes=None,
                        download_dir: Optional[str] = None,
                        dataset_name: Optional[str] = None,
                        num_workers: Optional[int] = Config.NUM_WORKERS,
                        shuffle: bool = True,
                        seed: int = Config.SEED,
                        prefer_manual_images: bool = True) -> fo.Dataset:
    """
    'train' (you already have), 'validation', and 'test'.
    For validation/test: use CVDF tarballs for images, direct CSV URLs for labels (validation).
    Zero FiftyOne S3 downloads are used here to avoid 403/404.
    """
    root = download_dir or os.path.expanduser(os.path.join("~", "fiftyone", "open-images-v7"))
    data_root = os.path.join(root, "data")
    labels_dir = os.path.join(root, "labels")
    metadata_dir = os.path.join(root, "metadata")
    _ensure_dir(root); _ensure_dir(data_root); _ensure_dir(labels_dir); _ensure_dir(metadata_dir)

    if dataset_name is None:
        dataset_name = f"openimages-v7-{split}-det" if split != "test" else "openimages-v7-test-imgs"

    # 1) Official image IDs CSV for the split (importer-friendly filename)
    ids_csv = _ensure_split_ids_csv(root, split)
    idset   = _read_ids(ids_csv)
    if not idset:
        raise FileNotFoundError(f"Could not read any ImageIDs for split '{split}' from {ids_csv}")

    split_dir = os.path.join(data_root, split)
    _ensure_dir(split_dir)

    # 2) Fill images (CVDF tarballs) and fix nesting
    missing_ids = _find_missing_ids(split, split_dir, idset)
    logger.info(f"[{split}] Present={len(idset)-len(missing_ids)}  Missing={len(missing_ids)}")

    if split in ("validation", "test") and (missing_ids and prefer_manual_images):
        print(f"[{split}] Fetching CVDF tarball…")
        _fetch_split_tar_to(split, split_dir)
        _flatten_split_dir(split_dir, split)
        # Recount after tar
        missing_ids = _find_missing_ids(split, split_dir, idset)
        logger.info(f"[{split}] After tar: Present={len(idset)-len(missing_ids)}  Missing={len(missing_ids)}")

    if split == "train" and missing_ids and prefer_manual_images:
        if len(missing_ids) > 10_000:
            # Bulk: 16 CVDF tarballs in parallel (aria2c per shard when available)
            _fetch_train_tars_parallel(split_dir)
            _flatten_split_dir(split_dir, split)
            missing_ids = _find_missing_ids(split, split_dir, idset)
            logger.info(f"---> [OI-TRAIN] After tarballs: "
                        f"Present={len(idset)-len(missing_ids)}  Missing={len(missing_ids)} <---")
        # Pick up any images not covered by tarballs (or if tarballs failed)
        if missing_ids:
            logger.info(f"---> [OI-TRAIN] HTTP-fetching {len(missing_ids)} remaining images "
                        f"(32 workers) <---")
            _download_missing_images_manual("train", split_dir, missing_ids)
            missing_ids = _find_missing_ids(split, split_dir, idset)
            logger.info(f"---> [OI-TRAIN] After HTTP fetch: "
                        f"Present={len(idset)-len(missing_ids)}  Still missing={len(missing_ids)} <---")

    # final clean
    cleaned = _cleanup_bad_media(split, split_dir)
    if cleaned:
        logger.info(f"[{split}] Cleaned {cleaned} files under {split_dir}")

    # 3) Labels
    if split == "validation" and "detections" in set(label_types):
        _ensure_validation_labels(root)
    if split == "train" and "detections" in set(label_types):
        _ensure_train_labels(root)

    # 4) Load dataset
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    if split == "test":
        # images-only
        ds = fo.Dataset.from_dir(
            dataset_dir=os.path.join(root, "data", "test"),
            dataset_type=fo.types.ImageDirectory,
            name=dataset_name,
        )
    else:
        # Build image_ids file in 'split/<ID>' format
        ids_file = _ensure_prefixed_ids_file(root, split)
        OIType = _resolve_oi_dataset_type()
        logger.info(f"---> [Load] root='{root}', split='{split}', dataset='{dataset_name}' <---")
        ds = fo.Dataset.from_dir(
            dataset_dir=root,
            dataset_type=OIType,
            label_types=list(label_types),   # ('detections',) for validation
            classes=classes,
            image_ids=ids_file,              # lines like 'validation/<ImageID>'
            include_id=True,
            name=dataset_name,
        )

    ds.persistent = True
    logger.info(f"[OpenImages] Loaded: {ds.name} | split={split} | samples={len(ds)}")
    return ds


def _ensure_prefixed_ids_file(root: str, split: str) -> str:
    """
    Builds metadata/{split}/image_ids_prefixed.txt from:
      metadata/{split}/{split}-images-with-rotation.csv
    with lines like 'validation/<ImageID>'.
    """
    meta_dir = os.path.join(root, "metadata", split)
    os.makedirs(meta_dir, exist_ok=True)
    src_csv = os.path.join(meta_dir, f"{split}-images-with-rotation.csv")
    out_txt = os.path.join(meta_dir, "image_ids_prefixed.txt")

    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        return out_txt
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Missing {src_csv}")

    with open(src_csv, "r", newline="", encoding="utf-8") as f, open(out_txt, "w") as g:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return out_txt
        if "ImageID" in header:
            idx = header.index("ImageID")
        else:
            # headerless CSV: first column is the ID; first row was already consumed as header
            idx = 0
            if header[idx].strip():
                g.write(f"{split}/{header[idx].strip()}\n")
        for row in r:
            if not row:
                continue
            g.write(f"{split}/{row[idx].strip()}\n")
    return out_txt


# ---------- main ----------
def main():
    parser = build_myargparser()
    args = parser.parse_args()

    # Fail fast if MongoDB isn't reachable before doing any real work
    try:
        get_db_client().server_info()
        print("OK: FiftyOne is using external Mongo:", fo.config.database_uri)
    except Exception as e:
        print(f"[ERROR] Cannot reach MongoDB at {fo.config.database_uri}: {e}")
        print("Start mongod on port 27100 first — see the docstring at the top of this file.")
        sys.exit(1)

    if args.config_file:
        par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file_path = os.path.join(par_path, str(args.config_file))

        if os.path.exists(config_file_path):
            try:
                load_env_from_json(config_file_path)
                logger.info(f"---> Configuration loaded from: {args.config_file} <---")

                Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT", str("datasets/mixture"))
                logger.info(f"---> STARTING [WORKFLOW: {os.getenv('RAPTOR_WORKFLOW_DOWNLOAD_DATA')}] "
                            f"FOR [MODULE: {os.getenv('RAPTOR_PROJECT_NAME')}] <---")

                base_path = Path(os.path.join(par_path, Config.ROOT))
                base_path.mkdir(parents=True, exist_ok=True)

                image_path = os.path.join(base_path, "images")
                Path(image_path).mkdir(parents=True, exist_ok=True)

                annotation_path = os.path.join(base_path, "annotations")
                Path(annotation_path).mkdir(parents=True, exist_ok=True)

                start_time = time.perf_counter()

                download_coco_2017_images(Path(image_path))
                download_lvis_annotations(Path(annotation_path))

                download_dir = os.path.join(base_path, "open-images-v7")
                os.makedirs(download_dir, exist_ok=True)

                # 1. Train
                download_openimages(split="train", label_types=("detections",), max_samples=None, download_dir=download_dir, prefer_manual_images=True)
                
                # Validation (detections)
                download_openimages(
                    split="validation",
                    label_types=("detections",),
                    download_dir=download_dir,
                    prefer_manual_images=True,
                )

                # Test (images-only)
                download_openimages(
                    split="test",
                    label_types=(),              # no detections on test
                    download_dir=download_dir,
                    prefer_manual_images=True,
                )

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

    logger.info(f"---> FINISHED [WORKFLOW: {os.getenv('RAPTOR_WORKFLOW_DOWNLOAD_DATA')}] "
                f"FOR [MODULE: {os.getenv('RAPTOR_PROJECT_MODE')}] <---")

if __name__ == '__main__':
    main()
