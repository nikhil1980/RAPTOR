import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import textwrap, tempfile, json, shutil
import pandas as pd
import csv
from typing import Iterable, Dict, Optional, Set, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import requests
import json, zipfile, urllib.request, tarfile   # <-- added tarfile
import time
from pathlib import Path
from tqdm import tqdm
import logging.config, os

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

get_db_client().server_info()
print("OK: FiftyOne is using external Mongo:", fo.config.database_uri)

from requests.adapters import HTTPAdapter, Retry

import warnings; warnings.filterwarnings("ignore")

""" System Modules """
from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config

logger = get_logger(__name__)

def download_progress_hook(blocknum, blocksize, totalsize):
    read_data = blocknum * blocksize
    if totalsize > 0:
        percentage = min(100, int(read_data * 100 / totalsize))
        print(f"\rDownloading: {percentage}%", end="")
    else:
        print(f"\rDownloading: {read_data} bytes", end="")

# ---------- COCO/LVIS ----------
def download_coco_2017_images(dst: Path):
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
        os.remove(zpath)
    logger.debug(f"---> [COCO] 2017 images ready at {dst} <---")

def _http_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
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
    target = zpath.with_suffix("")
    if target.exists():
        return target
    print(f"[UNZIP] {zpath}")
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(zpath.parent)
    return target

def download_lvis_annotations(dst_dir: Path):
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

# ---------- OpenImages V7 ----------
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
    os.makedirs(path, exist_ok=True)

def _download(url: str, dst: str) -> None:
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
    meta_dir = os.path.join(root, "metadata", split)
    _ensure_dir(meta_dir)
    out_csv = os.path.join(meta_dir, "image_ids.csv")
    if not os.path.exists(out_csv):
        key = f"{split}_ids"
        if key not in OI_URLS:
            raise ValueError(f"Unsupported split '{split}'")
        logger.info(f"[Meta] Downloading image IDs for {split} → {out_csv}")
        _download(OI_URLS[key], out_csv)
    return out_csv

def _read_ids(csv_path: str) -> Set[str]:
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

def _rehydrate_images(root: str, split: str, idset: Set[str]) -> int:
    data_root = os.path.join(root, "data")
    split_dir = os.path.join(data_root, split)
    _ensure_dir(split_dir)
    moved = 0
    for entry in tqdm(os.scandir(data_root), desc=f"rehydrate:{split}", unit="img"):
        if not entry.is_file():
            continue
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if stem in idset:
            dst = os.path.join(split_dir, entry.name)
            if not os.path.exists(dst):
                os.replace(entry.path, dst)
                moved += 1
    logger.info(f"[Rehydrate] Moved {moved} file(s) into {split_dir}")
    return moved

def _split_detections_if_needed(root: str, split: str, idset: Set[str]) -> Optional[str]:
    labels_dir = os.path.join(root, "labels")
    det_all = os.path.join(labels_dir, "detections.csv")
    det_split = os.path.join(labels_dir, "detections", f"{split}-annotations-bbox.csv")
    if os.path.exists(det_split):
        return det_split
    if not os.path.exists(det_all):
        return None
    _ensure_dir(os.path.dirname(det_split))
    logger.info(f"---> [Labels] Creating split boxes for {split} from detections.csv → {det_split} <---")
    chunksize = 1_000_000
    wrote_header = False
    with open(det_split, "w", newline="") as out_f:
        for chunk in pd.read_csv(det_all, chunksize=chunksize):
            if "ImageID" not in chunk.columns:
                chunk.rename(columns={chunk.columns[0]: "ImageID"}, inplace=True)
            sub = chunk[chunk["ImageID"].isin(idset)]
            if sub.empty:
                continue
            if not wrote_header:
                sub.to_csv(out_f, index=False)
                wrote_header = True
            else:
                sub.to_csv(out_f, index=False, header=False)
    if os.path.getsize(det_split) == 0:
        logger.warning(f"---> [Labels] No detections found for {split} in detections.csv (file left empty) <---")
    return det_split

def _resolve_oi_dataset_type():
    try:
        from fiftyone.types.dataset_types import OpenImagesV7Dataset as OIType
        return OIType
    except Exception:
        from fiftyone.types.dataset_types import OpenImagesDataset as OIType
        return OIType

def _existing_image_path(dir_split: str, image_id: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(dir_split, image_id + ext)
        if os.path.exists(p):
            return p
    return None

def _find_missing_ids(split: str, dir_split: str, idset: Set[str]) -> List[str]:
    """
    Recursively scan dir_split (handles accidental data/<split>/<split>/ nesting)
    and treat any *.jpg|*.jpeg|*.png as present for the image_id stem.
    """
    have = set()
    if os.path.isdir(dir_split):
        for root, _, files in os.walk(dir_split):
            for n in files:
                stem, ext = os.path.splitext(n)
                if ext.lower() in (".jpg", ".jpeg", ".png"):
                    have.add(stem)
    missing = sorted(list(idset - have))
    return missing

def _http_download_one(url_tmpl: str, dir_split: str, image_id: str, timeout=40) -> bool:
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

OI_IMG_URL = {
    "train":      "https://storage.googleapis.com/openimages/2018_04/train/{id}.jpg",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/{id}.jpg",
    "test":       "https://storage.googleapis.com/openimages/2018_04/test/{id}.jpg",
}

def _download_missing_images_manual(split: str, dir_split: str, missing_ids: List[str], workers: int = 32) -> int:
    url_tmpl = OI_IMG_URL[split]
    fetch = partial(_http_download_one, url_tmpl, dir_split)
    done = 0
    if not missing_ids:
        return 0
    logger.info(f"---> [Images] Missing count for {split}: {len(missing_ids)} — downloading only those <---")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, iid) for iid in missing_ids]
        for fut in as_completed(futs):
            if fut.result():
                done += 1
    return done

def _download_missing_images_via_fiftyone(root: str, split: str, missing_ids: List[str], shuffle: bool, seed: int) -> None:
    fouo.download_open_images_split(
        dataset_dir=root,
        split=split,
        version="v7",
        label_types=[],
        classes=None,
        image_ids=missing_ids,
        num_workers=os.cpu_count(),
        shuffle=shuffle,
        seed=seed,
        max_samples=None,
    )

# ---------- CVDF tar (validation/test) helpers ----------
S3_TARS = {
    "validation": "s3://open-images-dataset/tar/validation.tar.gz",
    "test":       "s3://open-images-dataset/tar/test.tar.gz",
}

def _run(cmd, cwd=None, check=True):
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout

def _has_awscli() -> bool:
    return shutil.which("aws") is not None

def _tar_extract_strip1(tar_path: str, dst_dir: str):
    """
    Extract tar.gz while stripping the first path component so that files in
    'validation/...jpg' land directly under dst_dir, not dst_dir/validation/.
    """
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = []
        for m in tf.getmembers():
            # strip first component if present
            parts = m.name.split("/", 1)
            if len(parts) == 2:
                m.name = parts[1]
            else:
                # already a file at root
                m.name = parts[0]
            if m.name and not m.name.endswith("/"):
                members.append(m)
        tf.extractall(dst_dir, members=members)

def _fetch_split_tar_to(split: str, split_dir: str):
    """
    Download split tar.gz from CVDF (no auth) and extract into data/{split}/
    with the top-level 'split/' folder stripped.
    """
    url = S3_TARS[split]
    tmp_tar = os.path.join(tempfile.gettempdir(), f"openimages_{split}.tar.gz")
    os.makedirs(split_dir, exist_ok=True)

    if _has_awscli():
        _run(["aws", "s3", "--no-sign-request", "cp", url, tmp_tar])
    else:
        _run(["curl", "-L", "-o", tmp_tar, url])

    _tar_extract_strip1(tmp_tar, split_dir)
    os.remove(tmp_tar)

def _download_with_official_downloader(root: str, split: str, ids):
    dl = os.path.join(tempfile.gettempdir(), "downloader.py")
    if not os.path.exists(dl):
        _run(["curl", "-L", "-o", dl, "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"])
    idstxt = os.path.join(tempfile.gettempdir(), f"oi_{split}_ids.txt")
    with open(idstxt, "w") as f:
        for _id in ids:
            f.write(_id if _id.startswith(f"{split}/") else f"{split}/{_id}\n")
    _run([sys.executable, dl, idstxt, "--download_folder", os.path.join(root, "data"), "--num_processes", "8"])

def _flatten_split_dir(split_dir: str, split: str) -> int:
    """
    If tar created data/<split>/<split>/..., move all files up one level.
    Returns count of files moved.
    """
    nested = os.path.join(split_dir, split)
    moved = 0
    if os.path.isdir(nested):
        for root, _, files in os.walk(nested):
            rel = os.path.relpath(root, nested)
            dst_root = os.path.join(split_dir, rel) if rel != "." else split_dir
            os.makedirs(dst_root, exist_ok=True)
            for fn in files:
                src = os.path.join(root, fn)
                dst = os.path.join(dst_root, fn)
                if not os.path.exists(dst):
                    os.replace(src, dst)
                    moved += 1
        # cleanup empty dirs
        for root, dirs, files in os.walk(nested, topdown=False):
            for d in dirs:
                p = os.path.join(root, d)
                try:
                    os.rmdir(p)
                except OSError:
                    pass
        try:
            os.rmdir(nested)
        except OSError:
            pass
        logger.info(f"[Flatten] Moved {moved} files up from {nested} into {split_dir}")
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
    Valid for 'train', 'validation', and 'test'.
    * For validation/test, uses CVDF tarballs (no auth) if images are missing.
    * Hydrates validation labels so OpenImagesV7 importer can enumerate.
    * Test is images-only (no detections exist).
    """
    root = download_dir or os.path.expanduser(os.path.join("~", "fiftyone", "open-images-v7"))
    data_root = os.path.join(root, "data")
    labels_dir = os.path.join(root, "labels")
    metadata_dir = os.path.join(root, "metadata")
    _ensure_dir(root); _ensure_dir(data_root); _ensure_dir(labels_dir); _ensure_dir(metadata_dir)

    if dataset_name is None:
        dataset_name = f"openimages-v7-{split}-det" if split != "test" else "openimages-v7-test-imgs"

    # 1) image IDs for the split
    ids_csv = _ensure_split_ids_csv(root, split)
    idset = _read_ids(ids_csv)
    if not idset:
        raise FileNotFoundError(f"Could not read any ImageIDs for split '{split}' from {ids_csv}")

    split_dir = os.path.join(data_root, split)
    _ensure_dir(split_dir)

    # 2) Count present/missing (recursive to handle accidental nesting)
    missing_ids = _find_missing_ids(split, split_dir, idset)
    logger.info(f"[{split}] Present={len(idset)-len(missing_ids)}  Missing={len(missing_ids)}")

    # 3) Fill images
    if missing_ids:
        if prefer_manual_images and split in ("validation", "test"):
            print(f"[{split}] Fetching CVDF tarball (S3)…")
            _fetch_split_tar_to(split, split_dir)
            # fix possible nested split folder
            _flatten_split_dir(split_dir, split)
            # recount after extraction
            missing_ids = _find_missing_ids(split, split_dir, idset)
            logger.info(f"[{split}] After tar: Present={len(idset)-len(missing_ids)}  Missing={len(missing_ids)}")
            # If still missing a subset (rare), grab those via official downloader
            if missing_ids:
                print(f"[{split}] Downloading {len(missing_ids)} missing via official downloader.py…")
                _download_with_official_downloader(root, split, missing_ids)
        else:
            # Either train or manual_images=False → try FO downloader or per-ID HTTP
            if split == "train":
                # You already fetched train; keep per-ID fallback here if needed
                fetched = _download_missing_images_manual(split, split_dir, missing_ids, workers=(num_workers or 32))
                logger.info(f"[{split}] Manually fetched {fetched} images")
            else:
                fouo.download_open_images_split(
                    dataset_dir=root,
                    split=split,
                    version="v7",
                    label_types=[],            # images only
                    classes=None,
                    image_ids=missing_ids,
                    num_workers=os.cpu_count(),
                    shuffle=shuffle,
                    seed=seed,
                    max_samples=None,
                )
                _flatten_split_dir(split_dir, split)

    # 4) Clean zero-byte/HTML leftovers
    cleaned = _cleanup_bad_media(split, split_dir)
    if cleaned:
        logger.info(f"[{split}] Cleaned {cleaned} bad files under {split_dir}")

    # 5) Hydrate labels for validation; test has no detections
    if split == "validation":
        fouo.download_open_images_split(
            dataset_dir=root,
            split="validation",
            version="v7",
            label_types=list(label_types),     # ('detections',)
            classes=classes,
            image_ids=None,                    # full split
            num_workers=8,
            shuffle=False,
            seed=seed,
            max_samples=None,
        )

    # 6) Load dataset
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
        # Ensure IDs file exists (split-prefixed)
        ids_file = _ensure_prefixed_ids_file(root, split)
        OIType = _resolve_oi_dataset_type()
        logger.info(f"---> [Load] root='{root}', split='{split}', dataset='{dataset_name}' <---")
        ds = fo.Dataset.from_dir(
            dataset_dir=root,
            dataset_type=OIType,
            label_types=list(label_types),
            classes=classes,
            image_ids=ids_file,      # lines like 'validation/<ImageID>'
            include_id=True,
            name=dataset_name,
        )

    ds.persistent = True
    logger.info(f"[OpenImages] Loaded: {ds.name} | split={split} | samples={len(ds)}")
    return ds

def _ensure_prefixed_ids_file(root: str, split: str) -> str:
    meta_dir = os.path.join(root, "metadata", split)
    os.makedirs(meta_dir, exist_ok=True)
    src_csv = os.path.join(meta_dir, "image_ids.csv")
    out_txt = os.path.join(meta_dir, "image_ids_prefixed.txt")
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        return out_txt
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Missing {src_csv}")
    with open(src_csv, "r", newline="", encoding="utf-8") as f, open(out_txt, "w") as g:
        r = csv.reader(f)
        header = next(r, None)
        idx = 0 if not header or "ImageID" not in header else header.index("ImageID")
        if header and ("ImageID" not in header):
            if header and len(header) > idx:
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

                # If needed:
                # download_coco_2017_images(Path(image_path))
                # download_lvis_annotations(Path(annotation_path))

                download_dir = os.path.join(base_path, "open-images-v7")
                os.makedirs(download_dir, exist_ok=True)

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

    logger.info(f"""---> FINISHED [WORKFLOW: {os.getenv("RAPTOR_WORKFLOW_DOWNLOAD_DATA")}] "
                f"FOR [MODULE: {os.getenv("RAPTOR_PROJECT_MODE")}] <---""")

if __name__ == '__main__':
    main()
