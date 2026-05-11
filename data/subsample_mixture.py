"""Subsample the merged COCO mixture for sanity-check training runs.

Produces small JSONs that the trainer can be pointed at via
`RAPTOR_PATHS_TRAIN_JSON` / `RAPTOR_PATHS_VAL_JSON`:

    datasets/mixture/annotations/coco_merged/instances_train_sub5k.json
    datasets/mixture/annotations/coco_merged/instances_val_sub1k.json

Strategy: greedy category coverage first (each pick adds new categories),
then random fill to reach the target image count. ALL categories from the
source JSON are preserved in the output so the model is built with the
same classifier shape as production (1,579 slots) — only `images` and
`annotations` are subset.
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common.config import Config  # noqa: E402

SEED = 42
TRAIN_N = 5000
VAL_N = 1000

SRC_TRAIN = REPO / Config.TRAIN_JSON
SRC_VAL   = REPO / Config.VAL_JSON
OUT_TRAIN = SRC_TRAIN.parent / "instances_train_sub5k.json"
OUT_VAL   = SRC_VAL.parent   / "instances_val_sub1k.json"


def subsample(src_json: Path, out_json: Path, target_n: int, seed: int):
    print(f"\n=== subsampling {src_json.name} -> {out_json.name} (target={target_n}) ===")
    js = json.load(open(src_json))
    images = js["images"]
    anns = js["annotations"]
    cats = js["categories"]
    print(f"  source: images={len(images):,}  anns={len(anns):,}  cats={len(cats):,}")

    # Build image_id -> annotations and image_id -> set(cat_id)
    anns_by_img = defaultdict(list)
    cats_by_img = defaultdict(set)
    for a in tqdm(anns, desc="  indexing anns", unit="ann"):
        anns_by_img[a["image_id"]].append(a)
        cats_by_img[a["image_id"]].add(a["category_id"])

    # Only images that actually have at least one annotation are useful
    annotated_imgs = [im for im in images if im["id"] in anns_by_img]
    print(f"  images with >=1 ann: {len(annotated_imgs):,}")

    rng = random.Random(seed)

    # --- Greedy category coverage ---
    # At each step pick the image that adds the most uncovered categories.
    # Cheap heuristic: bucket images by their "best uncovered count" and
    # iterate. For 1.8M imgs this is too slow as a strict argmax loop, so
    # we do a randomized-greedy: shuffle, scan, keep if it adds ≥1 new cat.
    uncovered = {c["id"] for c in cats}
    picked_ids = []
    shuffled = annotated_imgs[:]
    rng.shuffle(shuffled)

    # Pass 1: cover-driven sweep
    for im in tqdm(shuffled, desc="  greedy cover", unit="img"):
        if not uncovered or len(picked_ids) >= target_n:
            break
        gain = cats_by_img[im["id"]] & uncovered
        if gain:
            picked_ids.append(im["id"])
            uncovered -= gain

    # Pass 2: random fill the remainder
    if len(picked_ids) < target_n:
        chosen = set(picked_ids)
        remaining = [im["id"] for im in shuffled if im["id"] not in chosen]
        need = target_n - len(picked_ids)
        picked_ids.extend(remaining[:need])

    picked_ids = picked_ids[:target_n]
    picked_set = set(picked_ids)

    # Coverage report
    covered = {c["id"] for c in cats} - uncovered
    print(f"  picked={len(picked_ids):,}  covered_cats={len(covered)}/{len(cats)}  "
          f"(uncovered={len(uncovered)})")

    # Materialize subset
    out_images = [im for im in images if im["id"] in picked_set]
    out_anns = []
    for iid in picked_ids:
        out_anns.extend(anns_by_img[iid])

    out = {
        "images": out_images,
        "annotations": out_anns,
        "categories": cats,  # keep full category list so model shape is unchanged
    }
    json.dump(out, open(out_json, "w"))
    print(f"  wrote {out_json}  "
          f"images={len(out_images):,}  anns={len(out_anns):,}  cats={len(cats):,}")


def main():
    subsample(SRC_TRAIN, OUT_TRAIN, TRAIN_N, seed=SEED)
    subsample(SRC_VAL,   OUT_VAL,   VAL_N,   seed=SEED + 1)
    print("\nDone. Point training at the subsamples with:")
    print(f"  export RAPTOR_PATHS_TRAIN_JSON={OUT_TRAIN.relative_to(REPO)}")
    print(f"  export RAPTOR_PATHS_VAL_JSON={OUT_VAL.relative_to(REPO)}")


if __name__ == "__main__":
    main()
