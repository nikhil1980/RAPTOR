import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
""" System Modules """

from common.env import load_env_from_json
from common.logger import get_logger
from common.config import Config
from common.myargparser import build_myargparser
""" User Modules """

"""
Collapse cross-dataset taxonomy duplicates in the already-merged COCO JSONs.

Background: the merged mixture comes from LVIS + OpenImages + COCO. Each dataset
ships its own taxonomy (LVIS=1203 classes, OI=601 classes, COCO=80). After
name-based merging in `convert_and_merge.unify_and_merge()`, classes whose
*display names* differ remain separate even if they describe the same physical
object. The May 19 12-epoch run's top confusion table shows exactly this:

    pastry        <-> doughnut      208 confusions
    Girl          <-> Woman          96
    pigeon        <-> bird           68
    Human body    <-> Man            60
    Tire          <-> wheel          14

The OI label-set uses Title-case English ("Man", "Woman", "Girl") while LVIS
uses snake_case ("person", "bird", "wheel"). When both appear over the same
GT box, the model splits AP across them.

This script reads the already-merged train + val JSONs, applies a synonym map
(canonical -> set of aliases), and rewrites every aliased annotation to the
canonical class. Empty classes after the rewrite are pruned, and category IDs
are compacted to a contiguous range. Backups of the originals are written
alongside as `<file>.bak.json`.

Runs in 1-5 minutes for the 2.7GB merged train JSON. Much faster and safer
than re-running the full `convert_and_merge.py` pipeline.

@author: Nikhil Bhargava
@date: 2026-05-23
@license: Apache-2.0
"""

logger = get_logger(__name__)


# --- Synonym map ---
# Key = canonical class name kept after collapse.
# Values = display names from any source dataset that should be folded into it.
# Comparison is case-insensitive and whitespace-trimmed.
SYNONYM_MAP: Dict[str, List[str]] = {
    "person": [
        "person", "Man", "Woman", "Girl", "Boy", "Human body",
    ],
    "pastry": [
        "pastry", "doughnut", "Doughnut", "Croissant", "croissant",
    ],
    "wheel": [
        "wheel", "Tire", "Bicycle wheel",
    ],
    "bottle": [
        "bottle", "water_bottle", "Bottle",
    ],
    "orange_(fruit)": [
        "orange_(fruit)", "mandarin_orange", "Orange",
    ],
    "bird": [
        "bird", "pigeon", "Bird",
    ],
    "suitcase": [
        "suitcase", "trunk", "Suitcase",
    ],
}


def _norm(name: str) -> str:
    return name.strip().lower()


def _build_alias_to_canonical(syn_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Flatten the canonical->aliases map into alias_norm -> canonical_norm."""
    out: Dict[str, str] = {}
    for canonical, aliases in syn_map.items():
        cn = _norm(canonical)
        out[cn] = cn
        for a in aliases:
            an = _norm(a)
            if an in out and out[an] != cn:
                logger.warning(f"Alias collision: '{a}' maps to both "
                               f"'{out[an]}' and '{cn}' — keeping first")
                continue
            out[an] = cn
    return out


def _collapse_one(in_path: Path, out_path: Path, alias2canon: Dict[str, str]) -> Tuple[int, int, int]:
    """
    Rewrite a single merged COCO JSON with collapsed categories.

    1. Read.
    2. For every category whose name (normalized) is an alias, map it to its
       canonical. The new merged categories list keeps one entry per
       *canonical name*, preferring the canonical display name as written in
       SYNONYM_MAP keys. Non-aliased categories pass through unchanged.
    3. Rewrite every annotation's `category_id` to the new canonical id.
    4. Drop empty categories (no annotations after collapse) only if they
       were among the collapsed aliases — keep otherwise-empty originals.
    5. Compact category ids to 1..K so pycocotools is happy.

    :return: (n_categories_in, n_categories_out, n_annotations_rewritten)
    """
    logger.info(f"---> [COLLAPSE] reading {in_path} <---")
    js = json.load(open(in_path))
    cats_in = js["categories"]

    # Step A: map each original category id -> canonical normalized name.
    # If the cat name isn't in the synonym map, its canonical is itself.
    orig_to_canon_name: Dict[int, str] = {}
    canon_display: Dict[str, str] = {}  # canonical_norm -> display name to write
    # Pre-fill canonical display from SYNONYM_MAP keys (preserves capitalization).
    for canon in SYNONYM_MAP.keys():
        canon_display[_norm(canon)] = canon
    for c in cats_in:
        nm = _norm(c["name"])
        canon = alias2canon.get(nm, nm)
        orig_to_canon_name[c["id"]] = canon
        # If this canonical wasn't in SYNONYM_MAP (passthrough), use the cat's own display.
        if canon not in canon_display:
            canon_display[canon] = c["name"]

    # Step B: assign new compact ids to canonical names in stable order.
    # Sort by display name for determinism.
    canon_names_sorted = sorted(set(orig_to_canon_name.values()),
                                key=lambda n: canon_display[n].lower())
    canon_name_to_new_id: Dict[str, int] = {n: i + 1 for i, n in enumerate(canon_names_sorted)}
    new_categories = [
        {"id": canon_name_to_new_id[n], "name": canon_display[n]}
        for n in canon_names_sorted
    ]

    # Step C: rewrite annotations.
    orig_to_new_id: Dict[int, int] = {
        oid: canon_name_to_new_id[orig_to_canon_name[oid]] for oid in orig_to_canon_name
    }
    n_rewritten = 0
    new_anns = []
    for a in tqdm(js["annotations"], desc=f"[COLLAPSE] anns {in_path.name}", unit="ann"):
        old_cid = a["category_id"]
        new_cid = orig_to_new_id.get(old_cid)
        if new_cid is None:
            continue  # ann references unknown category — drop defensively
        if new_cid != old_cid:
            n_rewritten += 1
        new_anns.append({**a, "category_id": new_cid})

    # Step D: keep all canonical categories (even if empty post-collapse) so the
    # model's class list is stable across train/val. pycocotools is fine with
    # zero-annotation classes — they just get AP=0 / nan in eval.
    js["categories"] = new_categories
    js["annotations"] = new_anns
    json.dump(js, open(out_path, "w"))
    return len(cats_in), len(new_categories), n_rewritten


def main():
    parser = build_myargparser()
    args = parser.parse_args()
    if not args.config_file:
        logger.info("---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")
        return
    par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(par_path, str(args.config_file))
    if not os.path.exists(cfg_path):
        logger.error(f"---> CONFIG NOT FOUND: {cfg_path} <---")
        return
    load_env_from_json(cfg_path)
    Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT", Config.ROOT)

    coco_ann_dir = os.path.join(par_path, Config.ROOT, "annotations", "coco_merged")
    targets = [
        Path(coco_ann_dir) / "instances_train_merged.json",
        Path(coco_ann_dir) / "instances_val_merged.json",
        Path(coco_ann_dir) / "instances_train_sub5k.json",
        Path(coco_ann_dir) / "instances_val_sub1k.json",
    ]

    alias2canon = _build_alias_to_canonical(SYNONYM_MAP)
    logger.info(f"---> [COLLAPSE] {len(alias2canon)} aliases mapping to "
                f"{len(set(alias2canon.values()))} canonicals <---")

    t0 = time.perf_counter()
    for p in targets:
        if not p.exists():
            logger.warning(f"  skip (not found): {p}")
            continue
        bak = p.with_suffix(".bak.json")
        if not bak.exists():
            logger.info(f"  backing up {p} -> {bak}")
            shutil.copy2(p, bak)
        else:
            logger.info(f"  backup already exists at {bak}; reading from there to avoid double-collapse")
            shutil.copy2(bak, p)  # restore from backup first, then re-collapse
        ci, co, nr = _collapse_one(bak, p, alias2canon)
        logger.info(f"  {p.name}: categories {ci} -> {co}, annotations rewritten: {nr}")
    dt = time.perf_counter() - t0
    logger.info(f"---> [COLLAPSE] done in {dt:.1f}s <---")


if __name__ == "__main__":
    main()
