import os
import sys
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Silence noisy HTTP/filelock debug spam so PASS/FAIL lines stay readable.
# (HF Hub does several harmless 404 probes for optional files — they were
# the alarming-looking 404s in earlier runs.)
for _noisy in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11",
               "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import time
from pathlib import Path

import torch

from common.env import load_env_from_json
from common.logger import get_logger
from common.myargparser import build_myargparser
from common.config import Config

from train.train_dinov3_rtdetr_ov import (
    build_processor_and_datasets,
    build_model,
    collate_fn,
    DinoV3FPNBackbone,
    ModelWithOV,
    LongTailTrainer,
    compute_image_weights_from_json,
    compute_rfs_image_weights_from_json,
    build_weight_vector_for_dataset,
)
# Eagerly import callbacks too — they're lazy-imported inside train(), so a
# syntax error here only surfaces hours into the actual training launch. Pull
# them in at smoke-test time as a cheap static-import gate.
from train.wandb_callbacks import (  # noqa: F401
    SamplePredictionsCallback,
    PeriodicCOCOEvalCallback,
    BackboneUnfreezeCallback,
    EndOfRunArtifactsCallback,
    EMAWeightsCallback,
)
""" User Modules """

"""
End-to-end smoke test for the DINOv3 + RT-DETR + OV training pipeline.

Validates, in order:
    1. Dataset + processor build, COCO ann -> sample with valid labels.
    2. Model assembly: DinoV3FPNBackbone swap, OV head wrap, text_embeds
       on the right device.
    3. Backbone is fully frozen (no grads on DINOv3 body).
    4. FPN produces 3 feature maps at strides 8 / 16 / 32.
    5. One real batch -> forward -> backward gives finite loss; gradients
       flow into OV head and FPN, NOT into the frozen DINOv3 body.
    6. Overfit-on-one-batch: 300 steps @ lr=5e-4 must drive loss down >50%.
       This is the single best predictor of "will training learn anything";
       if it fails, do NOT launch a long run.
    7. Eval forward pass on a val sample produces finite loss.
    8. Class-aware sampler weights are non-uniform.

@author: Nikhil Bhargava
@date: 2026-05-08
@license: Apache-2.0
"""

logger = get_logger(__name__)

BASE_PATH = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------- helpers ----------
def _check(cond: bool, msg: str):
    """Assert-or-die with a uniform PASS/FAIL line."""
    if cond:
        logger.info(f"  [PASS] {msg}")
    else:
        logger.error(f"  [FAIL] {msg}")
        sys.exit(1)


def _get_fpn(model):
    """Reach the DinoV3FPNBackbone regardless of OV-head wrapping."""
    base = model.base if isinstance(model, ModelWithOV) else model
    return base.model.backbone


def _move_batch(batch, device):
    return {
        "pixel_values": batch["pixel_values"].to(device),
        "labels": [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in lab.items()}
            for lab in batch["labels"]
        ],
    }


def _find_indices_with_annotations(ds, k: int = 2, max_search: int = 100):
    """Pick k samples that actually have GT — needed for a meaningful overfit test."""
    picked = []
    for i in range(min(max_search, len(ds))):
        item = ds[i]
        if item["labels"][0]["class_labels"].numel() > 0:
            picked.append(i)
            if len(picked) >= k:
                return picked
    return picked or [0]


# --------- stages ----------
def stage_build_data():
    logger.info("[1/8] Building processor + datasets")
    image_processor, train_ds, val_ds, id2label, label2id = build_processor_and_datasets()
    logger.info(f"  train={len(train_ds)}  val={len(val_ds)}  classes={len(id2label)}")
    _check(len(train_ds) > 0, "train dataset is non-empty")
    _check(len(val_ds) > 0, "val dataset is non-empty")
    _check(len(id2label) > 0, "id2label is non-empty")
    return image_processor, train_ds, val_ds, id2label, label2id


def stage_build_model(id2label, label2id, device):
    logger.info(f"[2/8] Building model (RT-DETR + FPN + OV head)")
    logger.info(f"  backbone   = {Config.PRETRAINED_BACKBONE}")
    logger.info(f"  text enc.  = {Config.TEXT_ENCODER}")
    logger.info(f"  image size = {Config.IMAGE_SIZE}")
    t0 = time.perf_counter()
    model = build_model(len(id2label), id2label, label2id, device=device).to(device)
    dt = time.perf_counter() - t0
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  built in {dt:.1f}s  total={n_total/1e6:.1f}M  trainable={n_train/1e6:.1f}M")
    _check(n_train > 0, "at least some params are trainable")
    _check(n_train < n_total, "some params are frozen (backbone body)")
    return model


def stage_check_freeze(model):
    logger.info("[3/8] Checking backbone freeze")
    fpn = _get_fpn(model)
    _check(isinstance(fpn, DinoV3FPNBackbone), "backbone is DinoV3FPNBackbone")
    body_trainable = sum(int(p.requires_grad) for p in fpn.body.parameters())
    _check(body_trainable == 0, f"DINOv3 body fully frozen (got {body_trainable} trainable)")
    fpn_adapter_trainable = (
        sum(int(p.requires_grad) for p in fpn.lat8.parameters())
        + sum(int(p.requires_grad) for p in fpn.lat16.parameters())
        + sum(int(p.requires_grad) for p in fpn.lat32.parameters())
    )
    _check(fpn_adapter_trainable > 0, "FPN adapter (lat8/lat16/lat32) is trainable")


def stage_check_fpn_shapes(model, device):
    logger.info("[4/8] Checking FPN multi-scale output shapes")
    fpn = _get_fpn(model)
    fpn.eval()
    # Accept either RT-DETR-style {"height", "width"} or letterbox {"shortest_edge", "longest_edge"}.
    img_size = Config.IMAGE_SIZE.get("height", Config.IMAGE_SIZE.get("shortest_edge", 640))
    pix = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        feats = fpn(pix)
    use_p2 = getattr(fpn, "use_p2", False)
    expected_n = 4 if use_p2 else 3
    _check(len(feats) == expected_n,
           f"FPN returns {expected_n} levels (got {len(feats)}, use_p2={use_p2})")

    # Expected sizes derive from the backbone's patch_size (DINOv2=14, DINOv3=16, ...)
    # Levels with P2:  lat4 (4x upsample), lat8 (2x), lat16 (identity), lat32 (/2).
    # Levels without:  lat8, lat16, lat32. Off-by-one tolerated when conv stride
    # padding rounds an odd grid.
    grid = img_size // fpn.patch_size
    expected_full = [
        ("upsample 4x   ", 4 * grid,         0),
        ("upsample 2x   ", 2 * grid,         0),
        ("identity      ", grid,             0),
        ("downsample /2 ", (grid + 1) // 2,  1),
    ]
    expected = expected_full if use_p2 else expected_full[1:]
    actual_hw = []
    for (f, _), (label, h_exp, tol) in zip(feats, expected):
        h_act = f.shape[-1]
        actual_hw.append(h_act)
        logger.info(f"  {label} shape={tuple(f.shape)}  expected H,W~={h_exp} (tol +/- {tol})")
        _check(f.shape[-1] == f.shape[-2], f"{label.strip()} feature is square")
        _check(abs(h_act - h_exp) <= tol, f"{label.strip()} feature near expected size")

    # 2x ratio between adjacent levels is what RT-DETR's HybridEncoder depends on.
    for i in range(len(actual_hw) - 1):
        hi, hj = actual_hw[i], max(actual_hw[i + 1], 1)
        tol = 0.05 if i < len(actual_hw) - 2 else 0.15
        _check(abs(hi / hj - 2.0) < tol,
               f"level{i}/level{i+1} ratio ~2x (got {hi/hj:.3f})")

    if isinstance(model, ModelWithOV):
        _check(model.text_embeds.device.type == torch.device(device).type,
               f"text_embeds buffer is on {device}")


def stage_check_gradient_flow(model, batch):
    logger.info("[5/8] Single forward + backward — checking gradient flow")
    model.train()
    model.zero_grad()
    out = model(**batch)
    loss = out.loss
    logger.info(f"  initial loss={loss.item():.4f}")
    _check(torch.isfinite(loss).item(), "initial loss is finite")
    loss.backward()

    fpn = _get_fpn(model)
    # OV head must see gradients
    if isinstance(model, ModelWithOV):
        ov_grad = model.ov_head.proj.weight.grad
        ov_ok = ov_grad is not None and ov_grad.abs().sum().item() > 0
        _check(ov_ok, "OV head proj.weight has non-zero gradient (Hungarian-matched targets reach the head)")

    # FPN adapter must see gradients
    lat16_w = fpn.lat16[0].weight
    lat16_grad = lat16_w.grad
    fpn_ok = lat16_grad is not None and lat16_grad.abs().sum().item() > 0
    _check(fpn_ok, "FPN lat16 has non-zero gradient")

    # DINOv3 body must NOT see gradients
    body_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                        for p in fpn.body.parameters())
    _check(not body_has_grad, "DINOv3 body has no gradients (truly frozen)")
    model.zero_grad()


def stage_overfit_one_batch(model, batch, steps: int = 300, lr: float = 5e-4):
    logger.info(f"[6/8] Overfitting on a single batch ({steps} steps, lr={lr})")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for i in range(steps):
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(out.loss.item())
        if i == 0 or (i + 1) % 10 == 0:
            logger.info(f"  step {i+1:>3d}/{steps}  loss={out.loss.item():.4f}")
    init = losses[0]
    final = min(losses[-5:])  # min over last 5 to be robust to noise
    drop = (init - final) / max(init, 1e-6)
    logger.info(f"  initial={init:.4f}  final(min last 5)={final:.4f}  drop={drop*100:.1f}%")
    _check(all(torch.isfinite(torch.tensor(l)).item() for l in losses), "all losses finite")
    _check(drop > 0.5, f"loss dropped >50% on a single batch (got {drop*100:.1f}%)")
    if drop < 0.8:
        logger.warning("  drop < 80% — model is learning but the OV/closed-set balance "
                       "or LR may need tuning. Acceptable, but worth noting.")


def stage_eval_pass(model, val_ds, device):
    logger.info("[7/8] Eval forward pass")
    model.eval()
    idx = _find_indices_with_annotations(val_ds, k=1)[0]
    batch = collate_fn([val_ds[idx]])
    batch = _move_batch(batch, device)
    with torch.no_grad():
        out = model(**batch)
    _check(torch.isfinite(out.loss).item(), f"eval loss finite (val[{idx}], loss={out.loss.item():.4f})")


def stage_check_sampler(train_ds):
    sampler_kind = os.getenv("RAPTOR_SAMPLER_KIND", "rfs").lower()
    logger.info(f"[8/8] Checking class-aware sampler weights (kind={sampler_kind})")
    train_json = Path(os.path.join(BASE_PATH, str(Config.TRAIN_JSON)))
    if sampler_kind == "rfs":
        img_w = compute_rfs_image_weights_from_json(
            train_json, threshold=float(os.getenv("RAPTOR_SAMPLER_RFS_THRESHOLD", "0.001")),
        )
    else:
        img_w = compute_image_weights_from_json(train_json, beta=0.8)
    w = build_weight_vector_for_dataset(train_ds, img_w)
    _check(w.numel() == len(train_ds), "weight vector aligned with dataset length")
    std_over_mean = (w.std() / w.mean()).item() if w.mean() > 0 else 0.0
    logger.info(f"  weights: min={w.min().item():.4g}  max={w.max().item():.4g}  "
                f"std/mean={std_over_mean:.3f}")
    _check(std_over_mean > 0.05, "sampler weights are non-uniform (long-tail kicks in)")


# --------- driver ----------
def run_smoke_test():
    device = Config.DEVICE
    logger.info(f"---> SMOKE TEST starting on device={device} <---")
    t0 = time.perf_counter()

    image_processor, train_ds, val_ds, id2label, label2id = stage_build_data()
    model = stage_build_model(id2label, label2id, device)
    stage_check_freeze(model)
    stage_check_fpn_shapes(model, device)

    # Build one real training batch (samples that have annotations)
    train_idx = _find_indices_with_annotations(train_ds, k=2)
    logger.info(f"  using train indices {train_idx} for the overfit batch")
    batch = collate_fn([train_ds[i] for i in train_idx])
    batch = _move_batch(batch, device)

    stage_check_gradient_flow(model, batch)
    stage_overfit_one_batch(model, batch, steps=300, lr=5e-4)
    stage_eval_pass(model, val_ds, device)
    stage_check_sampler(train_ds)

    dt = time.perf_counter() - t0
    logger.info(f"---> ALL CHECKS PASSED in {dt:.1f}s — pipeline is ready for full training <---")


def main():
    parser = build_myargparser()
    args = parser.parse_args()
    if not args.config_file:
        logger.info("---> PLEASE SPECIFY A VALID CONFIG JSON FILE VIA --config-file <---")
        return
    par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(par_path, str(args.config_file))
    if not os.path.exists(config_file_path):
        logger.info(f"---> CONFIG FILE NOT FOUND AT: {config_file_path} <---")
        return
    load_env_from_json(config_file_path)
    Config.ROOT = os.getenv("RAPTOR_PATHS_ROOT", Config.ROOT)
    run_smoke_test()


if __name__ == "__main__":
    main()
