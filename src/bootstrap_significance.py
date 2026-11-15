"""Paired-bootstrap significance test for mAP@0.5 differences (thesis Section 4.2.4, Appendix B.5).

Both models are evaluated on the same resampled image sets in every replicate, so the
test measures the difference between two models on identical data.

Inputs are YOLO-format text files:
  labels dir:  <image>.txt  ->  "cls cx cy w h"          (normalized)
  preds dirs:  <image>.txt  ->  "cls cx cy w h conf"     (normalized; Ultralytics save_txt + save_conf)

Usage:
    python src/bootstrap_significance.py --labels val/labels \
        --preds-a runs/v8n_raw/labels --preds-b runs/v8n_clahe/labels --nc 3

The paired_bootstrap() core follows the thesis listing. The IoU matching and the
101-point AP computation (map50_from_images) were written for this repository,
because the thesis listing does not show them.
"""
import argparse
import glob
import os

import numpy as np


# ---------- per-image matching (added for this repository) ----------

def _xywh_to_xyxy(b):
    cx, cy, w, h = b.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _read(path, ncols):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, ncols))
    return np.loadtxt(path, ndmin=2)[:, :ncols]


def match_image(gt, pred, nc, iou_thr=0.5):
    """Greedy per-class matching, highest confidence first. Returns one record per image."""
    tp = np.zeros(len(pred), bool)
    n_gt = np.bincount(gt[:, 0].astype(int), minlength=nc) if len(gt) else np.zeros(nc, int)
    if len(pred) and len(gt):
        order = np.argsort(-pred[:, 5])
        pred = pred[order]
        tp = tp[order]
        gbox, pbox = _xywh_to_xyxy(gt[:, 1:5]), _xywh_to_xyxy(pred[:, 1:5])
        ious = _iou(pbox, gbox)
        used = np.zeros(len(gt), bool)
        for i in range(len(pred)):
            same = (gt[:, 0] == pred[i, 0]) & ~used
            if not same.any():
                continue
            j = np.where(same)[0][np.argmax(ious[i, same])]
            if ious[i, j] >= iou_thr:
                tp[i] = True
                used[j] = True
    return {"cls": pred[:, 0].astype(int), "conf": pred[:, 5], "tp": tp, "n_gt": n_gt}


def load_records(labels_dir, preds_dir, nc):
    records = []
    for lbl in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        name = os.path.basename(lbl)
        gt = _read(lbl, 5)
        pred = _read(os.path.join(preds_dir, name), 6)
        records.append(match_image(gt, pred, nc))
    return records


# ---------- AP computation (added for this repository) ----------

def _ap101(tp, conf, n_gt):
    if n_gt == 0:
        return np.nan
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    # precision envelope, then 101-point interpolation (COCO style)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def map50_from_images(records, idx):
    """mAP@0.5 over the images selected by idx (indices may repeat)."""
    sel = [records[i] for i in idx]
    nc = len(sel[0]["n_gt"])
    aps = []
    for c in range(nc):
        n_gt = sum(int(r["n_gt"][c]) for r in sel)
        tp = np.concatenate([r["tp"][r["cls"] == c] for r in sel])
        conf = np.concatenate([r["conf"][r["cls"] == c] for r in sel])
        aps.append(_ap101(tp, conf, n_gt))
    return float(np.nanmean(aps))


# ---------- paired bootstrap (thesis Appendix B.5) ----------

def paired_bootstrap(per_image_A, per_image_B, B=2000, seed=0):
    """per_image_X: per-image matched predictions (TP/FP/conf/class)
    from model X on the SAME validation images."""
    rng = np.random.default_rng(seed)
    n = len(per_image_A)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)              # resample images w/ replacement
        mA = map50_from_images(per_image_A, idx)  # 101-pt interpolated AP
        mB = map50_from_images(per_image_B, idx)
        deltas[b] = mB - mA
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))  # capped at 1 for ties
    return deltas.mean(), (lo, hi), p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paired-bootstrap test of mAP@0.5 (B minus A)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds-a", required=True)
    ap.add_argument("--preds-b", required=True)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A = load_records(args.labels, args.preds_a, args.nc)
    Bm = load_records(args.labels, args.preds_b, args.nc)
    all_idx = np.arange(len(A))
    print(f"images: {len(A)}  mAP@0.5 A={map50_from_images(A, all_idx):.4f}  "
          f"B={map50_from_images(Bm, all_idx):.4f}")
    delta, (lo, hi), p = paired_bootstrap(A, Bm, B=args.B, seed=args.seed)
    print(f"delta (B-A) = {delta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p = {p:.3f}")
