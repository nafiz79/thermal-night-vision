"""Training-free infrared-visible fusion and enhancement evaluation on TNO
(thesis Algorithm 2, Appendix B.2).

For each IR/visible pair and each fusion coefficient alpha, two outputs are scored:
  - plain fusion:  F = alpha * IR + (1 - alpha) * V
  - CLAHE fusion:  same, with CLAHE applied to the visible luminance (LAB L channel)
PSNR and SSIM are computed against the visible image; BRISQUE is no-reference.

Input is a CSV with columns ir_path,vis_path (one row per co-registered pair).

Usage:
    python src/fusion_eval.py --pairs pairs.csv --out results/tno
"""
import argparse
import csv
import os

import cv2
import numpy as np
import piq
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

ALPHAS = (0.6, 0.7, 0.8)


def clahe_luminance(img_bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return cv2.cvtColor(cv2.merge([c.apply(L), a, b]), cv2.COLOR_LAB2BGR)


def fuse(ir_gray, vis_bgr, alpha):
    ir3 = cv2.merge([ir_gray] * 3).astype(np.float32) / 255.0
    v = vis_bgr.astype(np.float32) / 255.0
    f = np.clip(alpha * ir3 + (1.0 - alpha) * v, 0.0, 1.0)
    return (f * 255.0 + 0.5).astype(np.uint8)


def psnr_ssim(img_bgr, ref_bgr):
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.0
    a, b = g(img_bgr), g(ref_bgr)
    return sk_psnr(b, a, data_range=1.0), sk_ssim(b, a, data_range=1.0)


def brisque(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(piq.brisque(t, data_range=1.0))


def load_pair(ir_path, vis_path):
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)  # grayscale visibles load as 3 equal channels
    if ir.shape[:2] != vis.shape[:2]:
        ir = cv2.resize(ir, (vis.shape[1], vis.shape[0]))
    return ir, vis


def evaluate(pairs_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(pairs_csv, newline="") as fh:
        pairs = list(csv.DictReader(fh))

    for pair in pairs:
        ir, vis = load_pair(pair["ir_path"], pair["vis_path"])
        vis_clahe = clahe_luminance(vis)
        rows.append({"pair": pair["vis_path"], "method": "visible", "alpha": "",
                     "psnr": "", "ssim": "", "brisque": brisque(vis)})
        for alpha in ALPHAS:
            for method, v in (("fused", vis), ("clahe_fused", vis_clahe)):
                f = fuse(ir, v, alpha)
                p, s = psnr_ssim(f, vis)
                rows.append({"pair": pair["vis_path"], "method": method, "alpha": alpha,
                             "psnr": p, "ssim": s, "brisque": brisque(f)})

    with open(os.path.join(out_dir, "tn_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # mean and std per method/alpha, as in thesis Table 4.1
    summary = {}
    for r in rows:
        summary.setdefault((r["method"], r["alpha"]), []).append(r)
    with open(os.path.join(out_dir, "tn_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "alpha", "n", "psnr_mean", "psnr_std",
                    "ssim_mean", "ssim_std", "brisque_mean", "brisque_std"])
        for (method, alpha), rs in summary.items():
            def ms(key):
                vals = [float(r[key]) for r in rs if r[key] != ""]
                return (np.mean(vals), np.std(vals)) if vals else ("", "")
            w.writerow([method, alpha, len(rs), *ms("psnr"), *ms("ssim"), *ms("brisque")])
    print(f"{len(pairs)} pairs evaluated -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True, help="CSV with ir_path,vis_path columns")
    ap.add_argument("--out", default="results/tno")
    args = ap.parse_args()
    evaluate(args.pairs, args.out)
