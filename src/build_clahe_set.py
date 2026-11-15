"""Build the CLAHE thermal training set (thesis Algorithm 1, Appendix B.1).

For every image in the existing raw set, the native 640x512 FLIR thermal frame
is loaded, enhanced with CLAHE (clip limit 2.0, 8x8 tiles), replicated to three
channels and resized to 640x640 with the same geometry as the raw set. Labels
are copied unchanged, so CLAHE is the only difference between the two sets.

Usage:
    python src/build_clahe_set.py --src "C:/Thesis dataset/FLIR_ADAS_v2" \
        --raw "C:/FLIR_FUSED" --dst "C:/FLIR_CLAHE"
"""
import argparse
import glob
import os
import shutil

import cv2

CLIP, TILES, SIZE = 2.0, (8, 8), (640, 640)


def index_originals(src_root):
    """Map file name -> path for every native thermal frame under src_root.

    Added for this repository: the thesis listing calls find_original()
    without showing it.
    """
    index = {}
    for path in glob.glob(os.path.join(src_root, "**", "*.jpg"), recursive=True):
        if "thermal" in path.replace("\\", "/").lower():
            index[os.path.basename(path)] = path
    return index


def lap_var(img):
    """Local-contrast measure used to verify the CLAHE set (variance of the Laplacian)."""
    return cv2.Laplacian(img, cv2.CV_64F).var()


def build(src, raw, dst):
    clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=TILES)
    originals = index_originals(src)

    for split in ["train", "val"]:
        os.makedirs(f"{dst}/images/{split}", exist_ok=True)
        raw_images = sorted(glob.glob(f"{raw}/images/{split}/*.jpg"))
        missing = 0
        for raw_path in raw_images:
            name = os.path.basename(raw_path)
            src_path = originals.get(name)
            if src_path is None:
                missing += 1
                continue
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)  # native 640x512 frame
            enh = clahe.apply(img)                             # CLAHE on intensity
            enh3 = cv2.merge([enh, enh, enh])                  # replicate to 3 channels
            out = cv2.resize(enh3, SIZE)                       # same stretch as raw set
            cv2.imwrite(f"{dst}/images/{split}/{name}", out)

        # labels copied unchanged -> identical split and boxes
        shutil.copytree(f"{raw}/labels/{split}", f"{dst}/labels/{split}", dirs_exist_ok=True)

        written = len(glob.glob(f"{dst}/images/{split}/*.jpg"))
        print(f"{split}: raw={len(raw_images)} clahe={written} missing_originals={missing}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="original FLIR ADAS v2 download")
    ap.add_argument("--raw", required=True, help="existing raw 640x640 set")
    ap.add_argument("--dst", required=True, help="output folder for the CLAHE set")
    args = ap.parse_args()
    build(args.src, args.raw, args.dst)
