"""Reference-configuration training for YOLOv8n / YOLO11n (thesis Appendix B.3, Table 3.3).

Examples:
    python src/train_ultralytics.py --weights yolov8n.pt --data configs/flir_clahe.yaml --name v8n_clahe_true
    python src/train_ultralytics.py --weights yolo11n.pt --data configs/flir.yaml --name v11n_raw
    # resolution experiment (thesis Section 4.2.6)
    python src/train_ultralytics.py --weights yolov8n.pt --data configs/flir.yaml --name v8n_raw_768 --hires
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt", help="COCO-pretrained weights")
    ap.add_argument("--data", default="configs/flir.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hires", action="store_true", help="768 px, batch 8, 100 epochs")
    args = ap.parse_args()

    cfg = dict(epochs=60, batch=16, imgsz=640, close_mosaic=10)
    if args.hires:
        cfg = dict(epochs=100, batch=8, imgsz=768, close_mosaic=15, patience=30)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        optimizer="SGD", cos_lr=True,
        copy_paste=0.1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, fliplr=0.5,
        name=args.name, exist_ok=True,
        **cfg,
    )


if __name__ == "__main__":
    main()
