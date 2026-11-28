"""Inférence YOLOv8 : détecte et classe les pièces, écrit le FORMAT STANDARD.

    # Jeu de test commun (même liste d'images que l'approche classique)
    python -m src.ai.predict --manifest evaluation/test_manifest.csv --data-root data \
        --weights models/yolov8_smartcoin_best.pt --out results/ai/test

    # Dossier quelconque
    python -m src.ai.predict --input data/yolo/images/val --out results/ai/val

Sorties : predictions_std.csv, timings.csv, annotated/*.jpg
Le seuil de confiance (--conf) se choisit sur la VALIDATION, jamais sur le test.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

from src.common.draw_utils import draw_detections
from src.common.labels import image_key, normalize_class

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STD_COLS = ["image_id", "x_min", "y_min", "x_max", "y_max", "class_name", "score"]


def load_model(weights: str | Path):
    from ultralytics import YOLO
    return YOLO(str(weights))


def predict_image(model, image, conf: float = 0.25, imgsz: int = 1024,
                  nms_iou: float = 0.45) -> pd.DataFrame:
    """`image` : chemin ou tableau BGR. Retourne les détections (sans la colonne image_id)."""
    res = model.predict(source=image, conf=conf, imgsz=imgsz, iou=nms_iou, verbose=False)[0]
    boxes = res.boxes
    rows = []
    for xyxy, c, k in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                          boxes.cls.cpu().numpy()):
        rows.append({"x_min": float(xyxy[0]), "y_min": float(xyxy[1]),
                     "x_max": float(xyxy[2]), "y_max": float(xyxy[3]),
                     "class_name": normalize_class(model.names[int(k)]), "score": float(c)})
    return pd.DataFrame(rows, columns=STD_COLS[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", default="models/yolov8_smartcoin_best.pt")
    ap.add_argument("--manifest")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--no-images", action="store_true", help="ne pas écrire les images annotées")
    a = ap.parse_args()

    if a.manifest:
        man = pd.read_csv(a.manifest)
        paths = [Path(a.data_root) / f for f in man["file_name"]]
    elif a.input:
        paths = sorted(p for p in Path(a.input).rglob("*") if p.suffix.lower() in IMG_EXT)
    else:
        sys.exit("Donner --manifest ou --input.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"Images introuvables (ex. {missing[0]}).")

    model = load_model(a.weights)
    predict_image(model, str(paths[0]), a.conf, a.imgsz, a.nms_iou)  # échauffement (non chronométré)

    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    all_det, timings = [], []
    for p in paths:
        t0 = time.perf_counter()
        det = predict_image(model, str(p), a.conf, a.imgsz, a.nms_iou)
        timings.append({"image_id": image_key(p), "time_ms": (time.perf_counter() - t0) * 1000})
        det.insert(0, "image_id", image_key(p))
        all_det.append(det)
        if not a.no_images:
            cv2.imwrite(str(out / "annotated" / f"{image_key(p)}.jpg"),
                        draw_detections(cv2.imread(str(p)), det))

    pred = pd.concat(all_det, ignore_index=True)[STD_COLS]
    pred.to_csv(out / "predictions_std.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    print(f"[OK] {len(pred)} détections sur {len(paths)} images "
          f"({pd.DataFrame(timings)['time_ms'].mean():.0f} ms/image) -> {out}")


if __name__ == "__main__":
    main()
