"""Pipeline de l'approche classique (sans IA).

Étapes : prétraitement -> détection (Hough + segmentation) -> caractéristiques (couleur,
forme, rayon) -> classification par règles -> comptage et montant total.

Usage :
    python -m src.classical.pipeline --input data/raw/new/train/L2
    python -m src.classical.pipeline --input chemin/vers/image.jpg --out results/classical/predictions

Sorties dans --out : une image annotée par photo et `predictions.csv` (une ligne par pièce,
coordonnées dans l'image ORIGINALE, au même format que la vérité terrain).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml

from src.classical.classification import CoinPrediction, classify_coins
from src.classical.detection import CoinCandidate, detect
from src.classical.features import image_features
from src.classical.preprocessing import load_image, resize_max_side, to_gray_enhanced
from src.common.amount import count_by_class, format_tnd, total_millimes
from src.common.coin_classes import UNKNOWN_LABEL
from src.common.visualization import draw_predictions

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config(path: str | Path = "configs/classical.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_coins(img_bgr, cfg: dict) -> tuple[list[CoinCandidate], object, float]:
    """Retourne (pièces détectées, image redimensionnée, facteur d'échelle)."""
    small, scale = resize_max_side(img_bgr, cfg["preprocessing"]["max_side"])
    gray = to_gray_enhanced(small, cfg["preprocessing"])
    coins = detect(small, gray, cfg["detection"])
    return coins, small, scale


def predict(img_bgr, cfg: dict) -> tuple[list[CoinPrediction], object, float]:
    """Pipeline complet sur une image : (prédictions, image redimensionnée, facteur d'échelle)."""
    coins, small, scale = detect_coins(img_bgr, cfg)
    feats = image_features(small, coins, cfg["detection"]["segmentation"]["border_ratio"])
    preds = classify_coins(coins, feats, cfg["classification"], min(small.shape[:2]))
    return preds, small, scale


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Approche classique SmartCoin")
    parser.add_argument("--input", type=Path, required=True, help="image ou dossier")
    parser.add_argument("--out", type=Path, default=Path("results/classical/predictions"))
    parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in list_images(args.input):
        t0 = time.perf_counter()
        preds, small, scale = predict(load_image(path), cfg)
        ms = (time.perf_counter() - t0) * 1000

        labels = [p.label for p in preds]
        cv2.imwrite(str(args.out / f"{path.parent.name}__{path.name}"),
                    draw_predictions(small, preds))
        for i, p in enumerate(preds):
            x0, y0, x1, y1 = p.coin.bbox
            rows.append(dict(image_id=path.stem, coin_id=i,
                             x_min=round(x0 / scale), y_min=round(y0 / scale),
                             x_max=round(x1 / scale), y_max=round(y1 / scale),
                             class_name=p.label, best_guess=p.best_guess,
                             confidence=round(p.confidence, 3), metal=p.metal,
                             polygonal=int(p.polygonal), time_ms=round(ms)))
        counts = {c: n for c, n in count_by_class(labels).items() if n}
        print(f"{path.parent.name}/{path.name}: {len(preds)} pièce(s), "
              f"{format_tnd(total_millimes(labels))}, "
              f"{counts.get(UNKNOWN_LABEL, 0)} inconnue(s), {ms:.0f} ms")
        print("    " + ", ".join(f"{c}:{n}" for c, n in counts.items()))

    pd.DataFrame(rows).to_csv(args.out / "predictions.csv", index=False)
    print(f"\nRésultats enregistrés dans : {args.out}")


if __name__ == "__main__":
    main()
