"""Pipeline de l'approche classique (sans IA).

Étapes : prétraitement -> détection (HoughCircles) -> [caractéristiques -> règles ->
comptage et montant : étapes suivantes, pas encore implémentées].

Usage :
    python -m src.classical.pipeline --input data/raw/new/train/L2 --out results/classical/detection
    python -m src.classical.pipeline --input chemin/vers/image.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from src.classical.detection import CoinCandidate, detect
from src.classical.preprocessing import load_image, resize_max_side, to_gray_enhanced
from src.common.visualization import draw_circles

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


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Approche classique SmartCoin (détection)")
    parser.add_argument("--input", type=Path, required=True, help="image ou dossier")
    parser.add_argument("--out", type=Path, default=Path("results/classical/detection"))
    parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in list_images(args.input):
        t0 = time.perf_counter()
        coins, small, _ = detect_coins(load_image(path), cfg)
        ms = (time.perf_counter() - t0) * 1000
        cv2.imwrite(str(args.out / f"{path.parent.name}__{path.name}"), draw_circles(small, coins))
        print(f"{path.parent.name}/{path.name}: {len(coins)} pièce(s), {ms:.0f} ms")


if __name__ == "__main__":
    main()
