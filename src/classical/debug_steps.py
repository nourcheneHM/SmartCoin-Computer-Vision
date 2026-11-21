"""Visualisation pas à pas de la détection classique (outil pédagogique et de débogage).

Pour chaque image, produit un panneau de 8 vignettes montrant chaque étape :

    1. Image redimensionnée        2. Gris + CLAHE + flou médian   3. Cercles de Hough
    4. Distance au fond (Lab)      5. Seuillage d'Otsu             6. Morphologie + trous remplis
    7. Masque final + transformée de distance                      8. Résultat fusionné

Code couleur du résultat : vert = cercle de Hough, orange = pièce ajoutée par la segmentation.

Usage :
    python -m src.classical.debug_steps --input data/raw/new/train/L4
    python -m src.classical.debug_steps --input chemin/vers/image.jpg --out results/classical/debug

Rappel : ne pas régler les paramètres en regardant les images de data/raw/new/test/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.classical.detection import (detect_hough, detect_segmentation, foreground_mask,
                                     merge_detections)
from src.classical.pipeline import list_images, load_config
from src.classical.preprocessing import load_image, resize_max_side, to_gray_enhanced

GREEN, ORANGE = (0, 200, 0), (0, 140, 255)


def _bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def _draw(img: np.ndarray, circles, color) -> np.ndarray:
    out = _bgr(img)
    t = max(2, round(min(out.shape[:2]) / 250))
    for c in circles:
        cv2.circle(out, (round(c.x), round(c.y)), round(c.r), color, t)
    return out


def _tile(img: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    """Redimensionne une vignette (en conservant les proportions) et ajoute un titre."""
    w, h = size
    img = _bgr(img)
    s = min(w / img.shape[1], (h - 30) / img.shape[0])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    tile = np.full((h, w, 3), 255, np.uint8)
    tile[30:30 + img.shape[0], :img.shape[1]] = img
    cv2.putText(tile, title, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def steps_panel(path: Path, cfg: dict) -> tuple[np.ndarray, str]:
    """Construit le panneau des étapes et un résumé texte pour une image."""
    det = cfg["detection"]
    small, _ = resize_max_side(load_image(path), cfg["preprocessing"]["max_side"])
    gray = to_gray_enhanced(small, cfg["preprocessing"])

    hough = detect_hough(gray, det["hough"])
    seg = detect_segmentation(small, det["segmentation"])
    merged = merge_detections(hough, seg)
    added = [c for c in merged if c not in hough]

    steps: dict = {}
    foreground_mask(small, det["segmentation"], steps)
    dt = cv2.distanceTransform(steps["masque_final"], cv2.DIST_L2, 5)
    dt_vis = cv2.applyColorMap(cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                               cv2.COLORMAP_JET)
    final = _draw(_draw(small, hough, GREEN), added, ORANGE)

    tiles = [
        (small, "1. Image (1024 px)"),
        (gray, "2. Gris + CLAHE + median"),
        (_draw(gray, hough, GREEN), f"3. Hough : {len(hough)} cercle(s)"),
        (steps["distance_fond"], "4. Distance au fond (Lab)"),
        (steps["otsu"], "5. Seuillage d'Otsu"),
        (steps["morphologie"], "6. Morphologie + trous"),
        (_draw(dt_vis, seg, (255, 255, 255)), f"7. Bords retires + distance : {len(seg)}"),
        (final, f"8. Fusion : {len(merged)} piece(s)"),
    ]
    size = (330, 420)
    row1 = np.hstack([_tile(i, t, size) for i, t in tiles[:4]])
    row2 = np.hstack([_tile(i, t, size) for i, t in tiles[4:]])
    summary = (f"{path.parent.name}/{path.name}: Hough {len(hough)}, segmentation {len(seg)}, "
               f"ajoutées par la segmentation {len(added)}, total {len(merged)}")
    return np.vstack([row1, row2]), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation pas à pas de la détection")
    parser.add_argument("--input", type=Path, required=True, help="image ou dossier")
    parser.add_argument("--out", type=Path, default=Path("results/classical/debug"))
    parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in list_images(args.input):
        panel, summary = steps_panel(path, cfg)
        cv2.imwrite(str(args.out / f"{path.parent.name}__{path.stem}_etapes.jpg"), panel)
        print(summary)
    print(f"\nPanneaux enregistrés dans : {args.out}")


if __name__ == "__main__":
    main()
