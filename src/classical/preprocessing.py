"""Prétraitement des images pour l'approche classique.

Étapes :
1. lecture (OpenCV, BGR) avec gestion d'erreur ;
2. redimensionnement (le plus grand côté ramené à `max_side` pixels) pour que les
   paramètres soient comparables d'une photo à l'autre et le traitement plus rapide ;
3. conversion en niveaux de gris, CLAHE (contraste local) puis flou (réduction du bruit
   et de la texture du fond, qui créerait de faux contours).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Charge une image en BGR. Lève une erreur explicite si le fichier est illisible."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image illisible ou introuvable : {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Redimensionne pour que le plus grand côté vaille `max_side`.

    Retourne l'image et le facteur d'échelle appliqué (pour reconvertir les coordonnées
    vers l'image originale lors de l'évaluation).
    """
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img, 1.0
    resized = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray_enhanced(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Image en niveaux de gris, contraste local rehaussé (CLAHE) puis lissée."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe_cfg = cfg.get("clahe", {})
    if clahe_cfg.get("enabled", False):
        clahe = cv2.createCLAHE(
            clipLimit=clahe_cfg.get("clip_limit", 2.0),
            tileGridSize=tuple(clahe_cfg.get("tile_grid_size", (8, 8))),
        )
        gray = clahe.apply(gray)

    k = int(cfg.get("blur_ksize", 5)) | 1  # taille de noyau impaire obligatoire
    if cfg.get("blur", "median") == "median":
        # Médian : supprime la texture fine (tissu) en gardant les bords nets des pièces.
        gray = cv2.medianBlur(gray, k)
    else:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray
