"""Fonctions d'affichage communes aux deux approches."""

from __future__ import annotations

import cv2
import numpy as np


SHORT_NAMES = {"10_millimes": "10", "20_millimes": "20", "50_millimes": "50",
               "100_millimes": "100", "200_millimes": "200", "500_millimes": "500",
               "1_dinars": "1D", "2_dinars": "2D", "5_dinars": "5D"}
METAL_COLORS = {"gold": (0, 200, 255), "silver": (200, 200, 200), "bimetal": (255, 0, 255)}


def draw_predictions(img_bgr: np.ndarray, preds) -> np.ndarray:
    """Dessine chaque pièce classée : couleur du cercle = métal détecté,
    texte = classe (« ? » + meilleure hypothèse si inconnue) et confiance."""
    out = img_bgr.copy()
    t = max(2, round(min(out.shape[:2]) / 300))
    for p in preds:
        c = p.coin
        color = METAL_COLORS.get(p.metal, (0, 255, 0))
        cv2.circle(out, (round(c.x), round(c.y)), round(c.r), color, t)
        name = SHORT_NAMES.get(p.best_guess, p.best_guess)
        text = f"{name} {p.confidence:.2f}" if p.label != "unknown" else f"?{name} {p.confidence:.2f}"
        org = (round(c.x - c.r * 0.6), round(c.y))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_circles(img_bgr: np.ndarray, circles, labels: list[str] | None = None) -> np.ndarray:
    """Dessine les cercles détectés (objets avec attributs x, y, r) et leurs étiquettes."""
    out = img_bgr.copy()
    thickness = max(2, round(min(out.shape[:2]) / 300))
    for i, c in enumerate(circles):
        center = (round(c.x), round(c.y))
        cv2.circle(out, center, round(c.r), (0, 255, 0), thickness)
        cv2.circle(out, center, thickness, (0, 0, 255), -1)
        text = labels[i] if labels else str(i)
        cv2.putText(out, text, (round(c.x - c.r), round(c.y - c.r) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
