"""Fonctions d'affichage communes aux deux approches."""

from __future__ import annotations

import cv2
import numpy as np


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
