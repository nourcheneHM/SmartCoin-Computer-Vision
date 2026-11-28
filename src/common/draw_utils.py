"""Dessin des détections sur une image (utilisé par predict.py et l'interface Streamlit)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.common.labels import SHORT_LABEL, UNKNOWN

# Couleurs BGR : teintes chaudes pour les millimes, froides pour les dinars
PALETTE = {
    "10_millimes": (60, 90, 200), "20_millimes": (40, 150, 230), "50_millimes": (30, 190, 220),
    "100_millimes": (60, 190, 120), "200_millimes": (170, 170, 40), "500_millimes": (200, 120, 40),
    "1_dinar": (200, 70, 90), "2_dinars": (160, 60, 160), "5_dinars": (110, 60, 220),
    UNKNOWN: (140, 140, 140),
}


def draw_detections(img_bgr: np.ndarray, det: pd.DataFrame, show_score: bool = True) -> np.ndarray:
    """Retourne une copie annotée. `det` : x_min, y_min, x_max, y_max, class_name[, score]."""
    out = img_bgr.copy()
    scale = max(out.shape[:2]) / 1024.0
    thick = max(2, int(round(2 * scale)))
    font_scale = max(0.5, 0.6 * scale)
    for _, d in det.iterrows():
        cls = d["class_name"]
        color = PALETTE.get(cls, PALETTE[UNKNOWN])
        p1, p2 = (int(d["x_min"]), int(d["y_min"])), (int(d["x_max"]), int(d["y_max"]))
        cv2.rectangle(out, p1, p2, color, thick)
        text = SHORT_LABEL.get(cls, "?")
        if show_score and "score" in d and pd.notna(d["score"]) and float(d["score"]) < 1.0:
            text += f" {float(d['score']):.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        y0 = max(p1[1] - th - base - 4, 0)
        cv2.rectangle(out, (p1[0], y0), (p1[0] + tw + 6, y0 + th + base + 4), color, -1)
        cv2.putText(out, text, (p1[0] + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return out
