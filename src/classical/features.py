"""Extraction des caractéristiques de chaque pièce détectée (approche classique).

Trois familles de caractéristiques, choisies après mesure sur les images de développement
(voir src/classical/calibrate.py) :

1. **Couleur (espace Lab)** — indépendante de la distance de prise de vue.
   - a* : axe vert (-) / rouge (+) ; b* : axe bleu (-) / jaune (+). Le laiton (pièces dorées)
     a un b* nettement positif, le cupronickel (pièces argentées) un b* proche de 0.
   - Le fond coloré (papier orange) se reflète sur le métal et décale la balance des blancs
     du téléphone : on mémorise donc aussi le b* du fond pour corriger la couleur.
   - Pièce bimétallique (5 dinars) : centre argenté et anneau doré, donc
     b*(anneau) - b*(centre) est grand.
   - Les reflets spéculaires (pixels presque blancs) sont exclus ; on prend la médiane,
     robuste aux gravures sombres.

2. **Forme (13 côtés)** — les pièces de 200 millimes et de 2 dinars sont des
   tridécagones. On suit le bord de la pièce en coordonnées polaires (rayon du bord pour
   chaque angle). Pour un tridécagone, ce rayon oscille 13 fois sur un tour : on mesure
   l'énergie de la 13e harmonique de Fourier par rapport aux harmoniques voisines.
   Le bord est suivi par programmation dynamique (chemin continu de gradient maximal)
   pour ne pas « sauter » sur les gravures intérieures.

3. **Rayon affiné** — rayon moyen du bord ainsi suivi, plus précis que le rayon de Hough.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.classical.detection import CoinCandidate, background_lab

N_ANGLES = 720        # échantillons angulaires du profil de bord (0,5°)
N_RADII = 120         # échantillons radiaux entre 0,7 r et 1,25 r


@dataclass
class CoinFeatures:
    """Caractéristiques d'une pièce (Lab centré : a* = b* = 0 pour un gris neutre)."""

    L: float              # luminosité du disque intérieur (0-255)
    a: float              # a* du disque intérieur
    b: float              # b* du disque intérieur (jaune +)
    b_center: float       # b* du centre (0 à 0,40 r)
    b_ring: float         # b* de l'anneau (0,65 à 0,85 r)
    b_background: float   # b* du fond de l'image
    poly_score: float     # énergie de la 13e harmonique / harmoniques voisines
    r_edge: float         # rayon affiné (pixels)

    @property
    def bimetal_contrast(self) -> float:
        """Différence de « jaune » entre l'anneau et le centre (grande pour le 5 dinars)."""
        return self.b_ring - self.b_center

    def corrected_b(self, alpha: float) -> float:
        """b* corrigé de la dominante colorée du fond : b - alpha * b_fond."""
        return self.b - alpha * self.b_background


def _ring_mask(shape: tuple[int, int], c: CoinCandidate, r_in: float, r_out: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - c.x) ** 2 + (yy - c.y) ** 2
    return (d2 >= (r_in * c.r) ** 2) & (d2 <= (r_out * c.r) ** 2)


def _median_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = lab[mask]
    px = px[px[:, 0] < 240]          # exclut les reflets spéculaires
    if len(px) == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.median(px, axis=0)


def _best_path(cost: np.ndarray, step: int = 2) -> np.ndarray:
    """Chemin de coût total maximal à travers `cost` (angles x rayons).

    Entre deux angles consécutifs, l'indice de rayon varie d'au plus `step` :
    le bord suivi est continu (programmation dynamique, comme un algorithme de Viterbi).
    """
    n, m = cost.shape
    acc = cost[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    idx = np.arange(m)
    for i in range(1, n):
        best = np.full(m, -np.inf)
        arg = np.zeros(m, dtype=np.int32)
        for d in range(-step, step + 1):
            shifted = np.roll(acc, d)
            if d > 0:
                shifted[:d] = -np.inf
            elif d < 0:
                shifted[d:] = -np.inf
            better = shifted > best
            best[better] = shifted[better]
            arg[better] = (idx - d)[better]
        acc = best + cost[i]
        back[i] = arg
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(acc.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def edge_profile(gx: np.ndarray, gy: np.ndarray, c: CoinCandidate) -> np.ndarray:
    """Rayon du bord (en pixels) pour chaque angle, suivi par programmation dynamique."""
    theta = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    radii = np.linspace(0.70 * c.r, 1.25 * c.r, N_RADII)
    xs = (c.x + np.outer(np.cos(theta), radii)).astype(np.float32)
    ys = (c.y + np.outer(np.sin(theta), radii)).astype(np.float32)
    # composante radiale du gradient : forte sur le bord de la pièce
    gr = (cv2.remap(gx, xs, ys, cv2.INTER_LINEAR) * np.cos(theta)[:, None]
          + cv2.remap(gy, xs, ys, cv2.INTER_LINEAR) * np.sin(theta)[:, None])
    gr = np.abs(gr)
    gr /= gr.max() + 1e-6
    return radii[_best_path(gr)]


def polygon_score(profile: np.ndarray, sides: int = 13) -> float:
    """Énergie de l'harmonique `sides` du profil, relative aux harmoniques voisines."""
    spectrum = np.abs(np.fft.rfft(profile - profile.mean()))
    neighbours = np.r_[spectrum[sides - 5:sides - 1], spectrum[sides + 2:sides + 17]]
    return float(spectrum[sides] / max(np.median(neighbours), 1e-9))


def image_features(img_bgr: np.ndarray, coins: list[CoinCandidate],
                   border_ratio: float) -> list[CoinFeatures]:
    """Caractéristiques de toutes les pièces d'une image (déjà redimensionnée)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] -= 128.0             # a* et b* centrés sur 0 (OpenCV les décale de 128)
    b_bg = float(background_lab(lab, border_ratio)[2])

    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    shape = lab.shape[:2]
    feats = []
    for c in coins:
        inner = _median_lab(lab, _ring_mask(shape, c, 0.0, 0.85))
        center = _median_lab(lab, _ring_mask(shape, c, 0.0, 0.40))
        ring = _median_lab(lab, _ring_mask(shape, c, 0.65, 0.85))
        profile = edge_profile(gx, gy, c)
        feats.append(CoinFeatures(
            L=float(inner[0]), a=float(inner[1]), b=float(inner[2]),
            b_center=float(center[2]), b_ring=float(ring[2]), b_background=b_bg,
            poly_score=polygon_score(profile), r_edge=float(np.median(profile)),
        ))
    return feats
