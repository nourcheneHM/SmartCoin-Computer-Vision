"""Détection des pièces par transformée de Hough pour les cercles.

Principe (cours S2 + Hough) : les contours de l'image (gradient, Canny) « votent » pour
des centres de cercles candidats. Un cercle est retenu si suffisamment de points de
contour sont situés sur sa circonférence.

On utilise la variante `cv2.HOUGH_GRADIENT_ALT` :
- `param1` : seuil haut du détecteur de Canny interne ;
- `param2` : score de « perfection » du cercle entre 0 et 1 (1 = cercle parfait).
  Plus il est haut, moins il y a de faux cercles, mais plus on risque d'en rater.

Les rayons et la distance minimale entre centres sont exprimés en proportion du plus
petit côté de l'image, pour rester valables quelle que soit la résolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CoinCandidate:
    """Pièce détectée : cercle (x, y, r) en pixels de l'image traitée."""

    x: float
    y: float
    r: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Boîte englobante (x_min, y_min, x_max, y_max), format de la vérité terrain."""
        return (round(self.x - self.r), round(self.y - self.r),
                round(self.x + self.r), round(self.y + self.r))


def detect_hough(gray: np.ndarray, cfg: dict) -> list[CoinCandidate]:
    """Détecte les pièces dans une image en niveaux de gris prétraitée."""
    min_side = min(gray.shape[:2])
    min_r = max(5, round(cfg["min_radius_ratio"] * min_side))
    max_r = round(cfg["max_radius_ratio"] * min_side)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT_ALT,
        dp=cfg["dp"],
        minDist=cfg["min_dist_ratio"] * min_side,
        param1=cfg["param1"],
        param2=cfg["param2"],
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    candidates = [CoinCandidate(float(x), float(y), float(r)) for x, y, r in circles[0]]
    return remove_nested(candidates)


def foreground_mask(img_bgr: np.ndarray, cfg: dict, steps: dict | None = None) -> np.ndarray:
    """Masque binaire des pièces par soustraction de la couleur du fond.

    1. La couleur du fond est estimée par la médiane (espace Lab) d'une bande sur les bords
       de l'image : les pièces sont au centre, les bords sont du fond.
    2. Chaque pixel reçoit sa distance colorimétrique au fond (distance euclidienne en Lab,
       le canal L étant pondéré par `l_weight` pour limiter l'effet des ombres).
    3. Seuillage automatique d'Otsu, puis morphologie : ouverture (supprime le bruit de
       texture), fermeture (consolide les bords), remplissage des trous internes.

    Si un dictionnaire `steps` est fourni, les images intermédiaires y sont enregistrées
    (utilisé par src/classical/debug_steps.py pour la visualisation).
    """
    if steps is None:
        steps = {}
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, round(cfg["border_ratio"] * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    diff = lab - bg
    diff[..., 0] *= cfg["l_weight"]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.clip(dist * 255.0 / max(dist.max(), 1e-6), 0, 255).astype(np.uint8)
    dist = cv2.medianBlur(dist, int(cfg["blur_ksize"]) | 1)
    steps["distance_fond"] = dist
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["otsu"] = mask

    k = int(cfg["morph_ksize"]) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    steps["morphologie"] = filled
    if cfg.get("drop_border_regions", True):
        filled = drop_border_regions(filled)
    steps["masque_final"] = filled
    return filled


def drop_border_regions(mask: np.ndarray) -> np.ndarray:
    """Supprime les régions du masque qui touchent le bord de l'image.

    Hypothèse du protocole d'acquisition : chaque pièce est entièrement dans le cadre.
    Les régions touchant le bord sont donc des ombres ou des dégradés d'éclairage du fond.
    Limite : une pièce coupée par le bord de l'image ne sera pas détectée.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[:2]
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, bw, bh = stats[i, :4]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            out[labels == i] = 255
    return out


def detect_segmentation(img_bgr: np.ndarray, cfg: dict) -> list[CoinCandidate]:
    """Détecte les pièces par segmentation + transformée de distance.

    Dans la transformée de distance du masque, chaque pixel vaut sa distance au fond.
    Le centre d'une pièce est un maximum local et sa valeur est le rayon (cercle inscrit).
    Deux pièces qui se touchent forment une seule tache, mais deux maxima distincts :
    elles sont donc séparées sans supposer une forme parfaitement circulaire.
    """
    mask = foreground_mask(img_bgr, cfg)
    min_side = min(mask.shape[:2])
    min_r = max(5, round(cfg["min_radius_ratio"] * min_side))
    max_r = round(cfg["max_radius_ratio"] * min_side)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    k = 2 * min_r + 1
    local_max = (dist == cv2.dilate(dist, np.ones((k, k), np.uint8))) & (dist >= min_r)
    ys, xs = np.nonzero(local_max)
    cands = [CoinCandidate(float(x), float(y), float(dist[y, x]))
             for x, y in zip(xs, ys) if dist[y, x] <= max_r]
    return remove_nested(cands)


def detect(img_bgr: np.ndarray, gray: np.ndarray, cfg: dict) -> list[CoinCandidate]:
    """Point d'entrée : choisit la méthode selon `cfg['method']`."""
    if cfg["method"] == "hough":
        return detect_hough(gray, cfg["hough"])
    if cfg["method"] == "segmentation":
        return detect_segmentation(img_bgr, cfg["segmentation"])
    if cfg["method"] == "hybrid":
        return merge_detections(detect_hough(gray, cfg["hough"]),
                                detect_segmentation(img_bgr, cfg["segmentation"]))
    raise ValueError(f"Méthode de détection inconnue : {cfg['method']}")


def merge_detections(hough: list[CoinCandidate],
                     seg: list[CoinCandidate]) -> list[CoinCandidate]:
    """Fusionne deux détecteurs complémentaires.

    - Hough (contours) : fiable pour les pièces rondes, même de couleur proche du fond
      (pièce argentée sur fond blanc), mais rate les pièces festonnées (200 millimes)
      et se trompe sur les fonds texturés.
    - Segmentation (couleur) : gère toutes les formes et les fonds texturés colorés,
      mais ne capture qu'un fragment d'une pièce de couleur proche du fond.

    Les cercles de Hough sont gardés ; une détection par segmentation est ajoutée
    seulement si son centre n'est à l'intérieur d'aucun cercle de Hough.
    """
    merged = list(hough)
    for s in seg:
        if all(np.hypot(s.x - h.x, s.y - h.y) > h.r for h in hough):
            merged.append(s)
    return remove_nested(merged)


def remove_nested(cands: list[CoinCandidate]) -> list[CoinCandidate]:
    """Supprime les cercles dont le centre est à l'intérieur d'un cercle plus grand.

    Cas typique : le cœur argenté d'une pièce de 5 dinars (bimétallique) ou le motif
    central d'une pièce, détecté comme un second cercle plus petit.

    Un cercle est supprimé si son centre est proche du centre d'un cercle plus grand,
    ou s'il est presque entièrement contenu dans celui-ci (distance + rayon <= 1,1 x R).
    """
    kept: list[CoinCandidate] = []
    for c in sorted(cands, key=lambda c: c.r, reverse=True):
        if all(_distinct(c, k) for k in kept):
            kept.append(c)
    return kept


def _distinct(small: CoinCandidate, big: CoinCandidate) -> bool:
    d = np.hypot(small.x - big.x, small.y - big.y)
    return d > 0.6 * big.r and d + small.r > 1.1 * big.r
