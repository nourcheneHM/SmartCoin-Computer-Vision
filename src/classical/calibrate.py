"""Calibration des règles de classification sur les images de DÉVELOPPEMENT uniquement.

Images utilisées : photos à une seule pièce dont la classe est connue par le dossier
(data/raw/group/<classe>/ et data/raw/new/train/L1_<fond>/<classe>/). Seules les images où
la détection trouve exactement une pièce sont gardées.

Le script mesure les caractéristiques (features.py) et affiche, pour chaque règle, les
statistiques qui justifient les seuils de configs/classical.yaml :
- contraste bimétal (5 dinars contre les autres) ;
- seuil doré / argenté et coefficient de correction du fond (recherche sur grille) ;
- score de forme à 13 côtés (rappel 200 millimes / 2 dinars, faux positifs sur pièces rondes) ;
- échelle a priori s0 (pixels par mm, rapportée au plus petit côté) sur les photos L1.

Les seuils ne sont PAS écrits automatiquement dans la configuration : ils sont choisis
et justifiés à la main à partir de ces chiffres.

Usage :
    python -m src.classical.calibrate --out results/classical/calibration

Le jeu de test (data/raw/new/test/) n'est jamais lu par ce script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.classical.features import image_features
from src.classical.pipeline import detect_coins, list_images, load_config
from src.classical.preprocessing import load_image
from src.common.coin_classes import CLASS_NAMES, COIN_DIAMETERS_MM, COIN_METAL, POLYGONAL_CLASSES

L1_FOLDERS = ["L1_B1_tissu", "L1_B2_papier_blanc", "L1_B3_orange"]


def collect(cfg: dict, data_root: Path) -> pd.DataFrame:
    rows = []
    for cls in CLASS_NAMES:
        sources = [("group", p) for p in list_images(data_root / "group" / cls)]
        for folder in L1_FOLDERS:
            sources += [(folder, p) for p in list_images(data_root / "new/train" / folder / cls)]
        for src, path in sources:
            coins, small, _ = detect_coins(load_image(path), cfg)
            if len(coins) != 1:
                continue
            f = image_features(small, coins, cfg["detection"]["segmentation"]["border_ratio"])[0]
            rows.append(dict(cls=cls, source=src, file=path.name, min_side=min(small.shape[:2]),
                             r_hough=coins[0].r, **vars(f), bimetal=f.bimetal_contrast))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"Pièces mesurées : {len(df)}\n")

    print("== Contraste bimétal b*(anneau) - b*(centre) ==")
    is5 = df.cls == "5_dinars"
    print(f"5 dinars : min {df[is5].bimetal.min():.1f}, médiane {df[is5].bimetal.median():.1f}")
    print(f"autres   : max {df[~is5].bimetal.max():.1f}, médiane {df[~is5].bimetal.median():.1f}\n")

    print("== Doré / argenté : b - alpha * b_fond > seuil (hors 5 dinars) ==")
    plain = df[~is5]
    gold = plain.cls.map(COIN_METAL).eq("gold").to_numpy()
    results = []
    for alpha in np.arange(0, 1.001, 0.05):
        v = (plain.b - alpha * plain.b_background).to_numpy()
        for t in np.arange(-10, 30.01, 0.5):
            results.append((((v > t) != gold).sum(), round(alpha, 2), t))
    results.sort()
    for n_err, alpha, t in results[:5]:
        print(f"alpha {alpha:.2f}  seuil {t:5.1f}  erreurs {n_err}/{len(plain)}")
    v0 = plain.b.to_numpy()
    print(f"(sans correction, meilleur seuil : "
          f"{min(((v0 > t) != gold).sum() for t in np.arange(-10, 30.01, 0.5))} erreurs)\n")

    print("== Forme à 13 côtés (score de la 13e harmonique) ==")
    poly = df.cls.isin(POLYGONAL_CLASSES)
    for t in [2.5, 3.0, 3.5, 4.0]:
        print(f"seuil {t:.1f} : rappel 200 mill. {(df[df.cls == '200_millimes'].poly_score >= t).mean():.0%}"
              f", rappel 2 dinars {(df[df.cls == '2_dinars'].poly_score >= t).mean():.0%}"
              f", faux positifs pièces rondes {(df[~poly].poly_score >= t).sum()}/{(~poly).sum()}")

    print("\n== Échelle a priori (photos L1) ==")
    l1 = df[df.source.isin(L1_FOLDERS)]
    s_rel = l1.r_edge * 2 / l1.cls.map(COIN_DIAMETERS_MM) / l1.min_side
    print(f"pixels par mm / plus petit côté : médiane {s_rel.median():.4f} "
          f"(min {s_rel.min():.4f}, max {s_rel.max():.4f}, n = {len(l1)})")
    print(f"rayon affiné / rayon de Hough : médiane {(df.r_edge / df.r_hough).median():.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Calibration des règles de classification")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/classical/calibration"))
    parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    args = parser.parse_args()

    df = collect(load_config(args.config), args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "features_dev.csv", index=False)
    report(df)
    print(f"\nMesures enregistrées dans : {args.out / 'features_dev.csv'}")


if __name__ == "__main__":
    main()
