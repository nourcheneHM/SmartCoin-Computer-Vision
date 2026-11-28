"""Annotation automatique des photos à UNE pièce (classe = nom du dossier).

Les 230 photos initiales (data/raw/group/<classe>/) et les 27 photos L1
(data/raw/new/train/L1_*/<classe>/) contiennent une seule pièce dont la valeur est donnée
par le dossier. La boîte vient de la DÉTECTION de l'approche classique (on ignore sa
classification). On garde la plus grande détection de chaque image.

    python -m src.data.auto_label_single --data-root data

Sorties :
  data/annotations/annotations_auto_single.csv   boîtes acceptées automatiquement
  data/annotations/needs_review.csv              images sans détection ou ambiguës
                                                 -> à annoter à la main dans Label Studio
  results/inspection/auto_label_<classe>.jpg     planches de contrôle visuel (À REGARDER)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.common.labels import image_key, parse_class_strict
from src.evaluation.classical_runner import IMG_EXT, run_on_images, standardize

COLS = ["image_id", "file_name", "coin_id", "x_min", "y_min", "x_max", "y_max",
        "class_name", "occluded"]


def collect(data_root: Path):
    items, ignored = [], set()
    for sub in (data_root / "raw" / "group", data_root / "raw" / "new" / "train"):
        for p in sorted(sub.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel = p.relative_to(data_root / "raw")
            if any("multiple" in part.lower() for part in rel.parts):
                continue
            cls = parse_class_strict(p.parent.name)
            if cls is None:
                if not re.fullmatch(r"L[1-4]", p.parent.name):
                    ignored.add(str(p.parent.relative_to(data_root)))
                continue
            items.append((p, cls))
    return items, sorted(ignored)


def area(r):
    return (r.x_max - r.x_min) * (r.y_max - r.y_min)


def contact_sheet(rows, data_root: Path, out: Path, cols=6, cell=(192, 256)) -> None:
    tiles = []
    for r in rows:
        img = cv2.imread(str(data_root / r["file_name"]))
        if img is None:
            continue
        t = max(4, img.shape[1] // 120)
        cv2.rectangle(img, (int(r["x_min"]), int(r["y_min"])), (int(r["x_max"]), int(r["y_max"])),
                      (0, 0, 255), t)
        s = min(cell[0] / img.shape[1], cell[1] / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        tile = np.full((cell[1], cell[0], 3), 255, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full((cell[1], cell[0], 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/annotations")
    ap.add_argument("--sheets-dir", default="results/inspection")
    ap.add_argument("--from-csv", help="predictions.csv déjà produit par l'approche classique")
    a = ap.parse_args()
    data_root = Path(a.data_root)

    items, ignored = collect(data_root)
    if not items:
        sys.exit("Aucune photo à une pièce trouvée (vérifier --data-root et les noms de dossiers).")
    if ignored:
        print("Dossiers ignorés (nom de classe non reconnu) :", ignored)
    print(f"{len(items)} photos à une pièce. Détection par l'approche classique...")

    if a.from_csv:
        det, _ = standardize(pd.read_csv(a.from_csv))
    else:
        det, ms = run_on_images([p for p, _ in items])
        print(f"Approche classique : {ms:.0f} ms/image")

    ok, review = [], []
    for p, cls in items:
        key = image_key(p)
        d = det[det["image_id"] == key]
        rel = p.relative_to(data_root).as_posix()
        if d.empty:
            review.append({"image_id": key, "file_name": rel, "class_name": cls, "reason": "aucune détection"})
            continue
        d = d.assign(a=d.apply(area, axis=1)).sort_values("a", ascending=False)
        if len(d) > 1 and d.iloc[1]["a"] > 0.3 * d.iloc[0]["a"]:
            review.append({"image_id": key, "file_name": rel, "class_name": cls, "reason": "plusieurs grandes détections"})
            continue
        b = d.iloc[0]
        ok.append({"image_id": key, "file_name": rel, "coin_id": 1,
                   "x_min": int(b.x_min), "y_min": int(b.y_min), "x_max": int(b.x_max),
                   "y_max": int(b.y_max), "class_name": cls, "occluded": 0})

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ok, columns=COLS)
    df.to_csv(out / "annotations_auto_single.csv", index=False)
    pd.DataFrame(review, columns=["image_id", "file_name", "class_name", "reason"]).to_csv(
        out / "needs_review.csv", index=False)
    for cls, g in df.groupby("class_name"):
        contact_sheet(g.to_dict("records"), data_root, Path(a.sheets_dir) / f"auto_label_{cls}.jpg")
    print(f"[OK] {len(df)} acceptées -> {out / 'annotations_auto_single.csv'}")
    print(f"[A FAIRE] {len(review)} à revoir à la main -> {out / 'needs_review.csv'}")
    print(f"Planches de contrôle : {a.sheets_dir}/auto_label_*.jpg (vérifier que chaque cadre rouge entoure la pièce)")


if __name__ == "__main__":
    main()
