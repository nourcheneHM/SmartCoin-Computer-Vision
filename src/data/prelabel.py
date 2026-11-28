"""Prépare les tâches à importer dans Label Studio.

Deux usages :

  # Images d'ENTRAÎNEMENT multi-pièces : pré-annotation par l'approche classique
  # (on corrige ensuite à la main : boîtes ET classes)
  python -m src.data.prelabel --data-root data --out data/annotations/ls_tasks_train.json \
      --input data/raw/group/multiple_coins data/raw/new/train/L2 \
              data/raw/new/train/L3 data/raw/new/train/L4 data/annotations/needs_review.csv

  # Images de TEST : AUCUNE pré-annotation (annotation manuelle pure, sinon la vérité terrain
  # serait biaisée en faveur de l'approche classique)
  python -m src.data.prelabel --data-root data --out data/annotations/ls_tasks_test.json \
      --input data/raw/new/test --no-predictions

--input accepte des dossiers (récursif) et/ou des CSV avec une colonne `file_name`
(chemin relatif à data/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from src.common.labels import UNKNOWN, image_key
from src.evaluation.classical_runner import IMG_EXT, run_on_images, standardize


def gather(inputs: list[str], data_root: Path) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += [q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT]
        elif p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                continue
            if "file_name" in df:
                paths += [data_root / f for f in df["file_name"]]
        else:
            sys.exit(f"--input : ni dossier ni CSV : {item}")
    return sorted(set(paths))


def build_tasks(paths, data_root: Path, det: pd.DataFrame | None) -> list[dict]:
    tasks = []
    for p in paths:
        rel = p.resolve().relative_to(data_root.resolve()).as_posix()
        task = {"data": {"image": f"/data/local-files/?d={rel}"}}
        if det is not None:
            w, h = Image.open(p).size
            rows = det[det["image_id"] == image_key(p)]
            result = []
            for i, r in enumerate(rows.itertuples()):
                label = "a_verifier" if r.class_name == UNKNOWN else r.class_name
                result.append({
                    "id": f"p{i}", "from_name": "label", "to_name": "image",
                    "type": "rectanglelabels", "original_width": w, "original_height": h,
                    "value": {
                        "x": max(r.x_min, 0) / w * 100, "y": max(r.y_min, 0) / h * 100,
                        "width": (min(r.x_max, w) - max(r.x_min, 0)) / w * 100,
                        "height": (min(r.y_max, h) - max(r.y_min, 0)) / h * 100,
                        "rotation": 0, "rectanglelabels": [label],
                    },
                })
            task["predictions"] = [{"model_version": "classique", "result": result}]
        tasks.append(task)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-predictions", action="store_true", help="tâches vides (jeu de test)")
    ap.add_argument("--from-csv", help="predictions.csv déjà produit par l'approche classique")
    a = ap.parse_args()
    data_root = Path(a.data_root)

    paths = gather(a.input, data_root)
    if not paths:
        sys.exit("Aucune image trouvée.")
    det = None
    if not a.no_predictions:
        if a.from_csv:
            det, _ = standardize(pd.read_csv(a.from_csv))
        else:
            det, ms = run_on_images(paths)
            print(f"Approche classique : {ms:.0f} ms/image")
    tasks = build_tasks(paths, data_root, det)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {len(tasks)} tâches -> {a.out}")


if __name__ == "__main__":
    main()
