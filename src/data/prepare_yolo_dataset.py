"""Construit le jeu de données YOLO (train/val) à partir des CSV d'annotations.

    python -m src.data.prepare_yolo_dataset \
        --annotations data/annotations/annotations_auto_single.csv \
                      data/annotations/annotations_train_multi.csv \
        --data-root data --manifest evaluation/test_manifest.csv --out data/yolo

- Refuse (erreur) toute image appartenant au jeu de test commun.
- Split de validation stratifié par dossier d'origine, graine fixe (reproductible).
- Les images sont redimensionnées (--max-side) : les labels YOLO sont normalisés, ils restent
  valables. Cela réduit beaucoup la taille à téléverser sur Kaggle / Colab.
- Écrit data/yolo/split.csv (image -> train/val) et configs/yolo_data.yaml.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

from src.common.labels import CLASS_NAMES, image_key

BOX = ["x_min", "y_min", "x_max", "y_max"]


def split_images(info: pd.DataFrame, val_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    split: dict[str, str] = {}
    for _, g in info.groupby("stratum"):
        ids = sorted(g["image_id"])
        rng.shuffle(ids)
        n_val = int(round(val_frac * len(ids)))
        if n_val == 0 and len(ids) >= 3:
            n_val = 1
        for i, iid in enumerate(ids):
            split[iid] = "val" if i < n_val else "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest", default="evaluation/test_manifest.csv")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--yaml", default="configs/yolo_data.yaml")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()
    data_root, out = Path(a.data_root), Path(a.out)

    ann = pd.concat([pd.read_csv(p) for p in a.annotations], ignore_index=True)
    ann = ann[ann["class_name"].isin(CLASS_NAMES)]
    test_keys = {image_key(i) for i in pd.read_csv(a.manifest)["image_id"]}
    leak = sorted(set(ann["image_id"]) & test_keys)
    if leak:
        sys.exit(f"ERREUR : {len(leak)} images du jeu de test sont dans les annotations "
                 f"d'entraînement (ex. {leak[:3]}). Les retirer.")
    dup = ann.groupby("image_id")["file_name"].nunique()
    if (dup > 1).any():
        sys.exit(f"ERREUR : image annotée deux fois avec des fichiers différents : {list(dup[dup > 1].index[:3])}")

    info = ann.drop_duplicates("image_id")[["image_id", "file_name"]].copy()
    info["stratum"] = info["file_name"].map(lambda f: Path(f).parent.name)
    split = split_images(info, a.val_frac, a.seed)

    for sub in ("train", "val"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    for _, row in info.iterrows():
        sp = split[row["image_id"]]
        img = cv2.imread(str(data_root / row["file_name"]))
        if img is None:
            sys.exit(f"Image illisible : {data_root / row['file_name']}")
        h, w = img.shape[:2]
        s = min(1.0, a.max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "images" / sp / f"{row['image_id']}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for r in ann[ann["image_id"] == row["image_id"]].itertuples():
            xc, yc = (r.x_min + r.x_max) / 2 / w, (r.y_min + r.y_max) / 2 / h
            bw, bh = (r.x_max - r.x_min) / w, (r.y_max - r.y_min) / h
            lines.append(f"{CLASS_NAMES.index(r.class_name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / sp / f"{row['image_id']}.txt").write_text("\n".join(lines) + "\n")

    pd.DataFrame({"image_id": list(split), "split": list(split.values())}).to_csv(
        out / "split.csv", index=False)
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "names": dict(enumerate(CLASS_NAMES))}
    Path(a.yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(a.yaml).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    ann["split"] = ann["image_id"].map(split)
    tab = ann.pivot_table(index="class_name", columns="split", values="image_id",
                          aggfunc="count", fill_value=0).reindex(CLASS_NAMES, fill_value=0)
    print("Instances par classe et par split :\n", tab.to_string())
    print(f"\n[OK] {len(info)} images -> {out}  |  {a.yaml}")
    if "val" not in tab or (tab["val"] == 0).any():
        print("[ATTENTION] une classe n'a aucune instance en validation : augmenter --val-frac.")


if __name__ == "__main__":
    main()
