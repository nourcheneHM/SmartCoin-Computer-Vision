"""Convertit un export Label Studio (format JSON) en CSV d'annotations + contrôle des comptages.

  # Entraînement (photos multi-pièces corrigées)
  python -m src.data.labelstudio_to_csv --export data/annotations/ls_export_train.json \
      --data-root data --out data/annotations/annotations_train_multi.csv \
      --counts data/raw/new/train/contenu_train.csv

  # Test -> evaluation/ground_truth.csv (vérité terrain commune aux deux approches)
  python -m src.data.labelstudio_to_csv --export data/annotations/ls_export_test.json \
      --data-root data --out data/annotations/annotations_test.csv \
      --ground-truth evaluation/ground_truth.csv \
      --manifest evaluation/test_manifest.csv --counts evaluation/test_counts.csv

Le contrôle `--counts` compare, image par image, le nombre de pièces annotées par classe au
comptage noté à la prise de vue. Les écarts sont AFFICHÉS : les corriger dans Label Studio
puis ré-exporter.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from src.common.labels import CLASS_NAMES, image_key, parse_class_strict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLS = ["image_id", "file_name", "coin_id", "x_min", "y_min", "x_max", "y_max",
        "class_name", "occluded"]


def build_index(data_root: Path) -> dict[str, str]:
    """clé d'image -> chemin relatif à data/ (pour retrouver le fichier d'une tâche)."""
    idx: dict[str, str] = {}
    for p in (data_root / "raw").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(image_key(p), p.relative_to(data_root).as_posix())
    return idx


def task_key(task: dict) -> str:
    data = task.get("data", {})
    img = data.get("image") or next(iter(data.values()))
    if "?d=" in img:
        img = unquote(img.split("?d=", 1)[1])
    key = image_key(img)
    if "/data/upload/" in img:  # Label Studio préfixe les uploads d'un hash
        key = re.sub(r"^[0-9a-f]{8}-", "", key)
    return key


def parse_export(path: Path, index: dict[str, str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    import json
    tasks = json.loads(path.read_text(encoding="utf-8"))
    rows, unverified, not_annotated = [], [], []
    for t in tasks:
        key = task_key(t)
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            not_annotated.append(key)
            continue
        regions: dict[str, dict] = {}
        for r in anns[-1]["result"]:
            reg = regions.setdefault(r["id"], {})
            if r["type"] == "rectanglelabels":
                v = r["value"]
                w, h = r["original_width"], r["original_height"]
                reg["box"] = (
                    max(v["x"] / 100 * w, 0), max(v["y"] / 100 * h, 0),
                    min((v["x"] + v["width"]) / 100 * w, w), min((v["y"] + v["height"]) / 100 * h, h),
                )
                reg["label"] = v["rectanglelabels"][0]
            elif r["type"] == "choices" and r.get("from_name") == "occluded":
                reg["occluded"] = int("occluded" in r["value"]["choices"])
        boxes = [r for r in regions.values() if "box" in r]
        boxes.sort(key=lambda r: (round(r["box"][1] / 50), r["box"][0]))  # lecture haut-gauche
        for i, r in enumerate(boxes, 1):
            cls = parse_class_strict(r["label"])
            if cls is None:
                unverified.append(f"{key} (étiquette '{r['label']}')")
                continue
            x0, y0, x1, y1 = (int(round(c)) for c in r["box"])
            rows.append({"image_id": key, "file_name": index.get(key, ""), "coin_id": i,
                         "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                         "class_name": cls, "occluded": r.get("occluded", 0)})
    return pd.DataFrame(rows, columns=COLS), sorted(set(unverified)), not_annotated


def read_counts(path: Path) -> dict[str, dict[str, int]] | None:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    id_col = next((low[k] for k in ("image_id", "image", "file_name", "filename", "file") if k in low),
                  df.columns[0])
    class_cols = {c: parse_class_strict(c) for c in df.columns if c != id_col}
    class_cols = {c: k for c, k in class_cols.items() if k}
    if not class_cols:
        print(f"[ATTENTION] {path} : aucune colonne de classe reconnue, contrôle des comptages ignoré.")
        return None
    return {image_key(r[id_col]): {k: int(r[c]) for c, k in class_cols.items()}
            for _, r in df.iterrows()}


def check_counts(df: pd.DataFrame, counts: dict[str, dict[str, int]]) -> int:
    bad = 0
    got = df.groupby(["image_id", "class_name"]).size()
    for key, expected in counts.items():
        for cls in CLASS_NAMES:
            n_exp, n_got = expected.get(cls, 0), int(got.get((key, cls), 0))
            if n_exp != n_got:
                print(f"  écart {key} : {cls} attendu {n_exp}, annoté {n_got}")
                bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--export", required=True, help="export Label Studio au format JSON")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True, help="CSV d'annotations (avec file_name)")
    ap.add_argument("--counts", help="CSV des comptages de référence (contrôle)")
    ap.add_argument("--manifest", help="evaluation/test_manifest.csv")
    ap.add_argument("--ground-truth", help="écrit aussi evaluation/ground_truth.csv")
    a = ap.parse_args()

    df, unverified, missing = parse_export(Path(a.export), build_index(Path(a.data_root)))
    if unverified:
        sys.exit("Étiquettes non valides (remplacer 'a_verifier' dans Label Studio) :\n  "
                 + "\n  ".join(unverified))
    if missing:
        print(f"[ATTENTION] {len(missing)} tâches sans annotation : {missing[:5]}...")
    if (df["file_name"] == "").any():
        print("[ATTENTION] fichiers non retrouvés dans data/raw pour :",
              sorted(df[df["file_name"] == ""]["image_id"].unique())[:5])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[OK] {len(df)} pièces sur {df['image_id'].nunique()} images -> {a.out}")
    print(df["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())

    if a.counts:
        counts = read_counts(Path(a.counts))
        if counts is not None:
            bad = check_counts(df, counts)
            print("Contrôle des comptages : " + ("tout concorde." if bad == 0 else f"{bad} écart(s) à corriger."))

    if a.ground_truth:
        gt = df.copy()
        if a.manifest:
            man = pd.read_csv(a.manifest)
            by_key = {image_key(i): i for i in man["image_id"]}
            absent = sorted(set(by_key) - set(gt["image_id"]))
            if absent:
                print(f"[ATTENTION] {len(absent)} images du manifeste sans annotation : {absent[:5]}...")
            gt = gt[gt["image_id"].isin(by_key)].copy()
            gt["image_id"] = gt["image_id"].map(by_key)
        gt = gt.drop(columns=["file_name"])
        Path(a.ground_truth).parent.mkdir(parents=True, exist_ok=True)
        gt.to_csv(a.ground_truth, index=False)
        print(f"[OK] vérité terrain -> {a.ground_truth} ({len(gt)} pièces)")


if __name__ == "__main__":
    main()
