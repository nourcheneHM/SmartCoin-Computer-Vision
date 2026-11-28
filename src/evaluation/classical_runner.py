"""Adaptateur autour de l'approche classique de Nourchene (sans la modifier).

Il lance `python -m src.classical.pipeline --input <dossier>` puis convertit le
predictions.csv produit au FORMAT STANDARD utilisé par tout le reste du projet :

    image_id, x_min, y_min, x_max, y_max, class_name, score

Usages (depuis la racine du dépôt) :

    # 1. Vérifier que la conversion comprend le CSV existant de Nourchene
    python -m src.evaluation.classical_runner --from-csv results/classical/dev/predictions.csv

    # 2. Lancer l'approche classique sur le jeu de test commun (même liste que l'IA)
    python -m src.evaluation.classical_runner --manifest evaluation/test_manifest.csv \
        --data-root data --out results/classical/test

    # 3. Lancer sur un dossier quelconque
    python -m src.evaluation.classical_runner --input data/raw/new/train/L2 --out results/classical/L2

SI LA CONVERSION ÉCHOUE : adapter le dictionnaire ALIASES ci-dessous (noms de colonnes du
predictions.csv de Nourchene). C'est le seul point de couplage avec son code.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.common.labels import image_key, normalize_class

ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STD_COLS = ["image_id", "x_min", "y_min", "x_max", "y_max", "class_name", "score"]

# Noms de colonnes acceptés (en minuscules). Adapter ici si le CSV de Nourchene diffère.
ALIASES = {
    "image_id": ["image_id", "image", "image_name", "file_name", "filename", "file", "path",
                 "image_path", "img"],
    "x_min": ["x_min", "xmin", "x1"], "y_min": ["y_min", "ymin", "y1"],
    "x_max": ["x_max", "xmax", "x2"], "y_max": ["y_max", "ymax", "y2"],
    "cx": ["cx", "center_x", "x_center", "xc", "x"],
    "cy": ["cy", "center_y", "y_center", "yc", "y"],
    "r": ["r", "radius", "rayon", "radius_px", "r_px"],
    "class_name": ["class_name", "pred_class", "predicted_class", "class", "label",
                   "prediction", "predicted", "value", "valeur", "value_mm", "valeur_mm",
                   "millimes"],
    "score": ["score", "confidence", "conf", "confiance", "reliability", "fiabilite"],
}


def _pick(cols: dict, key: str):
    for name in ALIASES[key]:
        if name in cols:
            return cols[name]
    return None


def standardize(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convertit le CSV de Nourchene au format standard. Retourne (df, correspondance)."""
    cols = {str(c).strip().lower(): c for c in raw.columns}
    m = {k: _pick(cols, k) for k in ALIASES}
    if m["image_id"] is None or m["class_name"] is None:
        raise ValueError(
            f"Colonnes image/classe introuvables dans {list(raw.columns)}. "
            "Adapter ALIASES dans src/evaluation/classical_runner.py."
        )
    if all(m[k] for k in ("x_min", "y_min", "x_max", "y_max")):
        x0, y0, x1, y1 = (raw[m[k]].astype(float) for k in ("x_min", "y_min", "x_max", "y_max"))
    elif all(m[k] for k in ("cx", "cy", "r")):
        cx, cy, r = (raw[m[k]].astype(float) for k in ("cx", "cy", "r"))
        x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    else:
        raise ValueError(
            f"Ni boîtes (x_min..y_max) ni cercles (cx, cy, r) dans {list(raw.columns)}. "
            "Adapter ALIASES dans src/evaluation/classical_runner.py."
        )
    out = pd.DataFrame({
        "image_id": raw[m["image_id"]].map(image_key),
        "x_min": x0.round(1), "y_min": y0.round(1), "x_max": x1.round(1), "y_max": y1.round(1),
        "class_name": raw[m["class_name"]].map(normalize_class),
        "score": raw[m["score"]].astype(float) if m["score"] else 1.0,
    })[STD_COLS]
    return out, m


def _run_pipeline(input_dir: Path) -> tuple[Path, float]:
    """Lance le pipeline de Nourchene ; retourne (predictions.csv produit, durée en s)."""
    start = time.time()
    cmd = [sys.executable, "-m", "src.classical.pipeline", "--input", str(input_dir)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    elapsed = time.time() - start
    if res.returncode != 0:
        raise RuntimeError(f"Le pipeline classique a échoué :\n{res.stderr[-1500:]}")
    found = [p for p in (ROOT / "results" / "classical").rglob("predictions.csv")
             if p.stat().st_mtime >= start - 1]
    if not found:
        raise FileNotFoundError(
            "Aucun predictions.csv récent sous results/classical/. Lancer le pipeline à la main, "
            "puis utiliser --from-csv <chemin du predictions.csv>."
        )
    return max(found, key=lambda p: p.stat().st_mtime), elapsed


def run_on_images(paths: list[Path], keep_tmp: bool = False) -> tuple[pd.DataFrame, float]:
    """Exécute l'approche classique sur une liste d'images.

    Retourne (prédictions au format standard, temps moyen par image en ms).
    Les images sont copiées dans un dossier temporaire pour que le pipeline traite
    EXACTEMENT cette liste (et rien d'autre).
    """
    tmp = ROOT / "results" / "_tmp_classical_input"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for p in paths:
            shutil.copy2(p, tmp / Path(p).name)
        csv_path, elapsed = _run_pipeline(tmp)
        std, _ = standardize(pd.read_csv(csv_path))
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    wanted = {image_key(p) for p in paths}
    std = std[std["image_id"].isin(wanted)].reset_index(drop=True)
    return std, elapsed * 1000.0 / max(len(paths), 1)


def _save(std: pd.DataFrame, ids: list[str], ms: float | None, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    std.to_csv(out / "predictions_std.csv", index=False)
    if ms is not None:
        pd.DataFrame({"image_id": ids, "time_ms": ms}).to_csv(out / "timings.csv", index=False)
    print(f"[OK] {len(std)} détections, {len(ids)} images -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", help="evaluation/test_manifest.csv (liste figée)")
    ap.add_argument("--data-root", default="data", help="dossier data/ (pour --manifest)")
    ap.add_argument("--input", help="dossier d'images (récursif)")
    ap.add_argument("--from-csv", help="convertir un predictions.csv existant (aucune exécution)")
    ap.add_argument("--out", help="dossier de sortie (predictions_std.csv, timings.csv)")
    a = ap.parse_args()

    if a.from_csv:
        std, mapping = standardize(pd.read_csv(a.from_csv))
        print("Correspondance de colonnes détectée :", {k: v for k, v in mapping.items() if v})
        print(std.head(8).to_string(index=False))
        print("Détections par classe :\n", std["class_name"].value_counts().to_string())
        ids = sorted(std["image_id"].unique())
        if a.manifest:  # ne garder que les images du jeu de test commun
            ids = [image_key(i) for i in pd.read_csv(a.manifest)["image_id"]]
            std = std[std["image_id"].isin(ids)].reset_index(drop=True)
        if a.out:
            _save(std, ids, None, Path(a.out))
        return

    if a.manifest:
        man = pd.read_csv(a.manifest)
        paths = [Path(a.data_root) / f for f in man["file_name"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.exit(f"Images introuvables (ex. {missing[0]}). Vérifier --data-root.")
    elif a.input:
        paths = sorted(p for p in Path(a.input).rglob("*") if p.suffix.lower() in IMG_EXT)
    else:
        sys.exit("Donner --manifest, --input ou --from-csv.")
    std, ms = run_on_images(paths)
    if a.out:
        _save(std, [image_key(p) for p in paths], ms, Path(a.out))
    else:
        print(std.head(), f"\n{ms:.0f} ms/image")


if __name__ == "__main__":
    main()
