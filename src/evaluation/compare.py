"""Comparaison finale : approche classique vs approche IA sur le jeu de test commun.

    python -m src.evaluation.compare \
        --manifest evaluation/test_manifest.csv \
        --ground-truth evaluation/ground_truth.csv \
        --classical-dir results/classical/test \
        --ai-dir results/ai/test \
        --out results/comparison

Chaque dossier de prédictions contient predictions_std.csv (et idéalement timings.csv),
produits par classical_runner.py et predict.py. Sorties dans --out :
  summary.csv / summary.md       tableau comparatif (à coller dans le rapport)
  by_level.csv                   métriques par niveau de difficulté
  per_class_<approche>.csv       métriques par classe
  per_image_<approche>.csv       erreurs image par image (pour l'analyse d'erreurs)
  confusion_<approche>.png       matrices de confusion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src.common.labels import CLASS_NAMES, UNKNOWN, image_key, normalize_class  # noqa: E402
from src.evaluation.matching import BOX, evaluate_approach, per_class_table, summarize  # noqa: E402

NEEDED = ["image_id", *BOX, "class_name"]


def load_boxes(path: Path, what: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        sys.exit(f"{what} ({path}) : colonnes manquantes {miss}")
    df = df.copy()
    df["image_id"] = df["image_id"].map(image_key)
    df["class_name"] = df["class_name"].map(normalize_class)
    return df


def plot_confusion(pairs: pd.DataFrame, title: str, out: Path) -> None:
    labels = CLASS_NAMES + [UNKNOWN]
    cm = confusion_matrix(pairs["gt_class"], pairs["pred_class"], labels=labels) if len(pairs) \
        else np.zeros((len(labels), len(labels)), int)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", default="evaluation/test_manifest.csv")
    ap.add_argument("--ground-truth", default="evaluation/ground_truth.csv")
    ap.add_argument("--classical-dir")
    ap.add_argument("--ai-dir")
    ap.add_argument("--out", default="results/comparison")
    ap.add_argument("--iou", type=float, default=0.5, help="seuil d'appariement (défaut 0,5)")
    a = ap.parse_args()

    approaches = {}
    if a.classical_dir:
        approaches["Classique"] = Path(a.classical_dir)
    if a.ai_dir:
        approaches["IA (YOLOv8)"] = Path(a.ai_dir)
    if not approaches:
        sys.exit("Donner --classical-dir et/ou --ai-dir.")

    man = pd.read_csv(a.manifest)
    man["key"] = man["image_id"].map(image_key)
    ids = list(man["key"])
    gt = load_boxes(Path(a.ground_truth), "Vérité terrain")
    absent = sorted(set(ids) - set(gt["image_id"]))
    if absent:
        sys.exit(f"Vérité terrain incomplète : {len(absent)} images du manifeste sans annotation "
                 f"(ex. {absent[:3]}). Terminer l'annotation avant de comparer.")
    gt = gt[gt["image_id"].isin(ids)]
    print(f"Jeu de test : {len(ids)} images, {len(gt)} pièces annotées.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    level = dict(zip(man["key"], man["difficulty_level"]))
    groups = {"niveaux 1-2": lambda l: l <= 2, "niveaux 3-4": lambda l: l >= 3}
    summaries, by_level_rows = {}, []

    for name, d in approaches.items():
        tag = "classique" if name == "Classique" else "ia"
        pred = load_boxes(d / "predictions_std.csv", f"Prédictions {name}")
        extra = sorted(set(pred["image_id"]) - set(ids))
        if extra:
            print(f"[ATTENTION] {name} : {len(extra)} images hors du jeu de test ignorées.")
        pred = pred[pred["image_id"].isin(ids)]
        per_img, pairs = evaluate_approach(ids, gt, pred, a.iou)
        s = summarize(per_img, pairs)
        tfile = d / "timings.csv"
        s["time_ms"] = float(pd.read_csv(tfile).query("image_id in @ids")["time_ms"].mean()) \
            if tfile.exists() else float("nan")
        summaries[name] = s

        per_img.insert(1, "level", per_img["image_id"].map(level))
        per_img.to_csv(out / f"per_image_{tag}.csv", index=False)
        per_class_table(per_img, pairs).round(3).to_csv(out / f"per_class_{tag}.csv", index=False)
        plot_confusion(pairs, f"Matrice de confusion — {name}", out / f"confusion_{tag}.png")

        for lv in sorted(set(level.values())):
            sel = per_img["level"] == lv
            m = summarize(per_img[sel], pairs[pairs["image_id"].isin(per_img[sel]["image_id"])])
            by_level_rows.append({"approche": name, "niveau": f"niveau {lv}", **m})
        for gname, cond in groups.items():
            sel = per_img["level"].map(cond)
            m = summarize(per_img[sel], pairs[pairs["image_id"].isin(per_img[sel]["image_id"])])
            by_level_rows.append({"approche": name, "niveau": gname, **m})

    cols = [("Précision de détection", "det_precision", 3), ("Rappel de détection", "det_recall", 3),
            ("F1 détection", "det_f1", 3), ("Exactitude classification (paires)", "cls_accuracy", 3),
            ("F1 classification macro", "cls_macro_f1", 3),
            ("Erreur moyenne de comptage (pièces/image)", "mean_count_error", 2),
            ("Erreur absolue moyenne sur le total (TND)", "mean_amount_error_tnd", 3),
            ("Temps moyen par image (ms)", "time_ms", 0)]
    summary = pd.DataFrame({"Critère": [c[0] for c in cols]})
    for name, s in summaries.items():
        summary[name] = [round(s[k], nd) if nd else round(s[k]) if s[k] == s[k] else "n/a"
                         for _, k, nd in cols]
    bl = pd.DataFrame(by_level_rows)
    extra_rows = []
    for g in groups:
        for label, key, nd in (("F1 détection", "det_f1", 3), ("Erreur sur le total (TND)", "mean_amount_error_tnd", 3)):
            extra_rows.append([f"{label} — {g}"] + [
                round(bl[(bl["approche"] == n) & (bl["niveau"] == g)].iloc[0][key], nd)
                for n in approaches])
    summary = pd.concat([summary, pd.DataFrame(extra_rows, columns=summary.columns)], ignore_index=True)

    summary.to_csv(out / "summary.csv", index=False)
    bl.round(3).to_csv(out / "by_level.csv", index=False)
    notes = (f"\nSeuil IoU d'appariement : {a.iou}. {len(ids)} images, {len(gt)} pièces.\n"
             "Classification évaluée sur les paires appariées ; comptage et montant sur toutes "
             "les prédictions ('unknown' exclues du montant). Temps classique = durée totale du "
             "pipeline / nombre d'images (inclut le démarrage Python) ; temps IA = inférence par image "
             "après échauffement.\n")
    (out / "summary.md").write_text(md_table(summary) + "\n" + notes, encoding="utf-8")
    print(md_table(summary) + "\n" + notes)
    print(f"[OK] résultats dans {out}")


if __name__ == "__main__":
    main()
