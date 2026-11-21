"""Évaluation par comptage (sans boîtes englobantes), commune aux deux approches.

Vérité terrain : un fichier de contenu par image (format `fichier,dossier,contenu`,
contenu = « 10:1;50:2;1D:1 »), comme data/raw/new/*/contenu_*.csv.

Sans boîtes, on ne sait pas QUELLE pièce prédite correspond à QUELLE pièce réelle.
On compare donc des **multi-ensembles** d'étiquettes par image :
- pièces correctes d'une classe = min(nb prédit, nb réel) ;
- précision / rappel / F1 par classe à partir de ces comptes (les « inconnues » comptent
  comme des pièces réelles non retrouvées, jamais comme des prédictions fausses d'une classe) ;
- erreur de nombre total de pièces, erreur absolue sur le montant (millimes),
  taux d'images exactement correctes.

Limite : une pièce de 50 prédite 20 et une pièce de 20 prédite 50 dans la même image se
compensent. L'évaluation finale (jalon 5) utilisera en plus l'appariement par IoU.

Usage :
    python -m src.evaluation.count_metrics --pred results/classical/predictions/predictions.csv \
        --truth data/raw/new/train/contenu_train.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.common.coin_classes import CLASS_NAMES, COIN_VALUES_MILLIMES, UNKNOWN_LABEL

SHORT_TO_CLASS = {"10": "10_millimes", "20": "20_millimes", "50": "50_millimes",
                  "100": "100_millimes", "200": "200_millimes", "500": "500_millimes",
                  "1D": "1_dinars", "2D": "2_dinars", "5D": "5_dinars"}


def parse_contenu(text: str) -> dict[str, int]:
    """'10:1;50:2;1D:1' -> {'10_millimes': 1, '50_millimes': 2, '1_dinars': 1}."""
    counts = {c: 0 for c in CLASS_NAMES}
    for part in str(text).split(";"):
        if part.strip():
            key, n = part.split(":")
            counts[SHORT_TO_CLASS[key.strip()]] += int(n)
    return counts


def load_truth(path: Path) -> pd.DataFrame:
    """Une ligne par image (index = image_id), une colonne par classe."""
    df = pd.read_csv(path)
    rows = {Path(f).stem: parse_contenu(c) for f, c in zip(df.fichier, df.contenu)}
    out = pd.DataFrame.from_dict(rows, orient="index")[CLASS_NAMES]
    out["dossier"] = pd.Series(dict(zip(df.fichier.map(lambda f: Path(f).stem), df.dossier)))
    return out


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """`pred` : une ligne par pièce (colonnes image_id, class_name)."""
    pc = pd.crosstab(pred.image_id, pred.class_name)
    pc = pc.reindex(index=truth.index, columns=[*CLASS_NAMES, UNKNOWN_LABEL], fill_value=0)
    tc = truth[CLASS_NAMES]
    tp = pd.DataFrame({c: pc[c].clip(upper=tc[c]) for c in CLASS_NAMES})

    per_class = pd.DataFrame({
        "réel": tc.sum(), "prédit": pc[CLASS_NAMES].sum(), "corrects": tp.sum()})
    per_class["précision"] = per_class.corrects / per_class["prédit"].where(per_class["prédit"] > 0)
    per_class["rappel"] = per_class.corrects / per_class["réel"]
    per_class["F1"] = (2 * per_class.précision * per_class.rappel
                       / (per_class.précision + per_class.rappel))

    values = pd.Series(COIN_VALUES_MILLIMES)[CLASS_NAMES]
    images = pd.DataFrame({
        "dossier": truth["dossier"],
        "pièces_réelles": tc.sum(axis=1),
        "pièces_détectées": pc.sum(axis=1),
        "inconnues": pc[UNKNOWN_LABEL],
        "corrects": tp.sum(axis=1),
        "total_réel": tc @ values,
        "total_prédit": pc[CLASS_NAMES] @ values,
    })
    images["erreur_total"] = (images.total_prédit - images.total_réel).abs()
    images["exacte"] = (pc[CLASS_NAMES].values == tc.values).all(axis=1) & (images.inconnues == 0)
    return {"per_class": per_class, "images": images}


def summary(images: pd.DataFrame) -> pd.DataFrame:
    g = images.groupby("dossier")
    out = pd.DataFrame({
        "images": g.size(),
        "pièces": g.pièces_réelles.sum(),
        "détectées": g.pièces_détectées.sum(),
        "inconnues": g.inconnues.sum(),
        "bien classées": g.corrects.sum(),
        "erreur nb pièces (moy.)": g.apply(lambda d: (d.pièces_détectées - d.pièces_réelles).abs().mean()),
        "erreur montant moy. (mill.)": g.erreur_total.mean(),
        "images exactes": g.exacte.sum(),
    })
    out.loc["TOTAL"] = [len(images), images.pièces_réelles.sum(), images.pièces_détectées.sum(),
                        images.inconnues.sum(), images.corrects.sum(),
                        (images.pièces_détectées - images.pièces_réelles).abs().mean(),
                        images.erreur_total.mean(), images.exacte.sum()]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Évaluation par comptage")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    args = parser.parse_args()

    truth = load_truth(args.truth)
    pred = pd.read_csv(args.pred)
    missing = set(truth.index) - set(pred.image_id)
    if missing:
        print(f"Attention : {len(missing)} image(s) sans aucune prédiction (comptées à 0 pièce).")
    res = evaluate(pred[pred.image_id.isin(truth.index)], truth)
    pd.set_option("display.width", 200)
    print(summary(res["images"]).round(2).to_string())
    print()
    print(res["per_class"].round(2).to_string())
    pc = res["per_class"]
    print(f"\nF1 macro (9 classes) : {pc.F1.fillna(0).mean():.3f}")


if __name__ == "__main__":
    main()
