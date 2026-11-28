"""Appariement prédiction <-> vérité terrain et calcul des métriques.

Définitions (identiques pour les deux approches) :
- Détection : appariement UN-À-UN par IoU >= seuil (0,5), sans tenir compte de la classe.
  Prédiction non appariée = faux positif ; pièce non détectée = faux négatif.
- Classification : évaluée sur les paires appariées (classe prédite vs classe réelle).
  Une prédiction 'unknown' est comptée comme une erreur de classification.
- Comptage : par image, somme sur les 9 classes de |n_prédit - n_réel|
  (toutes les prédictions comptent, appariées ou non ; les 'unknown' sont exclues).
- Montant : |total prédit - total réel| en TND par image (les 'unknown' valent 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.common.labels import CLASS_NAMES, count_by_class, total_millimes

BOX = ["x_min", "y_min", "x_max", "y_max"]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Appariement glouton un-à-un par IoU décroissant. Retourne [(i_gt, j_pred, iou)]."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    cand = sorted(
        ((iou(g, p), i, j) for i, g in enumerate(gt_boxes) for j, p in enumerate(pred_boxes)),
        reverse=True,
    )
    used_g, used_p, out = set(), set(), []
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, v))
    return out


def evaluate_approach(image_ids, gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float = 0.5):
    """gt / pred : image_id (clé), x_min..y_max, class_name. Retourne (par image, paires)."""
    gt_by = {k: g for k, g in gt.groupby("image_id")}
    pr_by = {k: g for k, g in pred.groupby("image_id")}
    empty_g, empty_p = gt.iloc[0:0], pred.iloc[0:0]
    rows, pairs = [], []
    for iid in image_ids:
        g, p = gt_by.get(iid, empty_g), pr_by.get(iid, empty_p)
        m = match_boxes(g[BOX].to_numpy(float), p[BOX].to_numpy(float), iou_thr)
        tp = len(m)
        for gi, pi, v in m:
            pairs.append({"image_id": iid, "gt_class": g["class_name"].iloc[gi],
                          "pred_class": p["class_name"].iloc[pi], "iou": v})
        gc, pc = count_by_class(g["class_name"]), count_by_class(p["class_name"])
        gt_mill, pr_mill = total_millimes(g["class_name"]), total_millimes(p["class_name"])
        row = {"image_id": iid, "n_gt": len(g), "n_pred": len(p), "tp": tp,
               "fp": len(p) - tp, "fn": len(g) - tp,
               "count_abs_error": sum(abs(gc[c] - pc[c]) for c in CLASS_NAMES),
               "gt_total_tnd": gt_mill / 1000, "pred_total_tnd": pr_mill / 1000,
               "amount_abs_error_tnd": abs(gt_mill - pr_mill) / 1000}
        row.update({f"gt_{c}": gc[c] for c in CLASS_NAMES})
        row.update({f"pred_{c}": pc[c] for c in CLASS_NAMES})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pairs, columns=["image_id", "gt_class", "pred_class", "iou"])


def summarize(per_image: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    tp, fp, fn = (int(per_image[c].sum()) for c in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if len(pairs):
        labels = [c for c in CLASS_NAMES if c in set(pairs["gt_class"]) | set(pairs["pred_class"])]
        macro = f1_score(pairs["gt_class"], pairs["pred_class"], labels=labels,
                         average="macro", zero_division=0)
        acc = float((pairs["gt_class"] == pairs["pred_class"]).mean())
    else:
        macro, acc = 0.0, 0.0
    return {"images": len(per_image), "tp": tp, "fp": fp, "fn": fn,
            "det_precision": prec, "det_recall": rec, "det_f1": f1,
            "cls_accuracy": acc, "cls_macro_f1": macro,
            "mean_count_error": float(per_image["count_abs_error"].mean()),
            "mean_amount_error_tnd": float(per_image["amount_abs_error_tnd"].mean())}


def per_class_table(per_image: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Précision / rappel / F1 de classification par classe + erreur de comptage par classe."""
    if len(pairs):
        p, r, f, s = precision_recall_fscore_support(
            pairs["gt_class"], pairs["pred_class"], labels=CLASS_NAMES, zero_division=0)
    else:
        p = r = f = np.zeros(len(CLASS_NAMES))
        s = np.zeros(len(CLASS_NAMES), dtype=int)
    tab = pd.DataFrame({"class": CLASS_NAMES, "precision": p, "recall": r, "f1": f, "support": s})
    tab["gt_total"] = [int(per_image[f"gt_{c}"].sum()) for c in CLASS_NAMES]
    tab["pred_total"] = [int(per_image[f"pred_{c}"].sum()) for c in CLASS_NAMES]
    tab["count_mae"] = [float((per_image[f"pred_{c}"] - per_image[f"gt_{c}"]).abs().mean())
                        for c in CLASS_NAMES]
    return tab
