"""IoU, appariement un-à-un et métriques : cœur de la comparaison équitable."""
import pandas as pd
import pytest

from src.evaluation.matching import evaluate_approach, iou, match_boxes, summarize


def box(img, x, cls, y=0, size=10):
    return {"image_id": img, "x_min": x, "y_min": y, "x_max": x + size, "y_max": y + size,
            "class_name": cls}


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_appariement_un_a_un():
    # deux prédictions sur la même pièce : une seule est appariée, l'autre est un faux positif
    m = match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10], [1, 0, 11, 10]], thr=0.5)
    assert len(m) == 1 and m[0][1] == 0


def test_metriques_et_montant():
    gt = pd.DataFrame([box("a", 0, "500_millimes"), box("a", 50, "1_dinar"), box("b", 0, "2_dinars")])
    pred = pd.DataFrame([
        box("a", 0, "500_millimes"),   # correcte
        box("a", 50, "2_dinars"),      # bien détectée, mauvaise classe (+1 TND d'erreur)
        box("b", 200, "5_dinars"),     # faux positif ; la pièce de b n'est pas détectée
    ])
    per_img, pairs = evaluate_approach(["a", "b"], gt, pred)
    s = summarize(per_img, pairs)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["det_precision"] == pytest.approx(2 / 3) and s["det_recall"] == pytest.approx(2 / 3)
    assert s["cls_accuracy"] == 0.5
    # image a : 1,5 vs 2,5 TND -> 1,0 ; image b : 2,0 vs 5,0 -> 3,0 ; moyenne 2,0
    assert s["mean_amount_error_tnd"] == pytest.approx(2.0)
