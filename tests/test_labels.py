"""Montant total et normalisation des classes : le socle du calcul en TND."""
from src.common.labels import (CLASS_NAMES, count_by_class, normalize_class,
                               parse_class_strict, total_millimes)


def test_normalize_class_variantes():
    assert normalize_class("500") == "500_millimes"
    assert normalize_class("500 millimes") == "500_millimes"
    assert normalize_class(1000) == "1_dinar"
    assert normalize_class("2 dinars") == "2_dinars"
    assert normalize_class("5") == "5_dinars"
    assert normalize_class("inconnue") == "unknown"
    assert normalize_class(None) == "unknown"


def test_parse_strict_ne_confond_pas_les_dossiers_de_niveau():
    assert parse_class_strict("L2") is None
    assert parse_class_strict("total_millimes") is None
    assert parse_class_strict("2_dinar") == "2_dinars"
    assert parse_class_strict("10_millimes") == "10_millimes"


def test_montant_et_comptage_ignorent_les_inconnues():
    pieces = ["500_millimes", "1_dinar", "1_dinar", "unknown"]
    assert total_millimes(pieces) == 2500
    counts = count_by_class(pieces)
    assert counts["1_dinar"] == 2 and sum(counts.values()) == 3
    assert list(counts) == CLASS_NAMES
