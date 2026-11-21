"""Comptage par classe et calcul du montant total (commun aux deux approches).

Les montants sont calculés en **millimes** (entiers) pour éviter les erreurs d'arrondi,
puis convertis en dinars uniquement pour l'affichage.
Les pièces « inconnues » sont comptées à part et n'entrent pas dans le total.
"""

from __future__ import annotations

from collections import Counter

from src.common.coin_classes import CLASS_NAMES, COIN_VALUES_MILLIMES, UNKNOWN_LABEL


def count_by_class(labels: list[str]) -> dict[str, int]:
    """Nombre de pièces par classe (toutes les classes + 'unknown', zéros inclus)."""
    counts = Counter(labels)
    return {c: counts.get(c, 0) for c in [*CLASS_NAMES, UNKNOWN_LABEL]}


def total_millimes(labels: list[str]) -> int:
    """Montant total en millimes ; les pièces inconnues ne sont pas comptées."""
    return sum(COIN_VALUES_MILLIMES[label] for label in labels if label != UNKNOWN_LABEL)


def format_tnd(millimes: int) -> str:
    """Ex. 8860 -> '8,860 DT' (notation tunisienne : 3 décimales)."""
    return f"{millimes // 1000},{millimes % 1000:03d} DT"
