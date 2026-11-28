"""Classes SmartCoin : noms canoniques, normalisation des étiquettes, comptage et montant.

Module PARTAGÉ par l'approche IA, la comparaison et l'interface Streamlit.
Il est autonome : il ne modifie pas et ne dépend pas de src/common/coin_classes.py
(fichier de Nourchene).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Ordre fixe = indices de classes YOLO (0 à 8)
CLASS_NAMES = [
    "10_millimes", "20_millimes", "50_millimes", "100_millimes",
    "200_millimes", "500_millimes", "1_dinar", "2_dinars", "5_dinars",
]
VALUE_MILLIMES = dict(zip(CLASS_NAMES, [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]))
_BY_VALUE = {v: k for k, v in VALUE_MILLIMES.items()}
UNKNOWN = "unknown"

SHORT_LABEL = {
    "10_millimes": "10 mill.", "20_millimes": "20 mill.", "50_millimes": "50 mill.",
    "100_millimes": "100 mill.", "200_millimes": "200 mill.", "500_millimes": "500 mill.",
    "1_dinar": "1 DT", "2_dinars": "2 DT", "5_dinars": "5 DT", UNKNOWN: "?",
}

_IMG_EXT = re.compile(r"\.(jpe?g|png|bmp|webp|tiff?)$", re.I)
_STRICT = re.compile(
    r"^\s*(\d+)\s*[_\s-]*\s*(millimes?|mill?\.?|dinars?|dt|tnd)?\s*$", re.I
)


def image_key(value) -> str:
    """Clé d'image robuste : nom de fichier sans dossier ni extension."""
    return _IMG_EXT.sub("", Path(str(value).replace("\\", "/")).name)


def _class_from_number_unit(num: int, unit: str | None) -> str | None:
    unit = (unit or "").lower()
    dinar = unit.startswith(("dinar", "dt", "tnd"))
    value = num * 1000 if dinar else num
    if not unit and num in (1, 2, 5):  # '5' seul = 5 dinars
        value = num * 1000
    return _BY_VALUE.get(value)


def parse_class_strict(text) -> str | None:
    """Reconnaissance stricte (noms de dossiers, en-têtes de colonnes).

    '10_millimes' -> '10_millimes' ; '1_dinar' -> '1_dinar' ; '2_dinar' -> '2_dinars' ;
    'L2' -> None ; 'total_millimes' -> None.
    """
    m = _STRICT.match(str(text))
    if not m:
        return None
    return _class_from_number_unit(int(m.group(1)), m.group(2))


def normalize_class(label) -> str:
    """Convertit une étiquette quelconque en nom canonique (ou 'unknown').

    Accepte : '500', 500, '500 millimes', '1 dinar', 1000, '5_dinars', 'inconnue', None...
    À utiliser pour des VALEURS de prédiction, pas pour des noms de dossiers.
    """
    if label is None:
        return UNKNOWN
    s = str(label).strip().lower()
    if s in VALUE_MILLIMES:
        return s
    s = s.replace("é", "e").replace("è", "e")
    m = re.search(r"(\d+(?:[.,]\d+)?)", s)
    if not m:
        return UNKNOWN
    num = float(m.group(1).replace(",", "."))
    dinar = "dinar" in s or bool(re.search(r"\d\s*(dt|tnd|d)\b", s))
    value = round(num * 1000) if dinar else round(num)
    if not dinar and value in (1, 2, 5):
        value *= 1000
    return _BY_VALUE.get(value, UNKNOWN)


def count_by_class(names) -> dict[str, int]:
    """Nombre de pièces par classe (les 'unknown' sont exclues)."""
    c = Counter(normalize_class(n) for n in names)
    return {k: c.get(k, 0) for k in CLASS_NAMES}


def total_millimes(names) -> int:
    """Montant total en millimes (les 'unknown' valent 0)."""
    return sum(VALUE_MILLIMES.get(normalize_class(n), 0) for n in names)


def format_tnd(millimes: int | float) -> str:
    return f"{millimes / 1000:.3f} TND"
