"""Définition des classes de pièces tunisiennes visées par SmartCoin.

Toutes les valeurs sont exprimées en **millimes** (1 dinar = 1000 millimes) afin
d'éviter les erreurs de conversion entre millimes et dinars lors du calcul du total.
"""

UNKNOWN_LABEL = "unknown"

# Nom de classe -> valeur faciale en millimes
COIN_VALUES_MILLIMES: dict[str, int] = {
    "10_millimes": 10,
    "20_millimes": 20,
    "50_millimes": 50,
    "100_millimes": 100,
    "200_millimes": 200,
    "500_millimes": 500,
    "1_dinars": 1000,
    "2_dinars": 2000,
    "5_dinars": 5000,
}

CLASS_NAMES: list[str] = list(COIN_VALUES_MILLIMES)

# Diamètres réels en mm, mesurés par le groupe sur ses propres pièces (une pièce par classe).
# Cohérents avec les deux valeurs publiées sur Wikipédia (200 millimes : 29 mm,
# 2 dinars : 29,4 mm). Seuls les RAPPORTS de diamètres sont utilisés : la taille absolue
# en pixels dépend de la distance de prise de vue.
COIN_DIAMETERS_MM: dict[str, float] = {
    "10_millimes": 19.0,
    "20_millimes": 22.0,
    "50_millimes": 25.0,
    "100_millimes": 27.0,
    "200_millimes": 29.2,
    "500_millimes": 24.0,
    "1_dinars": 28.0,
    "2_dinars": 29.4,
    "5_dinars": 29.0,
}

# Métal (couleur) de chaque pièce, observé sur nos photos.
COIN_METAL: dict[str, str] = {
    "10_millimes": "gold",
    "20_millimes": "gold",
    "50_millimes": "gold",
    "100_millimes": "gold",
    "200_millimes": "gold",
    "500_millimes": "silver",
    "1_dinars": "silver",
    "2_dinars": "silver",
    "5_dinars": "bimetal",   # centre argenté, anneau doré
}

# Pièces à 13 côtés (série 2013, source : Wikipédia « Tunisian dinar »).
POLYGONAL_CLASSES: set[str] = {"200_millimes", "2_dinars"}
