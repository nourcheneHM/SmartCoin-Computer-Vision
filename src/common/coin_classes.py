"""Définition des classes de pièces tunisiennes visées par SmartCoin.

Toutes les valeurs sont exprimées en **millimes** (1 dinar = 1000 millimes) afin
d'éviter les erreurs de conversion entre millimes et dinars lors du calcul du total.

La liste est provisoire : elle sera confirmée après l'inspection du dataset.
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
    "1_dinar": 1000,
    "2_dinars": 2000,
    "5_dinars": 5000,
}

CLASS_NAMES: list[str] = list(COIN_VALUES_MILLIMES)
