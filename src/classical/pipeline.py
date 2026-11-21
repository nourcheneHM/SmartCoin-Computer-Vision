"""Pipeline de l'approche classique (sans IA).

Étapes prévues : prétraitement (gris/HSV, flou, CLAHE) -> détection (HoughCircles
ou contours) -> extraction de caractéristiques (taille relative, couleur HSV,
circularité) -> classification par règles -> comptage et montant total.
Paramètres : configs/classical.yaml.

Statut : SQUELETTE — module non implémenté (phase de cadrage du projet).
"""


def main() -> None:
    raise NotImplementedError("Module prévu, pas encore implémenté.")


if __name__ == "__main__":
    main()
