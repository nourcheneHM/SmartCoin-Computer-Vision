# Résultats — SmartCoin

Ce dossier contiendra les sorties générées par les scripts. Il est exclu de Git
(`.gitignore`), à l'exception de ce fichier : les résultats doivent pouvoir être
**régénérés** à partir du code, des configurations et des données.

**Aucun résultat n'est disponible à ce stade.**

## Organisation prévue

```text
results/
├── classical/      # images annotées, prédictions (CSV) de l'approche classique
├── ai/             # images annotées, prédictions (CSV), courbes d'entraînement YOLO
└── comparison/     # tableaux de métriques, matrices de confusion, exemples côte à côte
```

Les tableaux finaux de comparaison (petits fichiers CSV/Markdown) pourront être ajoutés
au dépôt dans `docs/` au moment du rapport, avec la version du code qui les a produits.
