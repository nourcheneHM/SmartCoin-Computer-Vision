# Données — SmartCoin

## 1. Source

Le dataset SmartCoin est **original** : toutes les images ont été photographiées par les
membres du groupe avec leurs smartphones, sur des pièces tunisiennes réelles.
**Aucun dataset externe n'est utilisé** pour l'entraînement ou l'évaluation.

Ressources consultées uniquement comme documentation (identification des pièces,
dimensions, métal) :
- [Wikimedia Commons — Tunisian dinar coins](https://commons.wikimedia.org/wiki/Category:Tunisian_dinar_coins)
- [Catalogue Cnumis — Tunisie](https://fr.cnumis.com/catalogue/region-images/tunisia-5)

## 2. Pourquoi les images ne sont pas sur GitHub

- **Taille :** les photos de smartphone pèsent plusieurs Mo chacune ; un dépôt Git n'est
  pas conçu pour stocker des centaines de fichiers binaires volumineux.
- **Vie privée :** les photos de smartphone contiennent des **métadonnées EXIF** (modèle du
  téléphone, date et souvent **coordonnées GPS**). Elles ne doivent pas être publiées telles quelles.

Seul ce fichier est suivi par Git dans `data/` (voir `.gitignore`). Les petits fichiers
nécessaires à la reproductibilité (manifeste du jeu de test, vérité terrain) sont versionnés
dans [`../evaluation/`](../evaluation/).

## 3. Caractéristiques du dataset

| Information | Valeur |
|---|---|
| Nombre d'images | *à compléter après inspection* |
| Classes présentes et nombre de pièces par classe | *à compléter* |
| Format et résolution | *à compléter* |
| Smartphones utilisés | *à compléter* |
| Conditions (fond, éclairage, distance, nombre de pièces par image) | *à compléter* |
| Annotations (outil, format) | *à compléter* |

Les photos ne doivent contenir **aucune donnée personnelle** (visages, documents, lieux identifiables).

## 4. Organisation locale

```text
data/
├── README.md               # (versionné)
├── raw/                    # photos originales, jamais modifiées
│   └── group/
├── annotations/            # annotations exportées par l'outil d'annotation
├── dev/                    # images de développement / calibration (seuils classiques)
└── yolo/                   # données au format YOLO générées par script
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

Convention de nommage proposée (appliquée par script, sans modifier les originaux) :
`<niveau>_<session>_<numero>.jpg`, par exemple `L2_s03_0012.jpg`
(niveau de difficulté 1 à 4, session de prise de vue, numéro).

## 5. Prétraitement et splits

- Les images de `raw/` ne sont jamais modifiées ; tout prétraitement est fait par script.
- Répartition **train / validation / test** effectuée par **session de prise de vue** :
  les photos quasi identiques d'une même séquence restent dans le même split
  (prévention de la fuite de données).
- Le **jeu de test commun** est figé dans `evaluation/test_manifest.csv`. Il n'est utilisé
  ni pour l'entraînement du modèle IA, ni pour le réglage des seuils de l'approche classique
  (qui utilise `dev/`).
- Niveaux de difficulté :
  1. une pièce, fond uni, éclairage stable ;
  2. plusieurs pièces séparées ;
  3. variations de rotation, distance, luminosité et fond ;
  4. pièces proches, partiellement occultées ou superposées.

## 6. Récupérer les données

Les membres du groupe placent les photos dans `data/raw/group/`.
Partage éventuel (enseignant, autres membres) : *lien à compléter* — avec métadonnées EXIF supprimées.
