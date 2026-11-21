# Données — SmartCoin

Les images **ne sont pas versionnées** sur GitHub : elles sont volumineuses et leurs
conditions de redistribution doivent être respectées. Seul ce fichier est suivi par Git
(voir `.gitignore`). Les petits fichiers de référence nécessaires à la reproductibilité
(manifeste du jeu de test, vérité terrain) sont versionnés dans [`../evaluation/`](../evaluation/).

## 1. Sources

| Source | Lien | Statut |
|---|---|---|
| Images collectées par le groupe | — (stockage local / lien de partage à compléter) | Collectées, **à inspecter** |
| Tunisian Currency Detection (Roboflow Universe) | https://universe.roboflow.com/ahmed-khgsr/tunisian-currency-detection | Envisagée, **non vérifiée** |
| Wikimedia Commons — Tunisian dinar coins | https://commons.wikimedia.org/wiki/Category:Tunisian_dinar_coins | Référence visuelle |
| Catalogue Cnumis — Tunisie | https://fr.cnumis.com/catalogue/region-images/tunisia-5 | Référence (dimensions, métal) |

## 2. Informations connues et informations à vérifier

### Images collectées par le groupe

| Information | Valeur |
|---|---|
| Nombre d'images | *à compléter* |
| Classes présentes et nombre de pièces par classe | *à compléter* |
| Format et résolution | *à compléter* |
| Appareil(s) de prise de vue | *à compléter* |
| Conditions (fond, éclairage, distance, nombre de pièces) | *à compléter* |
| Annotations (outil, format) | *à compléter* |

Les photos ne doivent contenir **aucune donnée personnelle** (visages, documents, lieux identifiables).

### Dataset Roboflow (si utilisé)

Le document de sujet mentionne **423 images annotées**. Cette information est **rapportée,
non vérifiée**. Avant toute utilisation, vérifier sur la page actuelle :

- [ ] nombre réel d'images et de versions ;
- [ ] noms des classes et correspondance avec les 9 valeurs visées ;
- [ ] type d'annotation (boîtes englobantes) et qualité des boîtes ;
- [ ] répartition train / validation / test et augmentations déjà appliquées ;
- [ ] **licence** et conditions d'utilisation / de citation ;
- [ ] doublons et diversité (fonds, éclairages, faces des pièces).

## 3. Organisation locale prévue

```text
data/
├── README.md               # (versionné)
├── raw/                    # images originales, jamais modifiées
│   ├── group/              # photos du groupe
│   └── roboflow/           # export Roboflow (si utilisé)
├── annotations/            # annotations brutes exportées par l'outil d'annotation
├── dev/                    # images de développement / calibration (seuils classiques)
└── yolo/                   # données au format YOLO générées par script
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

Convention de nommage proposée pour les photos du groupe :
`<niveau>_<session>_<numero>.jpg`, par exemple `L2_s03_0012.jpg`
(niveau de difficulté 1 à 4, session de prise de vue, numéro).

## 4. Prétraitement et splits

- Les images de `raw/` ne sont jamais modifiées ; tout prétraitement est fait par script.
- Répartition **train / validation / test** effectuée par **session de prise de vue** :
  les images quasi identiques d'une même séquence restent dans le même split
  (prévention de la fuite de données).
- Le **jeu de test commun** est figé dans `evaluation/test_manifest.csv`. Il n'est utilisé
  ni pour l'entraînement du modèle IA, ni pour le réglage des seuils de l'approche classique
  (qui utilise `dev/`).
- Niveaux de difficulté :
  1. une pièce, fond uni, éclairage stable ;
  2. plusieurs pièces séparées ;
  3. variations de rotation, distance, luminosité et fond ;
  4. pièces proches, partiellement occultées ou superposées.

## 5. Récupérer les données

*À compléter* : lien de téléchargement des photos du groupe (si partagées) et procédure
d'export Roboflow (version, format « YOLOv8 »). Aucune clé API ne doit être écrite dans le dépôt.
