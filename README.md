# SmartCoin — Tri et comptage des pièces de monnaie tunisiennes par Vision par Ordinateur

> **Statut du projet : dataset constitué et inspecté (jalons 1 et 2).**
> Aucun modèle n'est encore implémenté ni entraîné. Les sections marquées
> « à compléter » seront mises à jour au fil du projet.

Projet universitaire — module Vision par Ordinateur (IRM2).

---

## 1. Contexte et problématique

Compter manuellement un ensemble de pièces de monnaie est une tâche répétitive et sujette
aux erreurs (caisses, distributeurs, tirelires, commerces). Les pièces tunisiennes posent
des difficultés spécifiques pour une solution automatique :

- plusieurs valeurs ont des **tailles et des couleurs proches** ;
- la **taille apparente** d'une pièce en pixels dépend de la distance et de l'angle de la caméra :
  le diamètre seul ne permet pas d'identifier une valeur ;
- les **reflets, ombres, usure, rotations** et les deux **faces différentes** d'une même pièce
  modifient son apparence ;
- les pièces peuvent être **proches, collées ou partiellement superposées**.

**Problématique :** à partir d'une photographie contenant une ou plusieurs pièces tunisiennes,
comment localiser chaque pièce, identifier sa valeur faciale, compter les pièces par catégorie
et calculer le montant total — et quelle approche (traitement d'image classique ou Deep Learning)
est la plus fiable selon les conditions de prise de vue ?

## 2. Objectifs

1. **Détecter et localiser** chaque pièce dans l'image (boîte englobante ou cercle).
2. **Identifier la valeur faciale** de chaque pièce détectée.
3. **Trier et compter** les pièces par catégorie.
4. **Calculer le montant total** en dinars tunisiens (TND).
5. **Afficher** une image annotée et un récapitulatif.
6. **Comparer expérimentalement** une approche classique (sans IA) et une approche par
   Deep Learning sur **exactement les mêmes images de test** et les mêmes annotations de référence.

### Classes visées

| Millimes | Dinars |
|---|---|
| 10, 20, 50, 100, 200, 500 | 1, 2, 5 |

La prise en charge effective de chaque classe dépendra de sa présence suffisante dans les
données (*à vérifier lors de l'inspection du dataset*). Une pièce non reconnue avec une
confiance suffisante sera signalée comme **« inconnue »** plutôt que forcée dans une classe.

## 3. Solution envisagée

```text
Image ──► Prétraitement ──► Détection des pièces ──► Identification de la valeur
      ──► Tri et comptage par classe ──► Calcul du total (TND) ──► Image annotée + récapitulatif
```

**Entrée :** une image (photo) contenant une ou plusieurs pièces tunisiennes.
**Sorties :** position de chaque pièce, valeur estimée et score de confiance/fiabilité,
nombre de pièces par catégorie, montant total en TND, image annotée.
La vidéo est une extension facultative.

Deux pipelines indépendants produiront ces mêmes sorties, ce qui permettra de les comparer.

## 4. Approches et architectures envisagées

### 4.1 Approche classique (sans IA)

Traitement d'image et règles de décision, avec **Python, OpenCV et NumPy** :

| Étape | Méthodes envisagées |
|---|---|
| Prétraitement | Niveaux de gris, espaces HSV/Lab, flou gaussien ou médian, CLAHE |
| Détection | `cv2.HoughCircles` (méthode de départ) ; en alternative : seuillage Otsu/adaptatif, morphologie mathématique, contours et composantes connexes (utile pour les pièces proches) |
| Caractéristiques | Rayon/diamètre apparent **relatif**, aire, périmètre, circularité `4π·Aire/Périmètre²`, couleur dominante en HSV |
| Classification | Règles et seuils **calibrés sur des images de développement**, combinant taille relative et couleur ; classe « inconnue » si aucune règle n'est satisfaite |

### 4.2 Approche avec IA (Deep Learning)

- **Solution principale : YOLOv8 (Ultralytics, PyTorch)** affiné (*fine-tuning*) sur nos données,
  avec **une classe par valeur faciale**. Un seul modèle réalise la détection et la classification.
  - *Justification :* pipeline simple, une seule source d'erreur, adapté au temps disponible
    et à un dataset de taille modeste grâce au transfer learning.
- **Alternative / expérience complémentaire : YOLO + CNN.** YOLO détecte une classe générique
  « pièce », puis un CNN (ResNet18 ou MobileNetV2 pré-entraîné sur ImageNet, affiné) classifie
  chaque pièce recadrée. Cette variante sera évaluée si le temps le permet, afin de mesurer
  si la séparation détection/classification apporte un gain réel.

Un seul framework de Deep Learning est utilisé : **PyTorch** (YOLOv8 repose sur PyTorch).

### 4.3 Distinction détection / classification

- **Détection** : trouver *où* se trouve chaque pièce (évaluée par IoU, précision/rappel de détection, mAP).
- **Classification** : déterminer *quelle* est la valeur d'une pièce détectée (évaluée par
  précision, rappel, F1 par classe, matrice de confusion).

## 5. Dataset

- **Source : dataset original constitué par le groupe.** Toutes les images ont été
  photographiées par les membres du groupe avec leurs smartphones, sur des pièces
  tunisiennes réelles. Aucun dataset externe n'est utilisé.
  - **342 images JPEG** (majoritairement 1536 × 2048), les **9 classes** visées présentes,
    4 fonds différents (tissu blanc, papier blanc, papier orange, tissu jaune).
  - 230 photos initiales (une pièce par photo, classées par valeur) et 112 photos
    complémentaires prises selon un protocole par niveau de difficulté (1 à 4).
  - **Jeu de test commun figé :** 30 images (10 par niveau 2, 3 et 4) contenant **220 pièces**.
  - Annotations par boîtes englobantes : *à réaliser avant le jalon 4*.
- **Références visuelles et numismatiques** (documentation uniquement, non utilisées pour l'entraînement) :
  [Wikimedia Commons — Tunisian dinar coins](https://commons.wikimedia.org/wiki/Category:Tunisian_dinar_coins),
  [Catalogue Cnumis — Tunisie](https://fr.cnumis.com/catalogue/region-images/tunisia-5).

Les images **ne sont pas versionnées** sur GitHub (taille, droits). Voir [data/README.md](data/README.md)
pour l'organisation locale et la préparation des données.

## 6. Protocole de comparaison (prévu)

1. Un **jeu de test commun figé**, listé dans [evaluation/test_manifest.csv](evaluation/test_manifest.csv),
   jamais utilisé pour entraîner le modèle IA ni pour régler les seuils classiques.
2. Un **fichier unique de vérité terrain**, [evaluation/ground_truth.csv](evaluation/ground_truth.csv),
   utilisé par les deux approches.
3. Appariement prédiction/annotation **un-à-un par IoU** (seuil initial 0,5, à justifier).
4. Métriques comparées : précision, rappel et F1 de détection ; F1 macro de classification ;
   matrice de confusion ; erreur de comptage par classe ; erreur absolue sur le montant total (TND) ;
   temps moyen par image ; résultats par **niveau de difficulté** (1 : pièce seule sur fond uni →
   4 : pièces proches ou superposées).

Détails : [evaluation/README.md](evaluation/README.md). Aucun résultat n'est disponible à ce stade.

## 7. Principales étapes de réalisation

| Jalon | Contenu | Statut |
|---|---|---|
| 1 — Dépôt GitHub | Dépôt, README, arborescence, dépendances, documentation | Terminé |
| 2 — Dataset | Collecte, inspection, organisation des images, jeu de test commun figé | Terminé (annotations des boîtes et split de validation à faire avant le jalon 4) |
| 3 — Approche classique | Détection et classification des pièces avec OpenCV et des règles | En cours (pipeline complet fonctionnel ; réglages sur les images de développement) |
| 4 — Approche IA | Entraînement et évaluation de YOLOv8 (variante YOLO + CNN si pertinent) | À faire |
| 5 — Comparaison | Les deux approches sur exactement les mêmes images de test | À faire |
| 6 — Application et rapport | Interface Streamlit, rapport, présentation | À faire |

## 8. Membres du groupe

| Nom | Prénom |
|---|---|
| Hammami | Nourchene |
| Gherab | Lasisa  |
| Bahrini| Adem |
| Maaoui | Rayene  |

## 9. Technologies et bibliothèques envisagées

- **Langage :** Python 3.12 (version retenue pour la compatibilité avec PyTorch/Ultralytics)
- **Traitement d'image :** OpenCV, NumPy
- **Données et visualisation :** Pandas, Matplotlib
- **Métriques :** scikit-learn
- **Deep Learning :** PyTorch, Ultralytics YOLOv8
- **Interface :** Streamlit
- **Environnement :** Jupyter / VS Code ; entraînement sur GPU via Kaggle ou Google Colab si nécessaire
- **Annotation :** outil à choisir (ex. Roboflow, Label Studio ou CVAT)

## 10. Installation et exécution (prévisionnelles)

```bash
# 1. Cloner le dépôt
git clone https://github.com/nourcheneHM/SmartCoin-Computer-Vision.git
cd SmartCoin-Computer-Vision

# 2. Créer et activer un environnement virtuel (Python 3.12)
py -3.12 -m venv .venv          # Windows
.venv\Scripts\activate          # Windows
# python3.12 -m venv .venv && source .venv/bin/activate   # Linux / macOS

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer les données selon data/README.md
```

Inspection du dataset (fonctionnelle) :

```bash
python -m src.data.inspect_dataset --root data/raw/group --out results/inspection
python -m src.data.inspect_dataset --root data/raw/new --out results/inspection_new
```

Approche classique (fonctionnelle) :

```bash
python -m src.classical.pipeline --input data/raw/new/train/L2      # détection + classification + montant
python -m src.classical.debug_steps --input data/raw/new/train/L4   # visualisation des étapes de détection
python -m src.classical.calibrate                                   # mesures justifiant les seuils
python -m src.evaluation.count_metrics --pred results/classical/dev/predictions.csv \
    --truth data/raw/new/train/contenu_train.csv                    # évaluation par comptage (dev)
```

Commandes prévues (**non fonctionnelles à ce stade**) :

```bash
python -m src.ai.train --config configs/yolo_data.yaml              # entraînement YOLO
python -m src.evaluation.compare --manifest evaluation/test_manifest.csv   # comparaison
streamlit run app/streamlit_app.py                                  # interface
```

## 11. Organisation du dépôt

```text
smartcoin/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/            # paramètres de l'approche classique et configuration YOLO
├── data/               # données locales (non versionnées) + README d'organisation
├── evaluation/         # manifeste du jeu de test commun + vérité terrain + protocole
├── notebooks/          # exploration et inspection du dataset
├── src/
│   ├── common/         # classes de pièces, calcul du montant, métriques, visualisation
│   ├── classical/      # pipeline OpenCV + règles (sans IA)
│   ├── ai/             # entraînement, prédiction et évaluation YOLO
│   └── evaluation/     # comparaison des deux approches
├── app/                # interface Streamlit (prévue)
├── tests/              # tests unitaires (prévus)
├── models/             # poids entraînés (non versionnés) + README
├── results/            # figures, tableaux, prédictions (non versionnés) + README
└── docs/               # références et documentation
```

Les fichiers de code actuels sont des **squelettes** documentés : ils décrivent le rôle
prévu de chaque module mais ne sont pas encore fonctionnels.

## 12. Références

Voir [docs/references.md](docs/references.md). Tout code, modèle pré-entraîné ou dataset
externe utilisé sera cité.
