# Remarques pour le rapport — jalons 1 à 3

Ce document rassemble les constats, choix techniques et limites à reprendre dans le rapport.
**Tous les chiffres ont été mesurés** avec les scripts du dépôt (commande indiquée pour chacun).
Sauf mention contraire, ils proviennent des images de **développement** : aucun chiffre ne
provient du jeu de test, qui reste réservé à la comparaison finale (jalon 5).

---

## 1. Dataset

### 1.1 Constitution
- Dataset **original** : 342 photos JPEG prises par le groupe avec des smartphones,
  9 classes (10, 20, 50, 100, 200, 500 millimes ; 1, 2, 5 dinars), 4 fonds (tissu blanc,
  papier blanc, papier orange, tissu jaune).
- 230 photos initiales (une pièce par photo, classées par valeur) + 112 photos prises
  ensuite selon un protocole par niveau de difficulté (1 : pièce seule → 4 : pièces qui
  se touchent ou se chevauchent).
- Jeu de test commun figé : 30 images (10 par niveau 2, 3, 4), 220 pièces.
- Inspection (`python -m src.data.inspect_dataset`) : 2 doublons exacts (hash MD5) supprimés ;
  aucune métadonnée EXIF (ni modèle, ni date, ni GPS).

### 1.2 Biais identifiés (à discuter dans le rapport)
- **Biais de prise de vue dans les photos initiales** : les conditions dépendent de la
  classe. 100, 20, 500 millimes et 5 dinars surtout photographiés de loin sur tissu blanc ;
  les autres classes surtout en gros plan sur papier blanc ou orange. Un modèle pourrait
  apprendre le fond ou la distance au lieu de la pièce (*shortcut learning*).
  → Correction : photos complémentaires par protocole, et ces photos initiales ne servent
  jamais au test.
- **Tissu jaune** : dans l'ensemble d'entraînement L3, il n'apparaît qu'avec des pièces
  argentées (biais de couleur possible).
- **Hauteur « fixe » non respectée en L1** : les photos L1 devaient être prises à la même
  hauteur, mais la mesure montre le contraire (ex. le 1 dinar mesure 120 px de rayon sur
  tissu et 94 px sur papier ; sur papier, le 2 dinars paraît plus grand que le 1 dinar,
  l'inverse sur tissu). Conséquence : la taille absolue en pixels est inutilisable, ces
  photos servent à calibrer la couleur et la forme seulement.
- **Leçon** : un protocole de prise de vue doit être vérifié par la mesure, pas supposé.

### 1.3 Données de référence ajoutées
- **Diamètres réels** mesurés par le groupe sur ses pièces :
  10 mill. 19 mm · 20 mill. 22 mm · 50 mill. 25 mm · 100 mill. 27 mm · 200 mill. 29,2 mm ·
  500 mill. 24 mm · 1 dinar 28 mm · 2 dinars 29,4 mm · 5 dinars 29 mm.
  Cohérents avec les deux seules valeurs trouvées sur Wikipédia (200 mill. 29 mm,
  2 dinars 29,4 mm). Les autres diamètres n'y figurent pas : on n'a pas inventé de valeurs.
- **Contenu des photos d'entraînement multi-pièces** (`data/raw/new/train/contenu_train.csv`) :
  55 photos (L2 : 29, L3 : 16, L4 : 10), 364 pièces. Permet d'évaluer et de régler
  l'approche classique sans toucher au test.

### 1.4 Remarques numismatiques utiles à l'approche classique
- Trois groupes de couleur : **dorés** (10, 20, 50, 100, 200 millimes), **argentés**
  (500 millimes, 1 et 2 dinars), **bimétallique** (5 dinars : centre argenté, anneau doré).
- Les pièces de **200 millimes et 2 dinars (série 2013) ont 13 côtés** (Wikipédia).
- Des pièces de même couleur ont des tailles très proches : 1 dinar et 2 dinars ne
  diffèrent que de 5 % (28 / 29,4 mm), 100 et 200 millimes de 8 %.

---

## 2. Approche classique — détection

### 2.1 Pipeline
Redimensionnement (plus grand côté 1024 px) → gris + CLAHE + flou médian 7 →
deux détecteurs complémentaires fusionnés :
1. **Hough** (`cv2.HOUGH_GRADIENT_ALT`) : cercles à partir des contours.
2. **Segmentation par soustraction du fond** : couleur du fond = médiane (Lab) d'une bande
   au bord de l'image ; distance colorimétrique (L pondéré 0,5 pour atténuer les ombres) ;
   seuil d'Otsu ; morphologie ; suppression des régions qui touchent le bord ;
   **transformée de distance** : un maximum local par pièce, ce qui sépare les pièces
   collées.
3. **Fusion** : on garde les cercles de Hough et on ajoute une détection de la
   segmentation seulement si son centre n'est dans aucun cercle de Hough ; suppression des
   cercles imbriqués (ex. cœur du 5 dinars détecté comme second cercle).

### 2.2 Pourquoi deux détecteurs (chaque problème rencontré → solution)
| Problème observé | Cause | Solution |
|---|---|---|
| Hough rate les 200 millimes | bord festonné, pas un cercle | ajout de la segmentation |
| Hough se trompe sur tissu jaune | texture et faible contraste de contour | segmentation couleur |
| La segmentation détecte des zones de fond | dégradés d'éclairage | suppression des régions touchant le bord |
| La segmentation ne capte qu'un fragment d'une pièce argentée sur fond blanc | couleur proche du fond | on garde Hough en priorité (fusion) |
| Pièces collées fusionnées en une tache | seuillage | transformée de distance (un pic par pièce) |

Figure conseillée : panneau des 8 étapes
(`python -m src.classical.debug_steps --input data/raw/new/train/L4`), photo L4 avec
200 millimes : Hough trouve 4 pièces, la segmentation ajoute le 200 millimes → 5.

### 2.3 Résultats de détection mesurés
Photos de développement à une seule pièce (part des images avec exactement 1 détection) :

| Méthode | L1 | Photos initiales | Temps / image |
|---|---|---|---|
| Hough seul | 100 % | 94 % | ~230 ms |
| Segmentation seule | 100 % | 96 % | ~110 ms |
| Hybride (retenu) | 100 % | 99 % (3 échecs, tous des 500 millimes) | ~310 ms |

Photos multi-pièces L2–L4 (55 images, 364 pièces) : **363 pièces détectées pour 364**,
erreur moyenne de 0,05 pièce par image (L2 et L4 : 0 ; L3 : 0,19).
Limite : ce sont des **comptes**, pas une évaluation par IoU (boîtes pas encore annotées).

---

## 3. Approche classique — classification par règles

### 3.1 Caractéristiques (`src/classical/features.py`)
- **Couleur Lab** (indépendante de la distance) : b* (axe jaune) du disque intérieur,
  du centre et de l'anneau ; reflets exclus ; médiane (robuste aux gravures).
- **Contraste bimétal** = b*(anneau) − b*(centre).
- **Score 13 côtés** : le bord est suivi en coordonnées polaires par **programmation
  dynamique** (chemin continu), puis on mesure l'énergie de la 13e harmonique de Fourier.
- **Rayon affiné** = rayon médian du bord suivi.

### 3.2 Calibration sur pièces seules (`python -m src.classical.calibrate`, 246 pièces)
- **5 dinars** : contraste bimétal ≥ 9 pour les 5 dinars, ≤ 7 pour toutes les autres
  pièces → séparation nette.
- **Doré / argenté** : un seuil fixe sur b* fait 41 erreurs / 227 pièces ; en corrigeant
  par la couleur du fond (b − 0,75 × b_fond) : 14 erreurs / 227. Les erreurs restantes
  viennent surtout du **papier orange** (le fond se reflète sur le métal et modifie la
  balance des blancs du téléphone) et de 20 millimes foncés.
- **13 côtés** (seuil 3,5) : détecte 71 % des 200 millimes et 36 % des 2 dinars, avec
  seulement 2 fausses alarmes sur 179 pièces rondes → indice utilisé **dans un seul sens**
  (« 13 côtés » est une preuve forte ; « pas 13 côtés » n'exclut rien).
- Première tentative du score de forme (maximum de gradient par angle) : **échec**, aucune
  séparation, car le contour « saute » sur les gravures et le centre détecté est décalé.
  Le suivi par programmation dynamique a corrigé cela. (Bon exemple de démarche
  itérative à raconter.)
- **Rayon affiné** contre rayon de Hough : erreur de taille relative moyenne
  1,9 % contre 3,1 % (L2), 1,8 % contre 3,0 % (L4).

### 3.3 Règles de décision (`src/classical/classification.py`)
1. **Métal** : 5 dinars si contraste bimétal ≥ 7. Sinon doré/argenté, par priorité :
   a. si la photo contient un 5 dinars, il sert de **référence interne** (son centre est
      argenté, son anneau doré, sous le même éclairage) ;
   b. sinon, si les pièces forment deux groupes de couleur nettement séparés, on coupe
      entre les deux (**comparaison relative** dans l'image) ;
   c. sinon seuil fixe corrigé du fond.
2. **Forme** : 13 côtés → 200 millimes (doré) ou 2 dinars (argenté).
3. **Taille relative** : toutes les pièces d'une photo partagent la même échelle
   (pixels/mm). On cherche l'échelle qui explique au mieux tous les rayons par les
   diamètres réels des classes compatibles (minimisation d'un coût, erreur plafonnée pour
   rester robuste à une pièce mal détectée).
4. **Inconnue** : si le diamètre ne correspond à aucune classe compatible (écart > 7 %).
   Une confiance est affichée pour chaque pièce.

Idée clé à défendre : **la taille absolue ne veut rien dire, seuls les rapports de tailles
dans une même photo comptent** ; et la couleur doit elle aussi être jugée **relativement**
aux autres pièces de la photo.

### 3.4 Choix validés sur les photos multi-pièces (55 photos, 364 pièces)
Évaluation par comptage : `python -m src.evaluation.count_metrics`.

| Variante | Métal correct | Valeur correcte | Inconnues | Erreur montant moy. |
|---|---|---|---|---|
| Sans comparaison relative des couleurs | 246 | 158 | 160 | 2138 mill. |
| Première version (comparaison relative, rejet si confiance < 0,5) | 319 | 203 | 129 | 1970 mill. |
| + référence 5 dinars, seuil bimétal 7 | 334 | 221 | 116 | 1434 mill. |
| + plus de rejet par la confiance (**version retenue**) | 334 | **274** | 14 | **861 mill.** |

- La **comparaison relative** est la règle de couleur la plus importante (+73 pièces au bon métal).
- Le **rejet par la confiance** éliminait plus de bonnes réponses que d'erreurs : il a été
  désactivé (la confiance reste affichée).

### 3.5 Résultats de la version retenue (développement, L2–L4)

| Niveau | Images | Pièces | Détectées | Inconnues | Bien classées | Erreur montant moy. | Images exactes |
|---|---|---|---|---|---|---|---|
| L2 | 29 | 212 | 212 | 2 | 169 | 894 mill. | 3 |
| L3 | 16 | 95 | 94 | 9 | 63 | 974 mill. | 3 |
| L4 | 10 | 57 | 57 | 3 | 42 | 580 mill. | 1 |
| **Total** | 55 | 364 | 363 | 14 | **274 (75 %)** | **861 mill.** | 7 |

Par classe (F1) : 5 dinars 0,93 · 10 mill. 0,92 · 20 mill. 0,89 · 200 mill. 0,77 ·
500 mill. 0,77 · 100 mill. 0,71 · 50 mill. 0,66 · 2 dinars 0,65 · 1 dinar 0,61.
**F1 macro : 0,767.** Temps moyen : ~1,1 s par image (CPU).

Analyse :
- Les classes les mieux reconnues ont un indice **propre** : 5 dinars (bimétal),
  10 millimes (la plus petite), 20 millimes.
- Les plus faibles sont les paires de **même couleur et de taille proche** : 1 dinar /
  2 dinars (5 % d'écart, et la forme ne détecte que 36 % des 2 dinars), 50 / 100 millimes.
- Le 200 millimes est sur-prédit (49 prédits pour 34 réels) : une erreur de métal ou
  d'échelle décale toute l'interprétation des tailles de la photo.

---

## 4. Limites à mentionner honnêtement
1. **Résultats optimistes** : les seuils ont été choisis sur ces mêmes 55 photos.
   Seul le jeu de test (jalon 5) donnera une mesure non biaisée.
2. **Évaluation par comptage** : une pièce de 50 prédite 20 et une de 20 prédite 50 dans la
   même photo se compensent. L'évaluation finale utilisera l'appariement par IoU, qui
   nécessite l'annotation des boîtes.
3. **Pièce seule** : sans autre pièce de référence, la taille ne permet pas de distinguer
   les pièces de même couleur (ex. une pièce dorée seule peut être 10 ou 100 millimes).
4. **Fond orange** : la couleur du fond se reflète sur les pièces, ce qui rend la
   distinction doré/argenté difficile sans référence dans l'image.
5. **Pièce coupée par le bord** : non détectée (hypothèse du protocole : pièces entières).
6. **Pièces superposées** : la transformée de distance sépare les pièces qui se touchent,
   mais une pièce largement cachée a un rayon sous-estimé.
7. **Dépendance aux versions des pièces** : les diamètres et formes sont ceux de nos pièces ;
   une ancienne série d'une même valeur pourrait différer.

## 5. Perspectives (approche classique)
- Annoter les boîtes (Label Studio) pour une évaluation par IoU.
- Distinguer 1 / 2 dinars par d'autres indices (gravures, texture) plutôt que la taille.
- Comparer avec l'approche IA (YOLOv8) sur exactement les mêmes 30 images de test.

## 6. Reproductibilité
```bash
python -m src.classical.calibrate                                   # §3.2
python -m src.classical.pipeline --input data/raw/new/train --out results/classical/dev
python -m src.evaluation.count_metrics --pred results/classical/dev/predictions.csv \
    --truth data/raw/new/train/contenu_train.csv                    # §2.3, §3.4, §3.5
python -m src.classical.debug_steps --input data/raw/new/train/L4   # figure §2.2
```
Paramètres et justification de chaque seuil : `configs/classical.yaml`.
Le tableau §3.4 (variantes) a été obtenu avec un script d'expérience temporaire (non
versionné) qui rejoue la classification sur les mêmes détections en changeant une règle
à la fois ; seule la version retenue est dans le code.
