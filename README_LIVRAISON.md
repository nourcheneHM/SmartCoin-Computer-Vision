# SmartCoin — Livraison : partie Lasisa (IA & interface) + partie Adem (comparaison & tests)

Ce zip complète le dépôt après le travail de Nourchene (jalons 1 à 3).
**Les chemins du zip sont les chemins du dépôt** : on le décompresse à la racine.

| Qui | Mission | Jalons |
|---|---|---|
| **Lasisa** | Préparation des données (annotations, split), entraînement YOLOv8, interface Streamlit | 2 (fin), 4, 6 (app) |
| **Adem** | Comparaison classique vs IA sur le test commun, tests, analyse pour le rapport | 5 |

Ordre de travail : **Étape 0 (ensemble) → Lasisa L1–L7 → Adem A1–A6** (Adem peut commencer A1–A2
pendant que Lasisa annote et entraîne). Lasisa L8–L9 (Streamlit) se fait en parallèle d'Adem.

---

## 0. Contenu du zip et où chaque fichier va

| Fichier | Rôle | Responsable |
|---|---|---|
| `src/common/labels.py` | classes, normalisation, comptage, montant (autonome) | partagé |
| `src/common/draw_utils.py` | dessin des détections | Lasisa |
| `src/evaluation/classical_runner.py` | **pont** vers le pipeline de Nourchene, sortie au format standard | partagé |
| `src/data/auto_label_single.py` | annotation automatique des 257 photos à une pièce | Lasisa |
| `src/data/prelabel.py` | tâches Label Studio (pré-annotées pour le train, vides pour le test) | Lasisa |
| `src/data/labelstudio_to_csv.py` | export Label Studio → CSV, contrôle des comptages, `ground_truth.csv` | Lasisa |
| `src/data/prepare_yolo_dataset.py` | split train/val, labels YOLO, `configs/yolo_data.yaml` | Lasisa |
| `src/ai/train.py`, `src/ai/predict.py` | entraînement et inférence YOLOv8 (remplacent les squelettes) | Lasisa |
| `app/streamlit_app.py`, `.streamlit/config.toml` | interface (remplace le squelette) | Lasisa |
| `configs/yolo_train.yaml`, `configs/label_studio_config.xml` | hyperparamètres, interface d'annotation | Lasisa |
| `src/evaluation/matching.py` | IoU, appariement un-à-un, métriques | Adem |
| `src/evaluation/compare.py` | comparaison finale (remplace le squelette) | Adem |
| `tests/conftest.py`, `test_labels.py`, `test_matching.py` | 6 tests essentiels | Adem |

**Format standard des prédictions** (utilisé par les deux approches) :
`image_id, x_min, y_min, x_max, y_max, class_name, score`. Noms de classes :
`10_millimes, 20_millimes, 50_millimes, 100_millimes, 200_millimes, 500_millimes, 1_dinar, 2_dinars, 5_dinars`.

### Installation (Lasisa, une seule fois, puis push pour qu'Adem récupère)

```bash
cd SmartCoin-Computer-Vision
git pull
git checkout -b ia-app-comparaison
# décompresser À LA RACINE (Windows 10+ : tar -xf ; Linux/macOS : unzip -o)
tar -xf smartcoin_part2_part3.zip
pip install -r requirements.txt
pip install ultralytics streamlit scikit-learn pandas matplotlib pyyaml pillow pytest
git add . && git commit -m "Ajout parties IA, interface et comparaison" && git push -u origin ia-app-comparaison
```

Les fichiers de Nourchene ne sont pas touchés : seuls les squelettes `src/ai/`, `compare.py`,
`app/` et `tests/` sont remplacés. Le dossier `.streamlit/` est caché (afficher les fichiers cachés).
Si Python dit `ModuleNotFoundError: src...`, lancer depuis la **racine** avec `python -m` et, au besoin,
ajouter un fichier vide `__init__.py` dans le dossier concerné.

**Label Studio** s'installe dans un environnement séparé (dépendances lourdes) :
`python -m venv .venv-ls`, l'activer, puis `pip install label-studio`.

---

## Étape 0 — Vérifier le pont avec le pipeline de Nourchene (Lasisa + Adem, 10 min)

Je n'avais pas accès au dossier `src/classical/` : le pont `classical_runner.py` devine les noms de
colonnes de son `predictions.csv` (image, boîtes `x_min..y_max` **ou** cercles `cx, cy, r`, classe, score).
**Vérifier avant de continuer :**

```bash
python -m src.classical.pipeline --input data/raw/new/train/L2
python -m src.evaluation.classical_runner --from-csv results/classical/dev/predictions.csv
```

- Le programme affiche la correspondance détectée, 8 lignes converties et le nombre de détections par
  classe. **Contrôler que les classes ont du sens** (ex. `500_millimes`, pas tout en `unknown`).
- Si erreur « colonnes introuvables » : ouvrir `src/evaluation/classical_runner.py`, ajouter les vrais
  noms de colonnes dans le dictionnaire `ALIASES` (en haut du fichier), relancer. C'est le **seul**
  point de couplage avec le code de Nourchene ; ne rien modifier dans `src/classical/`.
- Si son pipeline n'écrit pas `results/classical/**/predictions.csv`, le lancer à la main et utiliser
  `--from-csv` (disponible dans `auto_label_single.py`, `prelabel.py` et `classical_runner.py`, où on peut y ajouter `--manifest evaluation/test_manifest.csv --out results/classical/test` pour filtrer sur le jeu de test).

---

# PARTIE LASISA — Préparation IA & données + Interface Streamlit

## L1. Annoter automatiquement les photos à une pièce (≈ 10 min)

Les 230 photos `raw/group/<classe>/` + 27 photos `L1` ont une classe connue (le dossier). La boîte vient
de la détection classique.

```bash
python -m src.data.auto_label_single --data-root data
```

Produit `data/annotations/annotations_auto_single.csv`, `needs_review.csv` et des **planches de contrôle**
`results/inspection/auto_label_<classe>.jpg`.

- **Ouvrir les 9 planches** : chaque cadre rouge doit entourer la pièce. Si une image est fausse, la
  mettre dans `needs_review.csv` (ligne avec `file_name`) et supprimer sa ligne de `annotations_auto_single.csv`.
- Un message « Dossiers ignorés » signale un dossier de classe au nom non reconnu : le renommer
  (`10_millimes`, `1_dinar`, `2_dinars`…).

## L2. Annoter les photos multi-pièces d'entraînement dans Label Studio (≈ 1–2 h)

Cibles : `raw/group/multiple_coins` (8), `train/L2`, `L3`, `L4` (55), plus `needs_review.csv`.

```bash
python -m src.data.prelabel --data-root data --out data/annotations/ls_tasks_train.json --input data/raw/group/multiple_coins data/raw/new/train/L2 data/raw/new/train/L3 data/raw/new/train/L4 data/annotations/needs_review.csv
```

Lancer Label Studio (PowerShell ; sous Linux/macOS remplacer par `export`) :

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\chemin\vers\SmartCoin-Computer-Vision\data"
label-studio start
```

1. Créer le projet « smartcoin_train », *Labeling Interface → Code* : coller `configs/label_studio_config.xml`.
2. *Import* → `ls_tasks_train.json`.
3. *Settings → Annotation* → activer **Use predictions to prelabel tasks**, version « classique ».
4. Si les images ne s'affichent pas : *Settings → Cloud Storage → Add Source Storage → Local files*,
   chemin absolu `...\data\raw`, sans synchroniser.
5. Pour chaque image : corriger les boîtes (une boîte par pièce, serrée) **et surtout les classes**
   (la classification classique se trompe ~25 % du temps). Les boîtes grises `a_verifier` sont à
   reclasser. Cocher `occluded` si la pièce est partiellement cachée. Puis *Submit*.
6. *Export → JSON* (pas JSON-MIN) → `data/annotations/ls_export_train.json`.

## L3. Annoter le jeu de TEST (30 images, 220 pièces ≈ 1 h) — **sans pré-annotation**

Le test est annoté **à la main, de zéro** : pré-annoter avec l'approche classique biaiserait les boîtes
en sa faveur.

```bash
python -m src.data.prelabel --data-root data --out data/annotations/ls_tasks_test.json --input data/raw/new/test --no-predictions
```

Nouveau projet « smartcoin_test » (même configuration XML), importer `ls_tasks_test.json`, annoter les
30 images, *Export → JSON* → `data/annotations/ls_export_test.json`. **Adem relit les 30 images**
(deuxième regard) : la vérité terrain conditionne toute la comparaison.

## L4. Convertir les exports et contrôler les comptages

```bash
python -m src.data.labelstudio_to_csv --export data/annotations/ls_export_train.json --data-root data --out data/annotations/annotations_train_multi.csv --counts data/raw/new/train/contenu_train.csv
python -m src.data.labelstudio_to_csv --export data/annotations/ls_export_test.json --data-root data --out data/annotations/annotations_test.csv --ground-truth evaluation/ground_truth.csv --manifest evaluation/test_manifest.csv --counts evaluation/test_counts.csv
```

Le contrôle compare, image par image, le nombre annoté par classe au comptage noté à la prise de vue.
**Chaque ligne « écart … » est une erreur d'annotation à corriger dans Label Studio**, puis ré-exporter.
Objectif : « tout concorde ». (Si « aucune colonne de classe reconnue » : le format du CSV de comptage est
différent, comparer à la main.) Le second appel écrit `evaluation/ground_truth.csv` (220 lignes attendues).

## L5. Construire le jeu YOLO (train/val)

```bash
python -m src.data.prepare_yolo_dataset --annotations data/annotations/annotations_auto_single.csv data/annotations/annotations_train_multi.csv --data-root data --manifest evaluation/test_manifest.csv --out data/yolo --max-side 1280
```

Le script **refuse** toute image du test (garde-fou), fait un split de validation de 15 % stratifié par
dossier (graine 42), réduit les images à 1280 px (labels inchangés, normalisés) et écrit
`configs/yolo_data.yaml`. Vérifier le tableau affiché : **chaque classe doit avoir des instances en
validation** (sinon `--val-frac 0.2`).

## L6. Entraîner YOLOv8 (GPU conseillé)

Local avec GPU :

```bash
python -m src.ai.train --data configs/yolo_data.yaml --config configs/yolo_train.yaml
```

Sans GPU → **Kaggle** (GPU gratuit) : zipper `data/yolo`, l'ajouter comme *Dataset* Kaggle, ouvrir un
notebook avec GPU, puis (le code doit avoir été poussé sur GitHub) :

```python
!pip install ultralytics pyyaml
!git clone -b ia-app-comparaison https://github.com/nourcheneHM/SmartCoin-Computer-Vision.git
%cd SmartCoin-Computer-Vision
!python -m src.ai.train --data configs/yolo_data.yaml --data-root /kaggle/input/<nom-du-dataset>/yolo --export /kaggle/working/yolov8_smartcoin_best.pt
```

(`<nom-du-dataset>/yolo` : ajuster selon l'arborescence réellement montée ; un avertissement de cache
non inscriptible est sans conséquence.) Télécharger `yolov8_smartcoin_best.pt` et le placer dans `models/`.
Colab fonctionne de la même façon.

Si « CUDA out of memory » : `batch: 4` ou `imgsz: 800` dans `configs/yolo_train.yaml`, ou `yolov8n.pt`.
**Ne jamais régler un hyperparamètre en regardant le test** : seule la validation compte.
Garder pour le rapport : `runs/smartcoin/yolov8_coins/results.csv`, `confusion_matrix.png`, mAP50/mAP50-95.
Compléter `models/README.md` (modèle de départ, hyperparamètres, matériel, commande, version du dataset).

## L7. Prédire sur le test (vérification Lasisa)

```bash
python -m src.ai.predict --weights models/yolov8_smartcoin_best.pt --manifest evaluation/test_manifest.csv --data-root data --out results/ai/test
```

Regarder 5–6 images de `results/ai/test/annotated/`. Le seuil de confiance (`--conf`, défaut 0,25) se
choisit sur la validation (`--input data/yolo/images/val`), **jamais sur le test**.
Envoyer à Adem : `models/yolov8_smartcoin_best.pt` (Drive/Teams) et pousser `evaluation/ground_truth.csv`.

## L8. Interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

À lancer **depuis la racine du dépôt**. Trois modes : IA, classique, comparer les deux (image annotée,
tableau par valeur, montant total en TND, temps, export CSV). Si les poids manquent, le mode classique
marche quand même. Tester avec 3 photos (une pièce, plusieurs pièces, pièces collées) et faire des
**captures d'écran pour le rapport**. Le mode classique prend quelques secondes (il relance le pipeline de Nourchene à chaque photo).

## L9. Commits et documentation

`models/README.md` (statut « Entraîné »), README principal : jalon 4 → « Terminé », « Commandes prévues »
→ fonctionnelles. Les CSV d'annotations sont dans `data/` (ignoré par Git) mais **petits et sans
images** : `git add -f data/annotations/*.csv evaluation/ground_truth.csv`. Ne **jamais** versionner
les images ni les `.pt`.

---

# PARTIE ADEM — Comparaison et tests

## A1. Tests (2 min)

```bash
pytest tests -q
```

6 tests : normalisation des classes et montant, IoU, appariement un-à-un, métriques et erreur de montant.
Ils doivent tous passer avant toute comparaison.

## A2. Approche classique sur le test commun

**Aucun paramètre de `configs/classical.yaml` ne doit être modifié** (ils sont figés). Il faut
`evaluation/ground_truth.csv` (Lasisa, étape L4) pour l'étape A4, mais pas pour celle-ci.

```bash
python -m src.evaluation.classical_runner --manifest evaluation/test_manifest.csv --data-root data --out results/classical/test
```

Le pont copie exactement les 30 images du manifeste dans un dossier temporaire, lance le pipeline,
convertit en `results/classical/test/predictions_std.csv` et note le temps moyen (`timings.csv`).

## A3. Approche IA sur le **même** test, **sur la même machine**

Pour que les temps soient comparables, lancer aussi l'IA sur ta machine (poids reçus de Lasisa dans `models/`) :

```bash
python -m src.ai.predict --weights models/yolov8_smartcoin_best.pt --manifest evaluation/test_manifest.csv --data-root data --out results/ai/test
```

## A4. Comparaison

```bash
python -m src.evaluation.compare --manifest evaluation/test_manifest.csv --ground-truth evaluation/ground_truth.csv --classical-dir results/classical/test --ai-dir results/ai/test --out results/comparison
```

Le script s'arrête avec un message clair si la vérité terrain est incomplète. Sorties :

| Fichier (`results/comparison/`) | Pour le rapport |
|---|---|
| `summary.md` / `summary.csv` | tableau principal (reprend `evaluation/README.md`), à coller tel quel |
| `by_level.csv` | résultats par niveau 2/3/4 et groupes « simples » / « difficiles » |
| `per_class_*.csv` | précision/rappel/F1 par valeur, comptage prédit vs réel |
| `confusion_*.png` | matrices de confusion (4.3 : classification) |
| `per_image_*.csv` | erreurs image par image → choisir les exemples d'analyse |

Définitions (identiques pour les deux approches, à citer dans le rapport) : appariement un-à-un par IoU ≥ 0,5
sans tenir compte de la classe ; classification évaluée sur les paires appariées (`unknown` = erreur) ;
erreur de comptage = somme sur les 9 classes de |prédit − réel| par image ; erreur de montant en TND
(`unknown` exclues du total).

## A5. Analyse (c'est ce qui fait la qualité du rapport)

1. **Par niveau** : où chaque approche s'effondre-t-elle (niveau 4 : pièces collées ?) — `by_level.csv`.
2. **Confusions** : quelles valeurs sont confondues (tailles/couleurs proches : 10/20 mill., 100/200/500…) ?
   — `confusion_*.png`. Relier à l'approche classique (règles relatives au fond et à l'échelle).
3. **Erreurs concrètes** : trier `per_image_*.csv` par `count_abs_error`, retenir 4–6 images (succès et
   échecs des deux approches), les montrer côte à côte (`results/ai/test/annotated/` + sortie classique).
4. **Limites** : 30 images / 220 pièces seulement (écarts de quelques points ≠ preuve) ; chiffres de
   développement classiques (75 %) optimistes car seuils réglés sur ces photos ; vérité terrain annotée
   par le groupe ; temps classique inclut le démarrage Python (comparaison approximative).
5. **Conclusion** : quelle approche selon les conditions de prise de vue (réponse à la problématique).

## A6. Finalisation

Remplir le tableau de `evaluation/README.md` avec les chiffres de `summary.md`, copier
`summary.md`, `by_level.csv` et les `confusion_*.png` dans `docs/` avec le hash du commit du code
(`git rev-parse --short HEAD`), commit + push.

---

## Règles à ne pas enfreindre

1. Le jeu de test ne sert **jamais** à entraîner, à choisir `--conf` ni à régler un seuil classique.
2. Pas de prétraitement qui favorise une seule méthode sur les images de test.
3. Si on modifie le test (annotations corrigées), **ré-exécuter** A2–A4 en entier.
4. Un résultat « décevant » est un résultat : l'analyser, ne pas le retoucher.

## Dépannage rapide

| Symptôme | Cause / solution |
|---|---|
| `colonnes introuvables` dans `classical_runner` | adapter `ALIASES` (étape 0) |
| `étiquettes non valides … a_verifier` | reclasser ces boîtes dans Label Studio, ré-exporter |
| `ERREUR : … images du jeu de test sont dans les annotations d'entraînement` | retirer ces images des CSV d'entraînement |
| `Vérité terrain incomplète` dans `compare` | annotation du test non terminée (L3–L4) |
| Streamlit : « Poids introuvables » | placer le `.pt` dans `models/` ou choisir « Classique » |
| Streamlit : « Approche classique indisponible » | refaire l'étape 0 (format du CSV de Nourchene) |

## Ce qui n'est pas inclus

- **Variante YOLO + CNN** (README §4.2, « si le temps le permet ») : non implémentée. Elle se brancherait
  ainsi : entraîner un détecteur à une classe, recadrer les boîtes, classifier avec un CNN, puis écrire le même
  `predictions_std.csv` pour que `compare.py` la prenne en charge sans changement (ajouter un `--ai2-dir`).
- Le script a été validé sur des données synthétiques avec un faux pipeline classique et un faux
  `ultralytics` ; l'entraînement YOLO réel, Label Studio et le vrai `predictions.csv` de Nourchene n'ont pas
  pu être exécutés ici : d'où l'étape 0 et les contrôles intégrés (comptages, garde-fou du test).
