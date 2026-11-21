# Protocole d'évaluation commun

Ce dossier garantit une **comparaison équitable** entre l'approche classique et l'approche IA :
les deux pipelines sont exécutés sur **les mêmes fichiers images** et évalués avec
**la même vérité terrain**.

## Fichiers

| Fichier | Rôle |
|---|---|
| `test_manifest.csv` | Liste figée des 30 images du jeu de test commun |
| `test_counts.csv` | Nombre de pièces par classe dans chaque image de test (noté à la prise de vue) et montant total en millimes |
| `ground_truth.csv` | Annotations de référence, une ligne par pièce (*à remplir après l'annotation des boîtes*) |

Ces fichiers sont petits et **versionnés**. Toute modification du jeu de test doit faire
l'objet d'un commit explicite.

### `test_manifest.csv`

| Colonne | Description |
|---|---|
| `image_id` | Identifiant unique de l'image |
| `file_name` | Chemin relatif depuis `data/` |
| `difficulty_level` | 1 à 4 (voir `data/README.md`) |
| `source` | Origine des images (`photos_groupe`) |
| `notes` | Remarques éventuelles |

### `ground_truth.csv`

| Colonne | Description |
|---|---|
| `image_id` | Référence vers `test_manifest.csv` |
| `coin_id` | Numéro de la pièce dans l'image |
| `x_min, y_min, x_max, y_max` | Boîte englobante en pixels (image originale) |
| `class_name` | Valeur faciale (ex. `500_millimes`, `1_dinar`) |
| `occluded` | `0` = pièce entière visible, `1` = partiellement cachée |

## Règles

1. Le jeu de test n'est **jamais** utilisé pour entraîner le modèle IA ni pour régler les seuils classiques.
2. Aucun prétraitement favorisant une seule méthode n'est appliqué aux images de test.
3. Appariement prédiction ↔ vérité terrain **un-à-un** par IoU (seuil initial **0,5**) :
   prédiction non appariée = faux positif ; pièce non détectée = faux négatif.
4. Une classe présente dans le test mais absente de l'entraînement est signalée explicitement.

## Métriques prévues

| Critère | Approche classique | Approche IA |
|---|---:|---:|
| Précision de détection | À mesurer | À mesurer |
| Rappel de détection | À mesurer | À mesurer |
| F1 détection | À mesurer | À mesurer |
| F1 classification macro | À mesurer | À mesurer |
| Erreur moyenne de comptage | À mesurer | À mesurer |
| Erreur absolue moyenne sur le total (TND) | À mesurer | À mesurer |
| Temps moyen par image (ms) | À mesurer | À mesurer |
| Résultats sur images simples (niveaux 1–2) | À mesurer | À mesurer |
| Résultats sur images difficiles (niveaux 3–4) | À mesurer | À mesurer |

Un tableau par classe (9 valeurs) et une matrice de confusion par approche complèteront l'analyse.
