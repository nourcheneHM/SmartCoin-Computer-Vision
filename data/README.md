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

- **Taille :** plusieurs centaines de photos ; un dépôt Git n'est pas conçu pour stocker
  des fichiers binaires volumineux.
- **Vie privée :** les photos originales de smartphone contiennent des **métadonnées EXIF**
  (modèle du téléphone, date, souvent **coordonnées GPS**). Elles ne doivent pas être publiées.

Seul ce fichier est suivi par Git dans `data/` (voir `.gitignore`). Les petits fichiers
nécessaires à la reproductibilité (liste des images de test, comptages de référence) sont
versionnés dans [`../evaluation/`](../evaluation/).

## 3. Caractéristiques du dataset (vérifiées par `python -m src.data.inspect_dataset`)

| Information | Valeur |
|---|---|
| Nombre total d'images | **342** (230 photos initiales + 112 photos complémentaires) |
| Classes | Les 9 valeurs visées sont présentes : 10, 20, 50, 100, 200, 500 millimes ; 1, 2, 5 dinars |
| Format | JPEG |
| Résolution | Majoritairement 1536 × 2048 (portrait) ; quelques images recadrées de tailles variables |
| Métadonnées EXIF | Absentes (aucun modèle d'appareil, aucune date, aucune coordonnée GPS) |
| Doublons | 2 doublons exacts détectés (hash MD5) dans `20_millimes` et supprimés |
| Fonds | Tissu blanc, papier blanc, papier orange, tissu jaune |
| Annotations (boîtes englobantes) | **Pas encore réalisées** — prévues avec Label Studio avant le jalon 4 |

### Photos initiales (`raw/group/`) — une pièce par photo, classées par dossier

| 10 mill. | 20 mill. | 50 mill. | 100 mill. | 200 mill. | 500 mill. | 1 dinar | 2 dinars | 5 dinars | Plusieurs pièces |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 19 | 30 | 23 | 31 | 20 | 29 | 30 | 16 | 8 |

**Biais identifié :** les conditions de prise de vue dépendent de la classe. Les pièces de
100, 20, 500 millimes et 5 dinars sont surtout photographiées de loin sur tissu blanc ;
les autres classes surtout en gros plan sur papier blanc ou orange. Un modèle pourrait
apprendre le fond ou la distance au lieu de la pièce (*shortcut learning*), et la taille
apparente n'y est pas comparable d'une photo à l'autre. Ces photos servent donc
**uniquement à l'entraînement et au développement**, jamais au test.

### Photos complémentaires (`raw/new/`) — prises selon un protocole pour corriger ce biais

| Ensemble | Niveau | Contenu | Images |
|---|---|---|---:|
| `train/L1_B1_tissu`, `L1_B2_papier_blanc`, `L1_B3_orange` | 1 | Une pièce, **hauteur fixe**, chaque classe sur chacun des 3 fonds | 27 |
| `train/L2` | 2 | Plusieurs pièces séparées | 29 |
| `train/L3` | 3 | Conditions variées : inclinaison, distance, éclairage latéral, tissu jaune | 16 |
| `train/L4` | 4 | Pièces qui se touchent ou se chevauchent | 10 |
| `test/L2`, `test/L3`, `test/L4` | 2 à 4 | **Jeu de test commun figé** (10 images par niveau) | 30 |

Les photos L1 (même distance pour toutes les classes) permettent de comparer les tailles
réelles des pièces et de calibrer l'approche classique.

## 4. Organisation locale

```text
data/
├── README.md                  # (versionné)
└── raw/                       # photos originales, jamais modifiées
    ├── group/<classe>/        # photos initiales, une pièce par photo
    ├── group/multiple_coins/
    └── new/
        ├── train/L1_<fond>/<classe>/
        ├── train/L2/  train/L3/  train/L4/
        └── test/L2/  test/L3/  test/L4/  + contenu_test.csv
```

Les noms de fichiers (identifiants uniques générés par le téléphone) sont conservés tels
quels : ils sont uniques sur l'ensemble du dataset, ce qui permet de relier images,
annotations et résultats sans ambiguïté.

## 5. Découpage des données

| Ensemble | Images | Utilisation |
|---|---|---|
| **Test commun** | 30 (`raw/new/test/`, 220 pièces) | Évaluation finale des deux approches uniquement (jalon 5) |
| Développement / entraînement | 312 (`raw/group/` + `raw/new/train/`) | Réglage de l'approche classique ; entraînement de l'approche IA |
| Validation | *à définir avant le jalon 4* | Sélection du modèle IA (extraite de l'ensemble d'entraînement) |

Règles :
- Le jeu de test n'est utilisé **ni pour entraîner** le modèle IA, **ni pour régler** les
  seuils de l'approche classique.
- La liste des images de test est figée dans
  [`evaluation/test_manifest.csv`](../evaluation/test_manifest.csv) et le nombre de pièces
  par classe de chaque image dans [`evaluation/test_counts.csv`](../evaluation/test_counts.csv).
- Pièces du test par classe : 10 mill. 15 · 20 mill. 17 · 50 mill. 36 · 100 mill. 26 ·
  200 mill. 17 · 500 mill. 41 · 1 dinar 28 · 2 dinars 20 · 5 dinars 20 (**220 au total**).

## 6. Récupérer les données

Les photos sont stockées localement par les membres du groupe, dans `data/raw/` selon
l'organisation ci-dessus. Elles ne sont pas publiées ; elles peuvent être transmises à
l'enseignant sur demande.
