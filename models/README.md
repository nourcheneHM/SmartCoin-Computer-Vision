# Modèles — SmartCoin

Ce dossier contiendra **localement** les modèles entraînés. Les fichiers de poids
(`*.pt`, `*.pth`, `*.onnx`…) ne sont **pas versionnés** (taille ; le dossier est exclu par `.gitignore`).

## Contenu prévu

| Fichier                    | Description                                                        | Statut       |
| -------------------------- | ------------------------------------------------------------------ | ------------ |
| `yolov8_smartcoin_best.pt` | Meilleur modèle YOLOv8 (une classe par valeur) selon la validation | Non entraîné |
| `yolov8_coin_detector.pt`  | _(Variante)_ détecteur YOLO à une classe « pièce »                 | Non entraîné |
| `cnn_classifier.pt`        | _(Variante)_ CNN de classification des pièces recadrées            | Non entraîné |

L'approche classique n'a pas de poids : ses paramètres et seuils sont dans
[`../configs/classical.yaml`](../configs/classical.yaml) (versionné).

## Reproductibilité

Pour chaque modèle entraîné, seront documentés ici :

- le modèle de départ (ex. `yolov8n.pt` pré-entraîné COCO, Ultralytics) ;
- la version du dataset et des splits ;
- les hyperparamètres (epochs, batch size, taille d'image, etc.) et la commande exacte ;
- le matériel utilisé (CPU/GPU, Kaggle/Colab) ;
- un lien de téléchargement des poids si partagés (à compléter).

Modèle de départ : yolov8s.pt  
Hyperparamètres : epochs=100, imgsz=1024, batch=8, seed=42  
Matériel : Kaggle GPU Tesla T4  
Commande lancée : python -m src.ai.train --data configs/yolo_data.yaml --data-root /kaggle/input/datasets/lasisa/smartcoin-yolo/yolo --export /kaggle/working/yolov8_smartcoin_best.pt Dataset : 312 images (train + val, 9 classes de pièces tunisiennes)
