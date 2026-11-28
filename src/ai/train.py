"""Entraînement (fine-tuning) de YOLOv8 sur les pièces tunisiennes.

    python -m src.ai.train --data configs/yolo_data.yaml --config configs/yolo_train.yaml

Sur Kaggle / Colab (le jeu YOLO est ailleurs que sur le PC) :
    python -m src.ai.train --data-root /kaggle/working/yolo

Le meilleur modèle (selon la validation) est copié dans models/yolov8_smartcoin_best.pt.
Le jeu de test n'est JAMAIS utilisé ici.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="configs/yolo_data.yaml")
    ap.add_argument("--config", default="configs/yolo_train.yaml", help="hyperparamètres")
    ap.add_argument("--data-root", help="remplace `path:` du yaml de données")
    ap.add_argument("--project", default="runs/smartcoin")
    ap.add_argument("--name", default="yolov8_coins")
    ap.add_argument("--export", default="models/yolov8_smartcoin_best.pt")
    a = ap.parse_args()

    from ultralytics import YOLO  # import tardif : le reste du projet n'en dépend pas

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    base_model = cfg.pop("model", "yolov8s.pt")

    data = a.data
    if a.data_root:
        d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
        d["path"] = str(Path(a.data_root).resolve())
        data = str(Path(a.project) / "yolo_data_override.yaml")
        Path(data).parent.mkdir(parents=True, exist_ok=True)
        Path(data).write_text(yaml.safe_dump(d, allow_unicode=True))

    model = YOLO(base_model)
    model.train(data=data, project=a.project, name=a.name, exist_ok=True, **cfg)

    best = Path(model.trainer.best)
    Path(a.export).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, a.export)
    print(f"[OK] meilleur modèle : {best}\n     copié vers {a.export}")
    print("À noter dans models/README.md : modèle de départ, hyperparamètres, matériel, commande.")


if __name__ == "__main__":
    main()
