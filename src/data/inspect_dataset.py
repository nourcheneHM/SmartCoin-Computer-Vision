"""Inspection du dataset SmartCoin (photos du groupe).

Parcourt data/raw/group/<classe>/*.jpg et produit, sans modifier aucune image :
- un tableau par image (classe, résolution, orientation, taille, métadonnées EXIF) ;
- la détection des doublons exacts (hash MD5) et des quasi-doublons (hash perceptuel dHash) ;
- un résumé imprimé dans le terminal.

Le tableau détaillé est écrit dans results/inspection/ (non versionné), car il peut
contenir des métadonnées personnelles (modèle de téléphone, date, présence de GPS).

Usage :
    python -m src.data.inspect_dataset
    python -m src.data.inspect_dataset --root data/raw/group --out results/inspection
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXIF_MODEL = 0x0110        # modèle de l'appareil
EXIF_DATETIME = 0x0132     # date de prise de vue
EXIF_ORIENTATION = 0x0112  # rotation à appliquer à l'affichage
EXIF_GPS_IFD = 0x8825      # bloc des coordonnées GPS


def md5_of_file(path: Path) -> str:
    """Hash MD5 du fichier : identique uniquement si les octets sont identiques."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Hash perceptuel « difference hash ».

    L'image est réduite à (size+1) x size pixels, puis chaque bit indique si un pixel
    est plus clair que son voisin de droite. Deux photos presque identiques
    (même scène, léger décalage) ont des hash proches (peu de bits différents).
    """
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_exif(path: Path) -> dict:
    """Lit les métadonnées EXIF utiles sans modifier le fichier."""
    with Image.open(path) as im:
        exif = im.getexif()
        gps = exif.get_ifd(EXIF_GPS_IFD) if EXIF_GPS_IFD in exif else {}
        return {
            "camera_model": exif.get(EXIF_MODEL),
            "datetime": exif.get(EXIF_DATETIME),
            "exif_orientation": exif.get(EXIF_ORIENTATION),
            "has_gps": bool(gps),
        }


def inspect_image(path: Path, root: Path) -> dict:
    """Retourne les caractéristiques d'une image."""
    # cv2.imread applique l'orientation EXIF : (h, w) correspond à l'image telle qu'affichée.
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"file": str(path.relative_to(root)), "readable": False}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "file": path.relative_to(root).as_posix(),
        "class_folder": path.parent.name,
        "readable": True,
        "width": w,
        "height": h,
        "orientation": "portrait" if h > w else "paysage" if w > h else "carré",
        "size_kb": round(path.stat().st_size / 1024, 1),
        "mean_brightness": round(float(gray.mean()), 1),
        "md5": md5_of_file(path),
        "dhash": dhash(gray),
        **read_exif(path),
    }


def find_near_duplicates(df: pd.DataFrame, max_bits: int = 4) -> list[tuple[str, str, int]]:
    """Paires d'images dont les dHash diffèrent d'au plus `max_bits` bits (sur 64)."""
    hashes = df["dhash"].tolist()
    files = df["file"].tolist()
    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = bin(hashes[i] ^ hashes[j]).count("1")
            if dist <= max_bits:
                pairs.append((files[i], files[j], dist))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/raw/group"))
    parser.add_argument("--out", type=Path, default=Path("results/inspection"))
    parser.add_argument("--max-bits", type=int, default=4, help="seuil quasi-doublons (bits)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows : afficher accents et symboles

    paths = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"Aucune image trouvée dans {args.root}")

    df = pd.DataFrame(inspect_image(p, args.root) for p in paths)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "images.csv", index=False)

    ok = df[df["readable"]]
    print(f"Images trouvées : {len(df)}  (illisibles : {(~df['readable']).sum()})\n")
    print("Images par dossier :")
    print(ok["class_folder"].value_counts().sort_index().to_string(), "\n")
    print("Résolutions (largeur x hauteur) :")
    print((ok["width"].astype(str) + "x" + ok["height"].astype(str)).value_counts().to_string(), "\n")
    print("Orientation :", ok["orientation"].value_counts().to_dict())
    print("Modèles d'appareil :", ok["camera_model"].fillna("inconnu").value_counts().to_dict())
    print(f"Images avec GPS dans les EXIF : {int(ok['has_gps'].sum())} / {len(ok)}")
    print(f"Taille moyenne : {ok['size_kb'].mean():.0f} Ko")
    print(f"Luminosité moyenne (0-255) : min {ok['mean_brightness'].min()}, "
          f"max {ok['mean_brightness'].max()}\n")

    exact = ok[ok.duplicated("md5", keep=False)].sort_values("md5")
    print(f"Doublons exacts : {len(exact)} fichiers")
    if not exact.empty:
        print(exact[["md5", "file"]].to_string(index=False))

    near = find_near_duplicates(ok, args.max_bits)
    pd.DataFrame(near, columns=["file_a", "file_b", "hamming_bits"]).to_csv(
        args.out / "near_duplicates.csv", index=False)
    print(f"\nPaires de quasi-doublons (≤ {args.max_bits} bits) : {len(near)}")
    for a, b, d in near[:15]:
        print(f"  {d} bits : {a}  <->  {b}")
    print(f"\nDétails : {args.out / 'images.csv'}")


if __name__ == "__main__":
    main()
