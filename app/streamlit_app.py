"""SmartCoin — interface Streamlit.

Lancer depuis la RACINE du dépôt :   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src.common.draw_utils import draw_detections  # noqa: E402
from src.common.labels import (CLASS_NAMES, SHORT_LABEL, UNKNOWN, VALUE_MILLIMES,  # noqa: E402
                               count_by_class, format_tnd, total_millimes)

st.set_page_config(page_title="SmartCoin", page_icon="🪙", layout="wide")

MODE_AI, MODE_CLASSIC, MODE_BOTH = "IA (YOLOv8)", "Classique (OpenCV)", "Comparer les deux"
DEFAULT_WEIGHTS = str(Path.home() / "Desktop" / "School things" / "lasisa_things" / "IRM 2_2" / "Computer vision" / "SmartCoin" / "smartcoin" / "models" / "yolov8_smartcoin_best.pt")


@st.cache_resource(show_spinner=False)
def get_model(weights: str):
    from src.ai.predict import load_model
    return load_model(weights)


def run_ai(img_bgr: np.ndarray, weights: str, conf: float):
    from src.ai.predict import predict_image
    model = get_model(weights)
    t0 = time.perf_counter()
    det = predict_image(model, img_bgr, conf=conf)
    return det, (time.perf_counter() - t0) * 1000


def run_classical(img_bgr: np.ndarray):
    from src.evaluation.classical_runner import run_on_images
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.jpg"
        cv2.imwrite(str(path), img_bgr)
        det, ms = run_on_images([path])
    return det.drop(columns=["image_id"]), ms


def show_result(title: str, img_bgr: np.ndarray, det: pd.DataFrame, ms: float) -> None:
    st.subheader(title)
    st.image(cv2.cvtColor(draw_detections(img_bgr, det), cv2.COLOR_BGR2RGB), use_container_width=True)

    names = list(det["class_name"])
    n_unknown = names.count(UNKNOWN)
    c1, c2, c3 = st.columns(3)
    c1.metric("Montant total", format_tnd(total_millimes(names)))
    c2.metric("Pièces détectées", len(names))
    c3.metric("Temps", f"{ms:.0f} ms")
    if n_unknown:
        st.warning(f"{n_unknown} pièce(s) non reconnue(s) : exclue(s) du montant total.")

    counts = count_by_class(names)
    table = pd.DataFrame({
        "Valeur": [SHORT_LABEL[c] for c in CLASS_NAMES],
        "Nombre": [counts[c] for c in CLASS_NAMES],
        "Sous-total (TND)": [counts[c] * VALUE_MILLIMES[c] / 1000 for c in CLASS_NAMES],
    })
    st.dataframe(table[table["Nombre"] > 0], hide_index=True, use_container_width=True)
    with st.expander("Détail des détections"):
        st.dataframe(det.round(1), hide_index=True, use_container_width=True)
        st.download_button("Télécharger en CSV", det.to_csv(index=False).encode("utf-8"),
                           file_name="detections.csv", mime="text/csv", key=f"dl_{title}")


def load_uploaded(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")  # photos de téléphone
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.title("SmartCoin")
    st.caption("Tri et comptage des pièces tunisiennes")
    mode = st.radio("Méthode", [MODE_AI, MODE_CLASSIC, MODE_BOTH])
    conf = st.slider("Seuil de confiance (IA)", 0.05, 0.90, 0.25, 0.05,
                     help="Plus il est élevé, moins l'IA détecte de pièces douteuses.")
    weights = st.text_input("Poids YOLOv8", DEFAULT_WEIGHTS)
    if mode != MODE_CLASSIC and not (ROOT / weights).exists():
        st.error(f"Poids introuvables : {weights}. Entraîner le modèle (guide, étape L6) "
                 "ou choisir la méthode classique.")

# ---------------------------------------------------------------- page principale
st.header("Compter les pièces d'une photo")
tab_up, tab_cam = st.tabs(["Importer une photo", "Prendre une photo"])
with tab_up:
    uploaded = st.file_uploader("Photo contenant une ou plusieurs pièces", type=["jpg", "jpeg", "png"])
with tab_cam:
    shot = st.camera_input("Caméra")
source = uploaded or shot

if source is None:
    st.info("Importez une photo pour lancer la détection.")
    st.stop()

img = load_uploaded(source)
use_ai = mode in (MODE_AI, MODE_BOTH)
use_classic = mode in (MODE_CLASSIC, MODE_BOTH)
results: dict[str, tuple[pd.DataFrame, float]] = {}

with st.spinner("Analyse en cours…"):
    if use_ai:
        try:
            results[MODE_AI] = run_ai(img, weights, conf)
        except Exception as e:  # noqa: BLE001
            st.error(f"IA indisponible : {e}")
    if use_classic:
        try:
            results[MODE_CLASSIC] = run_classical(img)
        except Exception as e:  # noqa: BLE001
            st.error(f"Approche classique indisponible : {e}")

if not results:
    st.stop()

for col, (name, (det, ms)) in zip(st.columns(len(results)), results.items()):
    with col:
        show_result(name, img, det, ms)

if len(results) == 2:
    a, b = (total_millimes(d["class_name"]) for d, _ in results.values())
    if a == b:
        st.success("Les deux méthodes donnent le même montant.")
    else:
        st.warning(f"Écart de montant entre les deux méthodes : {abs(a - b) / 1000:.3f} TND.")
