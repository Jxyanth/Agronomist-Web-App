"""Offline-first Streamlit dashboard for leaf diagnosis and soil planning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "agronomist_leaf_model.pth"
CLASS_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
IMAGE_SIZE = 224
NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
LOW_CONFIDENCE_THRESHOLD = 0.55
DISEASE_INFO = {
    "Tomato___Healthy": {
        "name": "Tomato — Healthy",
        "organic": ["Continue regular monitoring and balanced watering."],
        "chemical": ["No pesticide treatment is indicated for a healthy leaf."],
    },
    "Tomato___Early_blight": {
        "name": "Tomato — Early blight",
        "organic": [
            "At first symptoms: remove affected lower leaves and dispose of them away from the crop.",
            "Throughout the season: mulch, water at soil level, and improve airflow.",
            "Before the next planting: rotate away from tomato and potato crops where practical.",
        ],
        "chemical": [
            "When disease pressure warrants: use a locally registered tomato fungicide "
            "(for example, a copper-based product) strictly according to its label.",
            "Repeat only at the label interval; observe pre-harvest intervals, protective "
            "equipment requirements, and local resistance-management guidance.",
        ],
    },
    "Tomato___Late_blight": {
        "name": "Tomato — Late blight",
        "organic": [
            "At detection: promptly remove infected plants or tissues; do not compost diseased material.",
            "Daily during conducive weather: keep foliage dry and monitor nearby potatoes and tomatoes.",
        ],
        "chemical": [
            "At detection: seek local extension guidance promptly; late blight can spread rapidly.",
            "Apply only a locally registered late-blight product as directed; repeat only "
            "at the label interval and rotate modes of action as required.",
        ],
    },
    "Potato___Late_blight": {
        "name": "Potato — Late blight",
        "organic": [
            "At detection: remove and safely dispose of infected foliage and volunteer potatoes.",
            "Throughout the season: avoid overhead irrigation and inspect nearby susceptible crops.",
        ],
        "chemical": [
            "At detection: consult local agricultural extension and use only locally registered "
            "potato products according to their labels.",
            "Repeat only at label intervals; respect harvest intervals, protective-equipment "
            "instructions, and resistance-management directions.",
        ],
    },
}
CROP_TARGETS = {
    "Tomato": {"N": 80.0, "P": 40.0, "K": 80.0},
    "Potato": {"N": 100.0, "P": 50.0, "K": 100.0},
    "Maize": {"N": 120.0, "P": 50.0, "K": 50.0},
    "Wheat": {"N": 90.0, "P": 40.0, "K": 40.0},
}


def count_images(folder: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.exists():
        return 0
    return sum(
        1 for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def list_test_images() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_dir = DATASET_DIR / "test"
    return sorted(
        (path for path in test_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: str(path).lower(),
    ) if test_dir.exists() else []


def build_model(class_count: int) -> torch.nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, class_count)
    return model


@st.cache_resource
def load_model(model_modified_ns: int | None = None):
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        return None, [], torch.device("cpu")
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names, device


def predict(image: Image.Image, model, class_names, device):
    input_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ])(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    top_indices = np.argsort(probabilities)[::-1][: min(3, len(class_names))]
    return [
        (class_names[int(index)], float(probabilities[index]))
        for index in top_indices
    ]


def analyze_image_pixels(image: Image.Image) -> dict[str, float] | None:
    """Estimate color cues within saturated, plant-like pixels in this image."""
    resized = image.convert("RGB").resize((256, 256))
    hsv = np.asarray(resized.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0].astype(np.float32) * (360.0 / 255.0)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    plant_like = (
        (saturation >= 35)
        & (value >= 30)
        & (hue >= 8)
        & (hue <= 170)
    )
    plant_pixel_count = int(plant_like.sum())
    if plant_pixel_count < 100:
        return None

    yellowing = (
        plant_like
        & (hue >= 25)
        & (hue <= 65)
        & (saturation >= 45)
        & (value >= 80)
    )
    brown = (
        plant_like
        & (hue >= 5)
        & (hue <= 35)
        & (saturation >= 45)
        & (value <= 175)
    )
    dark = plant_like & (value <= 65)
    return {
        "plant_like_pixels": plant_pixel_count / plant_like.size * 100,
        "yellow_pixels": int(yellowing.sum()) / plant_pixel_count * 100,
        "brown_or_dark_pixels": int((brown | dark).sum()) / plant_pixel_count * 100,
    }


def render_diagnosis(image: Image.Image, filename: str, true_class: str | None) -> None:
    model_modified_ns = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else None
    model, class_names, device = load_model(model_modified_ns)
    st.image(image, caption=filename, width=320)
    if true_class:
        st.caption(f"Dataset label (from folder): **{true_class}**")
    if model is None:
        st.warning("No trained model found. Run `python train_model.py` after preparing the dataset.")
        return
    predictions = predict(image, model, class_names, device)
    predicted_class, confidence = predictions[0]
    info = DISEASE_INFO.get(predicted_class, {})
    st.subheader(info.get("name", predicted_class.replace("___", " — ")))
    st.metric("Model confidence", f"{confidence:.1%}")
    st.caption(
        "Prediction scores are softmax outputs and are not calibrated probabilities."
    )
    st.markdown("**Top model predictions for this image**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Candidate": name.replace("___", " — "), "Score": f"{score:.1%}"}
                for name, score in predictions
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if true_class:
        st.caption(
            f"Prediction {'matches' if predicted_class == true_class else 'differs from'} "
            "the dataset folder label."
        )
    st.markdown("**Image-specific color cues**")
    pixel_analysis = analyze_image_pixels(image)
    if pixel_analysis is None:
        st.info("Could not identify enough plant-like colored pixels for a color-cue estimate.")
    else:
        cue_columns = st.columns(3)
        cues = (
            ("Plant-like pixels", "plant_like_pixels"),
            ("Yellowing-like pixels", "yellow_pixels"),
            ("Brown/dark, lesion-like pixels", "brown_or_dark_pixels"),
        )
        for column, (title, key) in zip(cue_columns, cues):
            column.metric(title, f"{pixel_analysis[key]:.1f}%")
        st.caption(
            "These are rough HSV color-pixel fractions, not leaf segmentation or "
            "confirmed chlorosis, necrosis, or fungal-spot detection. Background and "
            "lighting can affect them."
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "The model is uncertain about this image. Do not use this prediction to "
            "choose a disease-specific treatment; seek local expert confirmation."
        )
        return

    st.markdown("**Disease-class guidance**")
    st.caption(
        "Guidance below is associated with the predicted class; it is not generated "
        "from lesion segmentation."
    )
    treatment_columns = st.columns(2)
    for column, title, key in zip(
        treatment_columns,
        ("Organic / cultural schedule", "Chemical treatment guidance"),
        ("organic", "chemical"),
    ):
        column.markdown(f"**{title}**")
        for step, treatment in enumerate(info.get(key, []), start=1):
            column.markdown(f"{step}. {treatment}")


def analytics_tab() -> None:
    st.header("Dataset & Model Analytics")
    split_names = ("train", "valid", "test")
    counts = {
        split: count_images(DATASET_DIR / split) for split in split_names
    }
    metric_columns = st.columns(3)
    for column, split in zip(metric_columns, split_names):
        column.metric(f"{split.title()} images", counts[split])
    class_counts = {}
    for split in split_names:
        split_path = DATASET_DIR / split
        if split_path.exists():
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    class_counts.setdefault(class_dir.name, {})[split] = count_images(class_dir)
    if class_counts:
        distribution = pd.DataFrame(class_counts).T.fillna(0).astype(int)
        for split in split_names:
            if split not in distribution.columns:
                distribution[split] = 0
        st.subheader("Class distribution")
        st.dataframe(distribution[list(split_names)], width="stretch")
        st.bar_chart(distribution[list(split_names)])
    else:
        st.info("No dataset images found. Run `python prepare_dataset.py`.")

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if "test_accuracy" in metrics:
            st.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
        if "test_per_class" in metrics:
            class_rows = {
                name.replace("___", " — "): {
                    metric: values[metric]
                    for metric in ("precision", "recall", "f1-score", "support")
                    if metric in values
                }
                for name, values in metrics["test_per_class"].items()
                if isinstance(values, dict) and "precision" in values
            }
            if class_rows:
                st.subheader("Held-out test results by class")
                st.dataframe(
                    pd.DataFrame(class_rows).T.round(2),
                    width="stretch",
                )
        epochs = range(1, len(metrics.get("loss", [])) + 1)
        if epochs:
            figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(epochs, metrics.get("loss", []), label="Train")
            axes[0].plot(epochs, metrics.get("val_loss", []), label="Validation")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].legend()
            axes[1].plot(epochs, metrics.get("accuracy", []), label="Train")
            axes[1].plot(epochs, metrics.get("val_accuracy", []), label="Validation")
            axes[1].set_title("Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()
            figure.tight_layout()
            st.subheader("Training curves")
            st.pyplot(figure)
            plt.close(figure)
    else:
        st.info("Training curves appear here after `python train_model.py`.")


def diagnosis_tab() -> None:
    st.header("Multimodal Leaf Disease Diagnosis")
    st.caption(
        "Inference runs locally. Dataset labels are folder labels; symptom indicators "
        "are associated with the predicted class, not measured lesion segmentation."
    )
    mode = st.radio(
        "Choose an image source",
        ("Pick from local test set", "Upload an image"),
        horizontal=True,
    )
    test_images = list_test_images()
    if mode == "Pick from local test set":
        if not test_images:
            st.info("No local test images found. Run `python prepare_dataset.py` first.")
            return
        selected = st.selectbox(
            "Test image",
            test_images,
            format_func=lambda path: (
                f"{path.relative_to(DATASET_DIR).as_posix()} "
                f"({path.parent.name})"
            ),
        )
        with Image.open(selected) as source:
            image = source.convert("RGB")
        render_diagnosis(image, selected.name, selected.parent.name)
    else:
        uploaded = st.file_uploader(
            "Upload a leaf image", type=("jpg", "jpeg", "png", "webp", "bmp")
        )
        if uploaded is not None:
            with Image.open(uploaded) as source:
                image = source.convert("RGB")
            render_diagnosis(image, uploaded.name, None)


def calculate_fertilizer(
    n_available: float, p_available: float, k_available: float, crop: str
) -> dict[str, float]:
    target = CROP_TARGETS[crop]
    p_deficit = max(0.0, target["P"] - p_available)
    dap = p_deficit / 0.46
    n_from_dap = dap * 0.18
    urea = max(0.0, target["N"] - n_available - n_from_dap) / 0.46
    mop = max(0.0, target["K"] - k_available) / 0.60
    return {"Urea": urea, "DAP": dap, "MOP": mop}


def soil_tab() -> None:
    st.header("Soil & Fertilizer Optimization")
    st.caption(
        "Planning estimates in kg/acre from the entered available nutrient values; "
        "confirm with a local soil test and regional recommendations before application."
    )
    with st.form("soil_form"):
        first_row = st.columns(3)
        n_value = first_row[0].number_input("Nitrogen (N), kg/acre", 0.0, 500.0, 20.0)
        p_value = first_row[1].number_input("Phosphorus (P₂O₅), kg/acre", 0.0, 300.0, 15.0)
        k_value = first_row[2].number_input("Potassium (K₂O), kg/acre", 0.0, 500.0, 20.0)
        second_row = st.columns(3)
        soil_ph = second_row[0].number_input("Soil pH", 3.0, 10.0, 6.5, step=0.1)
        soil_type = second_row[1].selectbox(
            "Soil type", ("Loam", "Sandy", "Clay", "Silt", "Sandy loam")
        )
        crop = second_row[2].selectbox("Target crop", tuple(CROP_TARGETS))
        submitted = st.form_submit_button("Calculate plan")

    if submitted:
        dosages = calculate_fertilizer(n_value, p_value, k_value, crop)
        st.subheader(f"Estimated fertilizer plan for {crop}")
        st.dataframe(
            pd.DataFrame(
                {"Fertilizer": list(dosages), "Estimated rate (kg/acre)": list(dosages.values())}
            ).round(1),
            hide_index=True,
            width="stretch",
        )
        amendments = []
        if soil_ph < 5.5:
            amendments.append(
                "Soil is acidic: consider an agricultural lime recommendation based on "
                "buffer-pH testing; do not apply a generic rate."
            )
        elif soil_ph > 7.5:
            amendments.append(
                "Soil is alkaline: confirm with a soil test; sulfur or other amendments "
                "require locally calibrated rates."
            )
        else:
            amendments.append("pH is in a broadly suitable range for many crops.")
        if soil_type == "Sandy":
            amendments.append(
                "Sandy soil: split fertilizer applications and maintain organic matter "
                "to reduce leaching."
            )
        elif soil_type == "Clay":
            amendments.append(
                "Clay soil: avoid working it when wet; use drainage and organic matter "
                "management appropriate to the field."
            )
        else:
            amendments.append(
                f"{soil_type} soil: adjust timing and irrigation to local drainage "
                "and nutrient-retention conditions."
            )
        st.markdown("**Soil amendments & management**")
        for item in amendments:
            st.markdown(f"- {item}")
        st.info(
            "The calculator treats entered P and K as available P₂O₅ and K₂O equivalents. "
            "Fertilizer products vary; use product labels and local agronomic advice."
        )


def main() -> None:
    st.set_page_config(
        page_title="Offline Digital Agronomist",
        page_icon="🌱",
        layout="wide",
    )
    st.title("🌱 Offline Digital Agronomist")
    st.caption("Local-first leaf disease screening, dataset analytics, and soil planning")
    analytics, diagnosis, soil = st.tabs([
        "Dataset & Model Analytics",
        "Multimodal Leaf Disease Diagnosis",
        "Soil & Fertilizer Optimization",
    ])
    with analytics:
        analytics_tab()
    with diagnosis:
        diagnosis_tab()
    with soil:
        soil_tab()
    st.divider()
    st.caption(
        "Educational prototype only. Image predictions can be wrong; verify diagnoses "
        "with a qualified local agronomist before taking crop-protection action."
    )


if __name__ == "__main__":
    main()
