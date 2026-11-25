"""Download, validate, and split a small PlantVillage image dataset."""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_dataset"
DATASET_DIR = ROOT / "dataset"
CLASSES = {
    "Tomato___Healthy": "Tomato___healthy",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Potato___Late_blight": "Potato___Late_blight",
}
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
API_URL = (
    "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/"
    "contents/raw/color/{class_name}"
)


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError):
        return False


def cached_images(class_name: str) -> list[Path]:
    folder = RAW_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if valid_image(path):
                images.append(path)
            else:
                path.unlink()
    return images


def fetch_class_images(class_name: str, desired_count: int) -> list[Path]:
    """Reuse the local cache, fetching only the images needed to fill it."""
    images = cached_images(class_name)
    if len(images) >= desired_count:
        return images[:desired_count]

    headers = {"User-Agent": "Offline-Digital-Agronomist/1.0"}
    page = 1
    while len(images) < desired_count:
        response = requests.get(
            API_URL.format(class_name=class_name),
            params={"ref": "master", "per_page": 100, "page": page},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        entries = response.json()
        if not isinstance(entries, list):
            raise RuntimeError(f"Unexpected GitHub response for {class_name}.")
        image_entries = [
            item
            for item in entries
            if item.get("type") == "file"
            and Path(item.get("name", "")).suffix.lower() in IMAGE_EXTENSIONS
            and item.get("download_url")
        ]
        if not image_entries:
            break
        remaining = desired_count - len(images)
        # Spread selections across the listing instead of taking adjacent frames.
        if len(image_entries) > remaining:
            indices = [round(i * (len(image_entries) - 1) / (remaining - 1))
                       for i in range(remaining)] if remaining > 1 else [0]
            image_entries = [image_entries[index] for index in indices]

        for item in image_entries:
            if len(images) >= desired_count:
                break
            destination = RAW_DIR / class_name / f"{Path(item['name']).stem}.jpg"
            if destination.exists() and valid_image(destination):
                if destination not in images:
                    images.append(destination)
                continue
            image_response = requests.get(
                item["download_url"], headers=headers, timeout=30
            )
            image_response.raise_for_status()
            with Image.open(io.BytesIO(image_response.content)) as source:
                source.convert("RGB").save(destination, format="JPEG", quality=92)
            images.append(destination)

        if len(entries) < 100:
            break
        page += 1
    return images[:desired_count]


def create_visual_fallback(class_name: str, count: int = 20) -> list[Path]:
    """Create labeled leaf illustrations only when no source images can be fetched."""
    folder = RAW_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    healthy = "healthy" in class_name.lower()
    for index in range(count):
        image = Image.new("RGB", (256, 256), (220, 232, 210))
        draw = ImageDraw.Draw(image)
        variation = index % 5
        leaf_color = (64 + variation * 5, 135, 57) if healthy else (98, 124, 54)
        draw.ellipse((48, 30, 210, 230), fill=leaf_color, outline=(37, 89, 39), width=4)
        draw.line((65, 212, 196, 50), fill=(215, 205, 143), width=5)
        for spot in range(5 + variation):
            x = 76 + ((spot * 31 + index * 13) % 112)
            y = 69 + ((spot * 47 + index * 7) % 130)
            color = (125, 70, 49) if not healthy else (90, 155, 70)
            draw.ellipse((x, y, x + 12, y + 9), fill=color)
        draw.text((8, 8), "LOCAL VISUAL FALLBACK", fill=(30, 30, 30))
        path = folder / f"fallback_{index:03d}.jpg"
        image.save(path, format="JPEG", quality=90)
    return cached_images(class_name)


def clear_generated_images() -> None:
    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        path.unlink()
            else:
                folder.mkdir(parents=True, exist_ok=True)


def save_split(images: list[Path], class_name: str) -> dict[str, int]:
    labels = [class_name] * len(images)
    train_images, other_images = train_test_split(
        images, test_size=0.30, random_state=42, stratify=labels
    )
    valid_images, test_images = train_test_split(
        other_images, test_size=0.50, random_state=42,
        stratify=[class_name] * len(other_images),
    )
    counts = {}
    for split, split_images in zip(
        SPLITS, (train_images, valid_images, test_images)
    ):
        output_dir = DATASET_DIR / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(split_images):
            output_path = output_dir / f"{class_name}_{index:04d}.jpg"
            with Image.open(source_path) as source:
                source.convert("RGB").save(output_path, format="JPEG", quality=92)
            if not valid_image(output_path):
                raise RuntimeError(f"Saved image failed validation: {output_path}")
        counts[split] = len(split_images)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-per-class", type=int, default=200,
        help="Maximum number of cached/downloaded source images per class (default: 200).",
    )
    args = parser.parse_args()
    if args.images_per_class < 4:
        parser.error("--images-per-class must be at least 4 for stratified splits.")

    class_images: dict[str, list[Path]] = {}
    download_errors = []
    for class_name, source_class in CLASSES.items():
        try:
            images = fetch_class_images(source_class, args.images_per_class)
            class_images[class_name] = images
            print(f"{class_name}: {len(images)} valid cached/downloaded image(s)")
        except (requests.RequestException, OSError, ValueError, RuntimeError) as error:
            print(f"Download issue for {class_name}: {error}")
            download_errors.append(class_name)
            class_images[class_name] = cached_images(source_class)

    if not any(class_images.values()):
        print(
            "No PlantVillage images could be downloaded or read. Creating clearly "
            "labeled local visual fallback images; these are not real PlantVillage photos."
        )
        class_images = {
            class_name: create_visual_fallback(source_class)
            for class_name, source_class in CLASSES.items()
        }
    elif any(len(class_images[name]) < 4 for name in CLASSES):
        missing = [name for name in CLASSES if len(class_images[name]) < 4]
        raise RuntimeError(
            "Could not obtain at least four source images for every class "
            f"({', '.join(missing)}). No synthetic fallback was mixed with real images. "
            "Check the network and rerun the script."
        )

    clear_generated_images()
    totals = {split: 0 for split in SPLITS}
    print("\nSaved and verified image counts:")
    print(f"{'Class':32} {'train':>7} {'valid':>7} {'test':>7}")
    for class_name in CLASSES:
        counts = save_split(class_images[class_name], class_name)
        for split in SPLITS:
            totals[split] += counts[split]
        print(
            f"{class_name:32} {counts['train']:7d} "
            f"{counts['valid']:7d} {counts['test']:7d}"
        )
    print(
        f"{'TOTAL':32} {totals['train']:7d} "
        f"{totals['valid']:7d} {totals['test']:7d}"
    )
    print(f"\nDataset folders: {DATASET_DIR}")
    if download_errors:
        print("Some source requests failed; cached images were used where available.")
    return 0


if __name__ == "__main__":
    random.seed(42)
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        raise
