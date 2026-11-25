"""Fine-tune an ImageNet-pretrained MobileNetV2 on local leaf images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "agronomist_leaf_model.pth"
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    # Keep frozen backbone BatchNorm statistics fixed while training the head.
    model.features.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: list[int] = []
    targets: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (batch_predictions == labels).sum().item()
        total += labels.size(0)
        predictions.extend(batch_predictions.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    if total == 0:
        raise RuntimeError("The dataset split is empty.")
    return total_loss / total, correct / total, predictions, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        parser.error("epochs and batch-size must be positive; image-size must be >= 64.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    valid_data = datasets.ImageFolder(
        DATASET_DIR / "valid", transform=evaluation_transform
    )
    test_data = datasets.ImageFolder(
        DATASET_DIR / "test", transform=evaluation_transform
    )
    if not train_data.classes or not valid_data.classes or not test_data.classes:
        raise RuntimeError("A dataset split is empty. Run prepare_dataset.py first.")
    if train_data.classes != valid_data.classes or train_data.classes != test_data.classes:
        raise RuntimeError("Train, validation, and test folders have different class labels.")

    loader_options = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )

    try:
        model = build_model(len(train_data.classes)).to(device)
    except Exception as error:
        raise RuntimeError(
            "Could not load the ImageNet MobileNetV2 weights. Training needs an "
            "internet connection once to download them; inference remains local."
        ) from error

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy, _, _ = run_epoch(
            model, valid_loader, criterion, device
        )
        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={train_loss:.4f} accuracy={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy or (
            val_accuracy == best_val_accuracy and val_loss < best_val_loss
        ):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    test_loss, test_accuracy, test_predictions, test_targets = run_epoch(
        model, test_loader, criterion, device
    )
    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(test_data.classes))),
        target_names=test_data.classes,
        output_dict=True,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(train_data.classes, indent=2), encoding="utf-8"
    )
    history["test_loss"] = test_loss
    history["test_accuracy"] = test_accuracy
    history["test_per_class"] = report
    history["model"] = "MobileNetV2 ImageNet transfer learning (frozen feature extractor)"
    (MODEL_DIR / "training_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Held-out test accuracy: {test_accuracy:.3f}")
    print(f"Saved weights: {MODEL_PATH}")
    print(f"Saved classes and training metrics in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
