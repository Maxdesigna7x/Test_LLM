#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def resolve_image_root(data_dir: Path) -> Path:
    candidates = [data_dir]
    candidates.extend(path for path in data_dir.iterdir() if path.is_dir())
    for base in candidates:
        for candidate in (base / "train", base / "Train", base / "training", base / "Training", base):
            if candidate.exists() and candidate.is_dir():
                class_dirs = [p for p in candidate.iterdir() if p.is_dir()]
                if len(class_dirs) >= 2:
                    return candidate
    return data_dir


def make_stratified_split(dataset: datasets.ImageFolder, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    per_class: dict[int, list[int]] = defaultdict(list)
    for idx, class_idx in enumerate(dataset.targets):
        per_class[int(class_idx)].append(idx)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_idx, indices in per_class.items():
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_split))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def save_sample_images(dataset: datasets.ImageFolder, indices: list[int], classes: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    for idx in indices:
        _, class_idx = dataset.samples[idx]
        class_idx = int(class_idx)
        if class_idx in seen:
            continue
        seen.add(class_idx)
        image_path, _ = dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image.save(output_dir / f"train_{classes[class_idx]}.png")
        if len(seen) == len(classes):
            break


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una CNN simple para clasificación de imágenes")
    parser.add_argument("--data-dir", default="dataset", help="Directorio base del dataset")
    parser.add_argument("--output-dir", default="outputs", help="Directorio de salida")
    parser.add_argument("--epochs", type=int, default=20, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Tamaño de imagen")
    parser.add_argument("--val-split", type=float, default=0.2, help="Proporción para validación")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_image_root(Path(args.data_dir))
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    sample_dataset = datasets.ImageFolder(str(data_root))
    train_dataset = datasets.ImageFolder(str(data_root), transform=train_transform)
    eval_dataset = datasets.ImageFolder(str(data_root), transform=eval_transform)
    train_indices, val_indices = make_stratified_split(sample_dataset, args.val_split, args.seed)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    save_sample_images(sample_dataset, train_indices, sample_dataset.classes, assets_dir)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = SimpleCNN(num_classes=len(sample_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_history(history, output_dir / "history.png")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": sample_dataset.classes,
            "image_size": args.img_size,
            "history": history,
        },
        output_dir / "model.pth",
    )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(args.epochs):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])

    summary = {
        "data_root": str(data_root),
        "classes": sample_dataset.classes,
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "history": history,
        "model_path": str((output_dir / "model.pth").resolve()),
        "history_image": str((output_dir / "history.png").resolve()),
        "sample_images_dir": str(assets_dir.resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
