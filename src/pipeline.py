import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Keep every generated image and model input at the same size.
IMAGE_SIZE = 64
CATEGORIES = ("gear", "washer")
FALSE_REJECT_BUDGET = 0.05
ARTIFACT_DIR = Path("artifacts")


def render_part(category: str, defect: bool, seed: int, lighting_shift: float = 0.0) -> Image.Image:
    # Build one repeatable grayscale part from coordinate masks.
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    cx = IMAGE_SIZE // 2 + int(rng.integers(-2, 3))
    cy = IMAGE_SIZE // 2 + int(rng.integers(-2, 3))
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 35.0 + lighting_shift * 255.0)

    if category == "gear":
        angle = np.arctan2(yy - cy, xx - cx)
        toothed_edge = 20.0 + 2.5 * np.cos(12.0 * angle)
        part_mask = (radius <= toothed_edge) & (radius >= 7.0)
    else:
        part_mask = (radius <= 21.0) & (radius >= 10.0)

    canvas[part_mask] = 190.0
    if defect:
        # Add a bright diagonal scratch across the part surface.
        scratch = np.abs((yy - cy) - 0.55 * (xx - cx)) < 1.2
        canvas[scratch & part_mask] = 250.0

    canvas += rng.normal(0.0, 5.0, canvas.shape)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="L")


def assert_disjoint(records: list[dict[str, str]]) -> None:
    # Reject any stable image ID assigned to more than one split.
    memberships: dict[str, set[str]] = defaultdict(set)
    for record in records:
        memberships[record["image_id"]].add(record["split"])
    leaked = sorted(image_id for image_id, splits in memberships.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Split leakage detected: {leaked[:5]}")


def generate_dataset(inject_leak: bool = False, lighting_shift: float = 0.0) -> list[dict[str, str]]:
    # Rebuild the dataset so every run starts from a known state.
    root = Path("data")
    if root.exists():
        shutil.rmtree(root)
    records: list[dict[str, str]] = []
    samples_per_label = {"train": 60, "val": 20, "test": 20}
    seed = 1000

    for split, count in samples_per_label.items():
        for category in CATEGORIES:
            for label in (0, 1):
                kind = "defect" if label else "normal"
                target = root / split / category / kind
                target.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    image_id = f"{split}-{category}-{label}-{index:03d}"
                    path = target / f"{image_id}.png"
                    render_part(category, bool(label), seed, lighting_shift).save(path)
                    records.append({
                        "image_id": image_id,
                        "split": split,
                        "category": category,
                        "label": str(label),
                        "path": path.as_posix(),
                    })
                    seed += 1

    if inject_leak:
        # Copy one identity into a second split to challenge the guard.
        leaked_record = dict(records[0])
        leaked_record["split"] = "test"
        records.append(leaked_record)

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "category", "label", "path"])
        writer.writeheader()
        writer.writerows(records)

    assert_disjoint(records)
    counts = {split: sum(record["split"] == split for record in records) for split in samples_per_label}
    print(f"split counts: {counts}")
    return records


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Load one manifest split into model-ready arrays.
    images: list[np.ndarray] = []
    labels: list[int] = []
    categories: list[str] = []
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["split"] != split:
                continue
            with Image.open(record["path"]) as image:
                array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            images.append(array[None, :, :])
            labels.append(int(record["label"]))
            categories.append(record["category"])
    return np.stack(images), np.asarray(labels), np.asarray(categories)


class TinyCNN(nn.Module):
    # Produce a defect logit while exposing frozen spatial features.
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features).flatten(1))

    def patch_embeddings(self, images: torch.Tensor) -> np.ndarray:
        # Turn each spatial feature cell into one patch embedding.
        with torch.no_grad():
            features = self.features(images).permute(0, 2, 3, 1)
        return features.reshape(features.shape[0], -1, features.shape[-1]).numpy()


def train_cnn(images: np.ndarray, labels: np.ndarray, epochs: int = 18) -> TinyCNN:
    # Train the compact classifier on labeled synthetic parts.
    torch.manual_seed(7)
    model = TinyCNN()
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels.astype(np.float32))[:, None])
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(7))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def cnn_scores(model: TinyCNN, images: np.ndarray) -> np.ndarray:
    # Convert classifier logits into defect probabilities.
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(images))).numpy().reshape(-1)


def build_memory_banks(model: TinyCNN, images: np.ndarray, labels: np.ndarray, categories: np.ndarray) -> dict[str, np.ndarray]:
    # Store a compact sample of nominal patch embeddings per category.
    banks: dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        mask = (categories == category) & (labels == 0)
        patches = model.patch_embeddings(torch.from_numpy(images[mask])).reshape(-1, 16)
        banks[category] = patches[::4]
    return banks


def patch_scores(model: TinyCNN, banks: dict[str, np.ndarray], images: np.ndarray, categories: np.ndarray) -> np.ndarray:
    # Score each image by its most unusual patch distance.
    indexes = {category: NearestNeighbors(n_neighbors=1).fit(bank) for category, bank in banks.items()}
    embeddings = model.patch_embeddings(torch.from_numpy(images))
    scores: list[float] = []
    for patches, category in zip(embeddings, categories, strict=True):
        distances, _ = indexes[str(category)].kneighbors(patches)
        scores.append(float(distances.max()))
    return np.asarray(scores)


def calibrate(scores: np.ndarray, labels: np.ndarray) -> float:
    # Set the operating point from normal validation scores only.
    return float(np.quantile(scores[labels == 0], 1.0 - FALSE_REJECT_BUDGET, method="higher"))


def evaluate(scores: np.ndarray, labels: np.ndarray, categories: np.ndarray, threshold: float) -> dict[str, object]:
    # Report ranking quality and fixed-threshold behavior.
    auroc = {
        category: float(roc_auc_score(labels[categories == category], scores[categories == category]))
        for category in CATEGORIES
    }
    normal = labels == 0
    defect = labels == 1
    return {
        "threshold": threshold,
        "auroc": auroc,
        "false_reject_rate": float(np.mean(scores[normal] > threshold)),
        "defect_recall": float(np.mean(scores[defect] > threshold)),
    }


def train_and_report() -> dict[str, object]:
    # Train both inspectors and persist their reproducible evidence.
    train_images, train_labels, train_categories = load_split(Path("data"), "train")
    val_images, val_labels, val_categories = load_split(Path("data"), "val")
    test_images, test_labels, test_categories = load_split(Path("data"), "test")
    model = train_cnn(train_images, train_labels)

    cnn_val = cnn_scores(model, val_images)
    cnn_test = cnn_scores(model, test_images)
    banks = build_memory_banks(model, train_images, train_labels, train_categories)
    patch_val = patch_scores(model, banks, val_images, val_categories)
    patch_test = patch_scores(model, banks, test_images, test_categories)

    cnn_threshold = calibrate(cnn_val, val_labels)
    patch_threshold = calibrate(patch_val, val_labels)
    report = {
        "false_reject_budget": FALSE_REJECT_BUDGET,
        "cnn": evaluate(cnn_test, test_labels, test_categories, cnn_threshold),
        "patchcore_style": evaluate(patch_test, test_labels, test_categories, patch_threshold),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")
    np.savez(ARTIFACT_DIR / "memory_banks.npz", **banks)
    report_path = ARTIFACT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Record parameters, metrics, and key files in a local MLflow database.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("visual-inspection")
    with mlflow.start_run(run_name="cnn-vs-patchcore-style"):
        mlflow.log_params({"epochs": 18, "false_reject_budget": FALSE_REJECT_BUDGET})
        metrics: dict[str, float] = {}
        for name in ("cnn", "patchcore_style"):
            result = report[name]
            metrics[f"{name}_false_reject_rate"] = float(result["false_reject_rate"])
            metrics[f"{name}_defect_recall"] = float(result["defect_recall"])
            for category, value in result["auroc"].items():
                metrics[f"{name}_{category}_auroc"] = float(value)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(ARTIFACT_DIR / "model.pt"))
        mlflow.log_artifact(str(report_path))

    print(json.dumps(report, indent=2))
    return report