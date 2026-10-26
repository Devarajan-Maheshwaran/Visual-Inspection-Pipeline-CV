import csv
import json
from collections import defaultdict
from pathlib import Path

from src.export import MAX_ABS_ERROR, MAX_P95_MS
from src.pipeline import CATEGORIES


def test_manifest_splits_are_disjoint() -> None:
    # Confirm every stable image identity belongs to one split.
    memberships: dict[str, set[str]] = defaultdict(set)
    with Path("data/manifest.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            memberships[record["image_id"]].add(record["split"])
    assert all(len(splits) == 1 for splits in memberships.values())


def test_report_contains_per_category_auroc() -> None:
    # Require both inspectors to report every supported part category.
    report = json.loads(Path("artifacts/report.json").read_text(encoding="utf-8"))
    for inspector in ("cnn", "patchcore_style"):
        assert set(report[inspector]["auroc"]) == set(CATEGORIES)
        assert 0.0 <= report[inspector]["false_reject_rate"] <= 1.0
        assert 0.0 <= report[inspector]["defect_recall"] <= 1.0


def test_onnx_parity_and_latency() -> None:
    # Block models that miss either serving-quality limit.
    result = json.loads(Path("artifacts/validation.json").read_text(encoding="utf-8"))
    assert result["max_abs_error"] <= MAX_ABS_ERROR
    assert result["p95_ms"] <= MAX_P95_MS