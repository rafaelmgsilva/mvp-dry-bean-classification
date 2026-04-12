import json
from pathlib import Path

from src.training import train_and_save_best_model


def test_best_model_meets_minimum_performance_requirements() -> None:
    model_path = "models/model.joblib"
    metrics_path = "models/metrics.json"
    metadata_path = "models/metadata.json"

    metrics = train_and_save_best_model(
        dataset_path="data/Dry_Bean_Dataset.arff",
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        metadata_output_path=metadata_path,
    )

    assert Path(model_path).exists()
    assert Path(metrics_path).exists()
    assert Path(metadata_path).exists()

    assert metrics["accuracy"] >= 0.90
    assert metrics["macro_f1"] >= 0.92


def test_metrics_file_contains_expected_fields() -> None:
    model_path = "models/model.joblib"
    metrics_path = "models/metrics.json"
    metadata_path = "models/metadata.json"

    train_and_save_best_model(
        dataset_path="data/Dry_Bean_Dataset.arff",
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        metadata_output_path=metadata_path,
    )

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert "best_model_name" in metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
