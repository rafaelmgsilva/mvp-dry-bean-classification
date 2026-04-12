import json
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

METADATA_PATH = Path("models/metadata.json")


def validate_and_build_input(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Valida o payload, retorna DataFrame e avisos de extrapolação."""
    missing_fields = [column for column in FEATURE_COLUMNS if column not in payload]

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Campos obrigatórios ausentes: {missing}")

    row = {}
    for column in FEATURE_COLUMNS:
        try:
            row[column] = float(payload[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valor inválido para o campo '{column}'") from exc

    warnings = build_range_warnings(row)
    dataframe = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    return dataframe, warnings


def build_range_warnings(row: dict[str, float]) -> list[str]:
    """Gera avisos para valores fora da faixa observada no dataset."""
    if not METADATA_PATH.exists():
        return []

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_ranges = metadata.get("feature_ranges", {})
    warnings: list[str] = []

    for feature_name, value in row.items():
        if feature_name not in feature_ranges:
            continue

        min_value = feature_ranges[feature_name]["min"]
        max_value = feature_ranges[feature_name]["max"]

        if value < min_value or value > max_value:
            warnings.append(
                f"O valor de '{feature_name}' está fora da faixa observada "
                f"no treinamento [{min_value:.6f}, {max_value:.6f}]."
            )

    return warnings
