import json
from pathlib import Path

import joblib

from src.modeling import compare_models, get_candidate_models, train_model
from src.predictor import create_holdout, load_dataset, split_features_target


def train_and_save_best_model(
    dataset_path: str,
    model_output_path: str,
    metrics_output_path: str,
    metadata_output_path: str = "models/metadata.json",
) -> dict[str, float | str]:
    """Treina os modelos candidatos, salva o melhor e registra métricas."""
    dataframe = load_dataset(dataset_path)
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    candidate_models = get_candidate_models()
    ranking = compare_models(candidate_models, x_train, y_train, x_test, y_test)

    best_model_name = str(ranking.iloc[0]["model_name"])
    best_accuracy = float(ranking.iloc[0]["accuracy"])
    best_macro_f1 = float(ranking.iloc[0]["macro_f1"])

    best_model = candidate_models[best_model_name]
    trained_best_model = train_model(best_model, x_train, y_train)

    model_path = Path(model_output_path)
    metrics_path = Path(metrics_output_path)
    metadata_path = Path(metadata_output_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(trained_best_model, model_path)

    metrics = {
        "best_model_name": best_model_name,
        "accuracy": best_accuracy,
        "macro_f1": best_macro_f1,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    metadata = {
        "feature_columns": list(x_data.columns),
        "feature_ranges": {
            column: {
                "min": float(x_data[column].min()),
                "max": float(x_data[column].max()),
            }
            for column in x_data.columns
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return metrics
