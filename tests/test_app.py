from src.app import create_app
from src.schema import FEATURE_COLUMNS
from src.training import train_and_save_best_model


def build_valid_payload() -> dict[str, float]:
    base_values = {
        "Area": 28395.0,
        "Perimeter": 610.291,
        "MajorAxisLength": 208.178117,
        "MinorAxisLength": 173.888747,
        "AspectRation": 1.197191,
        "Eccentricity": 0.549812,
        "ConvexArea": 28715.0,
        "EquivDiameter": 190.141097,
        "Extent": 0.763923,
        "Solidity": 0.988856,
        "roundness": 0.958027,
        "Compactness": 0.913358,
        "ShapeFactor1": 0.007332,
        "ShapeFactor2": 0.003147,
        "ShapeFactor3": 0.834222,
        "ShapeFactor4": 0.998724,
    }

    return {column: base_values[column] for column in FEATURE_COLUMNS}


def ensure_model_artifacts() -> None:
    train_and_save_best_model(
        dataset_path="data/Dry_Bean_Dataset.arff",
        model_output_path="models/model.joblib",
        metrics_output_path="models/metrics.json",
        metadata_output_path="models/metadata.json",
    )


def test_health_endpoint_returns_ok() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_route_returns_html() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Dry Bean Classifier" in html
    assert "Predizer classe" in html


def test_predict_endpoint_returns_class_for_valid_payload() -> None:
    ensure_model_artifacts()

    app = create_app()
    client = app.test_client()

    response = client.post("/predict", json=build_valid_payload())
    response_data = response.get_json()

    assert response.status_code == 200
    assert "predicted_class" in response_data
    assert "warnings" in response_data


def test_predict_endpoint_returns_probability_for_valid_payload() -> None:
    ensure_model_artifacts()

    app = create_app()
    client = app.test_client()

    response = client.post("/predict", json=build_valid_payload())
    response_data = response.get_json()

    assert response.status_code == 200
    assert "predicted_probability" in response_data
    assert 0.0 <= response_data["predicted_probability"] <= 1.0


def test_predict_endpoint_returns_warning_for_out_of_range_value() -> None:
    ensure_model_artifacts()

    app = create_app()
    client = app.test_client()

    payload = build_valid_payload()
    payload["Area"] = 999999999.0

    response = client.post("/predict", json=payload)
    response_data = response.get_json()

    assert response.status_code == 200
    assert len(response_data["warnings"]) > 0


def test_predict_endpoint_returns_400_for_missing_field() -> None:
    app = create_app()
    client = app.test_client()

    invalid_payload = build_valid_payload()
    invalid_payload.pop(FEATURE_COLUMNS[0])

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert "error" in response.get_json()
