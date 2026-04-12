from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from src.schema import FEATURE_COLUMNS, validate_and_build_input

MODEL_PATH = Path("models/model.joblib")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[dict, int]:
        return {"status": "ok"}, 200

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
        )

    @app.post("/predict")
    def predict() -> tuple[dict, int]:
        if not MODEL_PATH.exists():
            return {"error": "Modelo não encontrado. Treine o modelo primeiro."}, 500

        payload = request.get_json(silent=True)
        if payload is None:
            return {"error": "Payload JSON inválido ou ausente."}, 400

        try:
            input_data, warnings = validate_and_build_input(payload)
            model = joblib.load(MODEL_PATH)

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(max(probabilities))
            else:
                confidence = None

        except ValueError as exc:
            return {"error": str(exc)}, 400
        except Exception:
            return {"error": "Erro interno ao realizar predição."}, 500

        response = {
            "predicted_class": str(prediction),
            "warnings": warnings,
        }

        if confidence is not None:
            response["predicted_probability"] = confidence

        return jsonify(response), 200

    return app


app = create_app()
