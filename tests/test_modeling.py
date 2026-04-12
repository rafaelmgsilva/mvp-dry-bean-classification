from src.modeling import (build_decision_tree_model, build_dummy_baseline,
                          build_gaussian_nb_model, build_knn_pipeline,
                          build_svm_pipeline, compare_models, evaluate_model,
                          get_candidate_models, train_model)
from src.predictor import create_holdout, load_dataset, split_features_target


def test_dummy_baseline_returns_metrics_between_zero_and_one() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    model = build_dummy_baseline()
    trained_model = train_model(model, x_train, y_train)
    metrics = evaluate_model(trained_model, x_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_knn_pipeline_returns_metrics_between_zero_and_one() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    model = build_knn_pipeline()
    trained_model = train_model(model, x_train, y_train)
    metrics = evaluate_model(trained_model, x_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_knn_beats_dummy_baseline_in_accuracy() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    dummy_model = train_model(build_dummy_baseline(), x_train, y_train)
    knn_model = train_model(build_knn_pipeline(), x_train, y_train)

    dummy_metrics = evaluate_model(dummy_model, x_test, y_test)
    knn_metrics = evaluate_model(knn_model, x_test, y_test)

    assert knn_metrics["accuracy"] > dummy_metrics["accuracy"]


def test_decision_tree_returns_metrics_between_zero_and_one() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    model = build_decision_tree_model()
    trained_model = train_model(model, x_train, y_train)
    metrics = evaluate_model(trained_model, x_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_gaussian_nb_returns_metrics_between_zero_and_one() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    model = build_gaussian_nb_model()
    trained_model = train_model(model, x_train, y_train)
    metrics = evaluate_model(trained_model, x_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_svm_returns_metrics_between_zero_and_one() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    model = build_svm_pipeline()
    trained_model = train_model(model, x_train, y_train)
    metrics = evaluate_model(trained_model, x_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_compare_models_returns_expected_columns_and_rows() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    models = get_candidate_models()
    ranking = compare_models(models, x_train, y_train, x_test, y_test)

    assert list(ranking.columns) == ["model_name", "accuracy", "macro_f1"]
    assert set(ranking["model_name"]) == {
        "dummy",
        "knn",
        "decision_tree",
        "gaussian_nb",
        "svm",
    }
    assert len(ranking) == 5


def test_best_model_beats_dummy_baseline_in_macro_f1() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    models = get_candidate_models()
    ranking = compare_models(models, x_train, y_train, x_test, y_test)

    best_row = ranking.iloc[0]
    dummy_row = ranking[ranking["model_name"] == "dummy"].iloc[0]

    assert best_row["macro_f1"] > dummy_row["macro_f1"]
