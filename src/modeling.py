import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_dummy_baseline() -> DummyClassifier:
    """Cria um baseline ingênuo para comparação inicial."""
    return DummyClassifier(strategy="most_frequent")


def build_knn_pipeline(n_neighbors: int = 5) -> Pipeline:
    """Cria pipeline com padronização e KNN."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def build_decision_tree_model(
    max_depth: int | None = None,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Cria um modelo de árvore de decisão."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )


def build_gaussian_nb_model() -> GaussianNB:
    """Cria um modelo Gaussian Naive Bayes."""
    return GaussianNB()


def build_svm_pipeline(
    c_value: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42,
) -> Pipeline:
    """Cria pipeline com padronização e SVM."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    C=c_value,
                    kernel=kernel,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_candidate_models() -> dict[str, ClassifierMixin]:
    """Retorna os modelos candidatos para comparação."""
    return {
        "dummy": build_dummy_baseline(),
        "knn": build_knn_pipeline(),
        "decision_tree": build_decision_tree_model(),
        "gaussian_nb": build_gaussian_nb_model(),
        "svm": build_svm_pipeline(),
    }


def train_model(
    model: ClassifierMixin,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> ClassifierMixin:
    """Treina um modelo de classificação."""
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: ClassifierMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Avalia um modelo com métricas principais."""
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
    }


def compare_models(
    models: dict[str, ClassifierMixin],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Treina e avalia múltiplos modelos, retornando ranking consolidado."""
    results: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        trained_model = train_model(model, x_train, y_train)
        metrics = evaluate_model(trained_model, x_test, y_test)

        results.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    ranking = pd.DataFrame(results)
    ranking = ranking.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return ranking
