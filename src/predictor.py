from pathlib import Path

import arff
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Carrega o dataset Dry Bean em formato ARFF."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        dataset = arff.load(file)

    dataframe = pd.DataFrame(dataset["data"])
    dataframe.columns = [attribute[0] for attribute in dataset["attributes"]]

    for column in dataframe.columns:
        if dataframe[column].dtype == object:
            dataframe[column] = dataframe[column].apply(_decode_if_bytes)

    return dataframe


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos preditores e variável alvo."""
    x_data = dataframe.drop(columns=["Class"]).copy()
    y_data = dataframe["Class"].copy()

    return x_data, y_data


def create_holdout(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Cria separação holdout estratificada."""
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    return x_train, x_test, y_train, y_test


def _decode_if_bytes(value: object) -> object:
    """Decodifica bytes para string quando necessário."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
