from src.predictor import create_holdout, load_dataset, split_features_target


def test_load_dataset_returns_dataframe() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")

    assert dataframe is not None
    assert not dataframe.empty
    assert "Class" in dataframe.columns


def test_dataset_has_expected_shape() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")

    assert dataframe.shape == (13611, 17)


def test_split_features_target_returns_expected_parts() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")

    x_data, y_data = split_features_target(dataframe)

    assert x_data.shape == (13611, 16)
    assert y_data.shape == (13611,)
    assert "Class" not in x_data.columns


def test_target_has_seven_classes() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")

    _, y_data = split_features_target(dataframe)

    assert y_data.nunique() == 7


def test_create_holdout_returns_expected_shapes() -> None:
    dataframe = load_dataset("data/Dry_Bean_Dataset.arff")
    x_data, y_data = split_features_target(dataframe)

    x_train, x_test, y_train, y_test = create_holdout(x_data, y_data)

    assert x_train.shape[1] == 16
    assert x_test.shape[1] == 16
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == 13611
