import pandas as pd


def test_dataset_not_empty():

    df = pd.read_csv("data/processed/df_clean.csv")

    assert len(df) > 0


def test_no_missing_values():

    df = pd.read_csv("data/processed/df_clean.csv")

    assert df.isnull().sum().sum() == 0


def test_required_columns():

    df = pd.read_csv("data/processed/df_clean.csv")

    required_columns = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
        "Source",
    ]

    for col in required_columns:

        assert col in df.columns
