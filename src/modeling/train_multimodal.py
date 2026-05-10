import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_Multimodal")

    df = pd.read_csv("src/data/df_clean.csv")

    TARGET = "Categorie"
    TEXT_COL = "Rapport_Collecte"

    NUM_COLS = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
        "Prix_Revente",
        "Source",
    ]

    X = df[[TEXT_COL] + NUM_COLS]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    preprocessor = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
                TEXT_COL,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUM_COLS,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="Multimodal_RF"):

        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "multimodal_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/multimodal_model.pkl")


if __name__ == "__main__":
    main()
