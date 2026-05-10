import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    # =========================
    # CONFIG
    # =========================
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_Numeric")

    RANDOM_STATE = 42

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("src/data/df_clean.csv")

    TARGET = "Categorie"

    NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

    X = df[NUM_COLS]
    y = df[TARGET]

    print(df["Categorie"].unique())

    print(sorted(df["Categorie"].unique()))
    # =========================
    # SPLIT 70/15/15
    # =========================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    # =========================
    # PIPELINE PROPRE (IMPORTANT FIX)
    # =========================
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # =========================
    # TRAIN + MLflow
    # =========================
    with mlflow.start_run(run_name="Numeric_RF_Corrected"):

        pipeline.fit(X_train, y_train)

        # validation
        val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        # test
        test_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average="weighted")

        print("Validation Accuracy:", val_acc)
        print("Test Accuracy:", acc)
        print("F1-score:", f1)

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("f1", f1)

        # =========================
        # SAVE MODEL
        # =========================
        os.makedirs("models", exist_ok=True)

        joblib.dump(pipeline, "models/numeric_model.pkl")
        mlflow.sklearn.log_model(pipeline, "model")

    print(" Numeric model trained correctly")


if __name__ == "__main__":
    main()
