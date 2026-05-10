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
    # CONFIG

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_Multimodal")

    # LOAD DATA

    df = pd.read_csv("src/data/df_clean.csv")

    # COLUMNS

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

    # FEATURES

    X = df[[TEXT_COL] + NUM_COLS]
    y = df[TARGET]

    # SPLIT 70 / 15 / 15

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(" Split done")

    # PREPROCESSOR

    preprocessor = ColumnTransformer(
        [
            ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2)), TEXT_COL),
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

    # MODEL

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )

    # TRAINING

    with mlflow.start_run(run_name="Multimodal_RF"):

        # ================= TRAIN =================
        pipeline.fit(X_train, y_train)

        # ================= VALIDATION =================
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"\n Validation Accuracy: {val_acc:.4f}")

        # ================= TEST =================
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ================= RESULTS =================
        print("\n========== RESULTS ==========")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-score  : {f1:.4f}")

        # ================= MLFLOW =================
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("fusion", "TFIDF + Numerical")
        mlflow.log_param("split", "70/15/15")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, artifact_path="multimodal_model")

        # ================= SAVE =================
        os.makedirs("models", exist_ok=True)

        joblib.dump(pipeline, "models/multimodal_model.pkl")

    print("\n Multimodal model saved successfully")


if __name__ == "__main__":
    main()
