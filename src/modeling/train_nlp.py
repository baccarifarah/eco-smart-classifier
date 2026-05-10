import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def main():
    # CONFIG

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_NLP")


    # LOAD DATA 

    df = pd.read_csv("src/data/df_clean.csv")


    # COLUMNS

    TARGET = "Categorie"
    TEXT_COL = "Rapport_Collecte"

    # FEATURES

    X = df[TEXT_COL]
    y = df[TARGET]


    # SPLIT 70 / 15 / 15 (CORRECT CAHIER DES CHARGES)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(" Split done")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")


    # PIPELINE NLP

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC())
    ])


    # TRAINING + VALIDATION + TEST

    with mlflow.start_run(run_name="NLP_LinearSVC"):

        # ================= TRAIN =================
        pipeline.fit(X_train, y_train)

        # ================= VALIDATION =================
        y_val_pred = pipeline.predict(X_val)

        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"\n Validation Accuracy : {val_acc:.4f}")

        # ================= TEST =================
        y_test_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_test_pred)

        prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        # ================= RESULTS =================
        print("\n========== RESULTS ==========")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-score  : {f1:.4f}")

        # ================= MLFLOW =================
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("vectorizer", "TFIDF")
        mlflow.log_param("split", "70/15/15")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="nlp_model"
        )

        # ================= SAVE MODEL =================
        os.makedirs("models", exist_ok=True)

        joblib.dump(
            pipeline,
            "models/nlp_model.pkl"
        )

    print("\n NLP model saved successfully")

if __name__ == "__main__":
    main()