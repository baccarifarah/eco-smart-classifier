import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_NLP")

    df = pd.read_csv("src/data/df_clean.csv")

    TARGET = "Categorie"
    TEXT_COL = "Rapport_Collecte"

    X = df[TEXT_COL]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("classifier", LinearSVC()),
        ]
    )

    with mlflow.start_run(run_name="NLP_LinearSVC"):

        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        y_test_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average="weighted")
        rec = recall_score(y_test, y_test_pred, average="weighted")
        f1 = f1_score(y_test, y_test_pred, average="weighted")

        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "nlp_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/nlp_model.pkl")


if __name__ == "__main__":
    main()
