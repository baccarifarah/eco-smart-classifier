import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EcoSmart_NLP")

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    df = pd.read_csv("src/data/df_clean.csv")

    TEXT_COL = "Rapport_Collecte"
    TARGET = "Categorie"

    X = df[TEXT_COL].astype(str)
    y = df[TARGET]

    # ==========================================================
    # SPLIT DATA
    # ==========================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # ==========================================================
    # HYBRID TF-IDF (ROBUSTE AUX FAUTES)
    # ==========================================================

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=3000
    )

    vectorizer = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf)
    ])

    # ==========================================================
    # MODEL
    # ==========================================================

    model = LinearSVC()

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])

    # ==========================================================
    # TRAIN + EVALUATION
    # ==========================================================

    with mlflow.start_run(run_name="NLP_LinearSVC"):

        pipeline.fit(X_train, y_train)

        # validation
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        # test
        y_test_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average="weighted")
        rec = recall_score(y_test, y_test_pred, average="weighted")
        f1 = f1_score(y_test, y_test_pred, average="weighted")

        # MLflow logs
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "nlp_model")

        # ======================================================
        # SAVE MODEL

        os.makedirs("models", exist_ok=True)

        joblib.dump(pipeline, "models/nlp_model.pkl")

        print("\n Training terminé")
        print(f" Validation Accuracy : {val_acc:.4f}")
        print(f" Test Accuracy       : {acc:.4f}")
        print(f" F1-score            : {f1:.4f}")


if __name__ == "__main__":
    main()
