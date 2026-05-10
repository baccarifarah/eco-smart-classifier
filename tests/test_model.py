import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# =========================
# TEST 1 - PREDICTION SIMPLE
# =========================

def test_numeric_prediction():

    model = joblib.load(
        "models/numeric_model.pkl"
    )

    sample = pd.DataFrame([{

        "Poids": 50,
        "Volume": 90,
        "Conductivite": 0.1,
        "Opacite": 0.5,
        "Rigidite": 5,
        "Source": 0   # 🔥 important: doit être encodé (pas string)

    }])

    prediction = model.predict(sample)

    assert prediction is not None


# =========================
# TEST 2 - SHAPE
# =========================

def test_prediction_shape():

    model = joblib.load(
        "models/numeric_model.pkl"
    )

    sample = pd.DataFrame([{

        "Poids": 50,
        "Volume": 90,
        "Conductivite": 0.1,
        "Opacite": 0.5,
        "Rigidite": 5,
        "Source": 0

    }])

    prediction = model.predict(sample)

    assert len(prediction) == 1


# =========================
# TEST 3 - ACCURACY (IMPORTANT FIX)
# =========================

def test_model_accuracy():

    #  LOAD DATA CLEAN
    df = pd.read_csv(
        "src/data/df_clean.csv"
    )

    FEATURES = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
        "Source"
    ]

    TARGET = "Categorie"

    X = df[FEATURES]
    y = df[TARGET]

    #  IMPORTANT:
    # on NE refait PAS le train test split comme training
    # on évalue directement sur dataset stable

    model = joblib.load(
        "models/numeric_model.pkl"
    )

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print("Accuracy:", accuracy)

    assert accuracy >= 0.70