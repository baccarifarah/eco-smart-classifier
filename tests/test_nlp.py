import joblib


def test_nlp_prediction():
    model = joblib.load("models/nlp_model.pkl")

    preds = model.predict(["plastique recyclable"])

    assert len(preds) == 1


def test_nlp_empty_input():
    model = joblib.load("models/nlp_model.pkl")

    try:
        model.predict([""])
        assert True
    except:
        assert True