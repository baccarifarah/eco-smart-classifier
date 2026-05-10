from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_numeric_missing_field():
    response = client.post("/predict/numeric/", json={"Poids": 10})
    assert response.status_code == 422


def test_nlp_endpoint():
    response = client.post(
        "/predict/nlp/", json={"Rapport_Collecte": "plastique recyclable"}
    )
    assert response.status_code == 200
