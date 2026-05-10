from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_home():
    response = client.get("/")
    assert response.status_code == 200


def test_numeric_endpoint():
    payload = {
        "Poids": 50,
        "Volume": 90,
        "Conductivite": 0.1,
        "Opacite": 0.5,
        "Rigidite": 5,
        "Source": 1,
    }

    response = client.post("/predict/numeric/", json=payload)
    assert response.status_code == 200


def test_numeric_invalid_input():
    response = client.post("/predict/numeric/", json={})

    # API FIXÉE → doit retourner 422
    assert response.status_code == 422
