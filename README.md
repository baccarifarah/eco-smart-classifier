# eco-smart-classifier

Eco-Smart Classifier est une plateforme intelligente de classification des déchets basée sur le Machine Learning, le NLP et le MLOps.

Le projet combine :
- Pipeline ML numérique
- Pipeline NLP
- Pipeline multimodal
- Tracking MLflow
- Versionnement DVC
- API REST FastAPI
- CI/CD GitHub Actions
- Monitoring Evidently AI

---

# Rejouer le Pipeline en 3 Commandes

## 1. Cloner le projet

```bash
git clone https://github.com/baccarifarah/eco-smart-classifier.git
cd eco-smart-classifier
```

## 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 3. Rejouer le pipeline complet

```bash
dvc repro
```

---

# Pipeline DVC Fonctionnel

Le pipeline DVC permet :

- le versionnement des données
- la reproductibilité des expériences
- l’automatisation des étapes ML

## Fichier versionné

```bash
data/processed/df_clean.csv.dvc
```

## Commandes DVC

```bash
dvc status
dvc repro
dvc dag
```

## Pipeline défini dans

```bash
dvc.yaml
```

---

# Expériences MLflow Documentées

Toutes les expériences sont suivies avec MLflow.

| Expérience | Description |
|---|---|
| Numeric_RF | Classification numérique avec Random Forest |
| NLP_TFIDF | Classification NLP avec TF-IDF |
| NLP_CamemBERT | Classification NLP avec CamemBERT |
| Multimodal_RF | Fusion multimodale avec Random Forest |
| Multimodal_Stacking | Fusion multimodale avec Stacking |
| FastText_Classifier | Classification NLP avec FastText |

## Lancer MLflow

```bash
mlflow ui
```

## Interface MLflow

```bash
http://localhost:5000
```

---

# Entraînement des Modèles

## Modèle numérique

```bash
python src/modeling/train_numeric.py
```

## Modèle NLP

```bash
python src/modeling/train_nlp.py
```

## Modèle multimodal

```bash
python src/modeling/train_multimodal.py
```

---

# API REST FastAPI

## Lancer l’API

```bash
uvicorn src.api.main:app --reload
```

## Documentation Swagger

```bash
http://127.0.0.1:8000/docs
```

---

# Dockerfile Opérationnel

## Build de l’image Docker

```bash
docker build -t eco-smart-classifier .
```

## Exécution du conteneur

```bash
docker run -p 8000:8000 eco-smart-classifier
```

---

# Tests Automatisés

Les tests sont réalisés avec Pytest.

## Exécution des tests

```bash
pytest --cov=src --cov-report=term-missing
```

## Couverture des Tests

```bash
Coverage >= 70%
```

## Tests réalisés

- validation dataset
- pipeline NLP
- vectorisation
- prédictions
- endpoints API
- monitoring
- text drift

---

# CI/CD GitHub Actions

Le pipeline CI/CD exécute automatiquement :

- installation des dépendances
- entraînement des modèles
- linting Black
- linting Flake8
- linting isort
- tests pytest
- coverage
- build Docker

## Workflow GitHub Actions

```bash
.github/workflows/ci.yml
```

---

# Monitoring

Le monitoring est réalisé avec Evidently AI.

## Fonctionnalités

- Drift numérique
- Drift textuel
- Jensen-Shannon divergence

## Rapports générés

```bash
reports/report.html
reports/evidently/drift_metrics.json
```

---

# Technologies Utilisées

- Python
- Scikit-learn
- FastAPI
- MLflow
- DVC
- Docker
- GitHub Actions
- Evidently AI
- Pandas
- NumPy
- CamemBERT
- Joblib

---

# Auteur

Farah Baccari

Projet réalisé dans le cadre du module MLOps et Machine Learning.
