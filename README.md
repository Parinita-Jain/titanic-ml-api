# Titanic ML API

An end-to-end Machine Learning API project built using FastAPI, MLflow, Scikit-learn, and SHAP Explainability.

This project predicts passenger survival probability on the Titanic dataset and provides model explainability using SHAP values.

---

# Features

- FastAPI-based REST API
- MLflow experiment tracking
- Automated best model selection
- Scikit-learn ML pipeline
- SHAP explainability
- CI/CD using GitHub Actions
- Modular project structure
- Production-ready inference pipeline

---

# Tech Stack

- Python
- FastAPI
- Scikit-learn
- MLflow
- SHAP
- Pandas
- GitHub Actions

---

# Project Structure

```bash
titanic-ml-api/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── explain.py
│   ├── schemas.py
│   └── logger.py
│
├── data/
│   └── titanic.csv
│
├── model/
│   ├── train.py
│   └── model.pkl
│
├── tests/
│
├── .github/workflows/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ML Pipeline Architecture

The project follows an end-to-end machine learning workflow:

1. Data preprocessing
2. Feature engineering
3. Model training
4. MLflow experiment tracking
5. Best model registration
6. Model export
7. FastAPI inference serving
8. SHAP explainability generation

---

# MLflow Architecture

MLflow is used for:

- Experiment tracking
- Hyperparameter logging
- Metric logging
- Model versioning
- Best model selection

Multiple RandomForest models are trained using different hyperparameters.

The best model is selected based on accuracy and exported for production inference.

---

# Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to explain predictions.

The API returns feature-wise contribution values showing:

- which features increased survival probability
- which features decreased survival probability

This improves transparency and interpretability of predictions.

---

# API Endpoints

## Home Endpoint

```http
GET /
```

Response:

```json
{
    "message": "Titanic API Running"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Sample Input:

```json
{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 100,
    "Embarked": "S"
}
```

Sample Response:

```json
{
    "prediction": 1,
    "survival_probability": 0.92
}
```

---

## Explainability Endpoint

```http
POST /explain
```

Sample Response:

```json
{
    "shap_values": {
        "num__Age": -0.12,
        "num__Fare": 0.21,
        "cat__Sex_female": 0.55
    }
}
```

---

#  Running Locally

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python model/train.py
```

---

## Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

---

# Open Swagger Documentation

```bash
http://127.0.0.1:8000/docs
```

---

# CI/CD

GitHub Actions is used for Continuous Integration.

The workflow automatically:

- installs dependencies
- validates imports
- checks API startup

on every push to GitHub.

---

# Future Improvements

- Docker deployment
- Cloud deployment
- Model monitoring
- Drift detection
- Authentication
- Kubernetes deployment

---

# Author

Parinita Jain

Data Science | Machine Learning | FastAPI | MLflow | NLP