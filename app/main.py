from fastapi import FastAPI
from app.schemas import Passenger
from app.predict import make_prediction
from app.explain import explain_prediction

app = FastAPI(title="Titanic ML API")

@app.get("/")
def home():
    return {"message": "Titanic API Running"}

@app.post("/predict")
def predict(data: Passenger):
    pred, prob = make_prediction(data)
    return {
        "prediction": pred,
        "survival_probability": prob
    }

@app.post("/explain")
def explain(data: Passenger):
    explanation = explain_prediction(data)
    return {
        "shap_values": explanation
    }