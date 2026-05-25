import pandas as pd
from app.model import model

def make_prediction(data):
    df = pd.DataFrame([data.dict()])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return int(pred), float(prob)