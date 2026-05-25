import shap
import pandas as pd
def explain_prediction(data):
    from app.model import model   # lazy load

    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)

    df = pd.DataFrame([data.dict()])
    X_transformed = preprocessor.transform(df)

    shap_values = explainer(X_transformed)

    values = shap_values.values[0,:,1]
    features = preprocessor.get_feature_names_out()

    explanation = {}

    for i, val in enumerate(values):
        explanation[features[i]] = float(val)

    return {"shap_values": explanation}