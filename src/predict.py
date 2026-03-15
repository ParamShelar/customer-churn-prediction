import joblib
import pandas as pd

CHURN_THRESHOLD = 0.35


def load_model(model_path="models/churn_model.pkl"):
    return joblib.load(model_path)


def predict_churn(input_data, model_path="models/churn_model.pkl", threshold=CHURN_THRESHOLD):
    model = load_model(model_path)

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    churn_probability = model.predict_proba(input_df)[0, 1]
    churn_prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": float(churn_probability),
        "churn_prediction": churn_prediction,
        "threshold": threshold
    }
