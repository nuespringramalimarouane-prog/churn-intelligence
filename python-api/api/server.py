from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import joblib
import os

import pandas as pd

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_model_cache: dict[str, object] = {}


def load_model(model_name: str):
    if model_name not in ALL_MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

    _model_cache[model_name] = model
    return model


class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    Model: str


class ModelResult(BaseModel):
    model: str
    prediction: str
    probability: float
    error: str | None = None


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model: str
    customer: CustomerData
    all_models: list[ModelResult]


ALL_MODEL_FILES = [
    "logistic_regression.pkl",
    "gradient_boosting.pkl",
    "random_forest.pkl",
    "voting_classifier.pkl",
]

FEATURE_COLUMNS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "gender_Female", "InternetService_DSL",
    "InternetService_No", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]


def _yes_no(value: str) -> int:
    return 1 if value == "Yes" else 0


def clean_customer_data(data: CustomerData) -> pd.DataFrame:
    """Convert one API request into the exact ordered model feature matrix."""
    row = {
        "SeniorCitizen": data.SeniorCitizen,
        "Partner": _yes_no(data.Partner),
        "Dependents": _yes_no(data.Dependents),
        "tenure": data.tenure,
        "PhoneService": _yes_no(data.PhoneService),
        "MultipleLines": _yes_no("No" if data.MultipleLines == "No phone service" else data.MultipleLines),
        "OnlineSecurity": _yes_no("No" if data.OnlineSecurity == "No internet service" else data.OnlineSecurity),
        "OnlineBackup": _yes_no("No" if data.OnlineBackup == "No internet service" else data.OnlineBackup),
        "DeviceProtection": _yes_no("No" if data.DeviceProtection == "No internet service" else data.DeviceProtection),
        "TechSupport": _yes_no("No" if data.TechSupport == "No internet service" else data.TechSupport),
        "StreamingTV": _yes_no("No" if data.StreamingTV == "No internet service" else data.StreamingTV),
        "StreamingMovies": _yes_no("No" if data.StreamingMovies == "No internet service" else data.StreamingMovies),
        "PaperlessBilling": _yes_no(data.PaperlessBilling),
        "MonthlyCharges": data.MonthlyCharges,
        "gender_Female": 1 if data.gender == "Female" else 0,
        "InternetService_DSL": 1 if data.InternetService == "DSL" else 0,
        "InternetService_No": 1 if data.InternetService == "No" else 0,
        "Contract_One year": 1 if data.Contract == "One year" else 0,
        "Contract_Two year": 1 if data.Contract == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if data.PaymentMethod == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if data.PaymentMethod == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if data.PaymentMethod == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if data.PaymentMethod == "Mailed check" else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _score_model(model_name: str, df: pd.DataFrame) -> ModelResult:
    try:
        model = load_model(model_name)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[0]
            classes = list(getattr(model, "classes_", range(len(probabilities))))
            positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
            churn_probability = float(probabilities[positive_index])
        else:
            prediction = model.predict(df)[0]
            churn_probability = 1.0 if prediction in (1, "Yes", "CHURN") else 0.0

        label = "CHURN" if churn_probability >= 0.5 else "NO CHURN"
        return ModelResult(
            model=model_name,
            prediction=label,
            probability=churn_probability,
        )
    except Exception as error:
        return ModelResult(
            model=model_name,
            prediction="ERROR",
            probability=0.0,
            error=str(error),
        )


@app.post("/predict", response_model=PredictionResponse)
async def prediction(data: CustomerData):
    if data.Model not in ALL_MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {data.Model}")

    features = clean_customer_data(data)
    all_models = [_score_model(name, features) for name in ALL_MODEL_FILES]
    primary = next(result for result in all_models if result.model == data.Model)

    if primary.error:
        raise HTTPException(
            status_code=500,
            detail=f"Selected model failed: {primary.error}",
        )

    return PredictionResponse(
        prediction=primary.prediction,
        probability=primary.probability,
        model=primary.model,
        customer=data,
        all_models=all_models,
    )
