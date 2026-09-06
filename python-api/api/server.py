from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import pickle
import joblib
import os

app = FastAPI(title="Churn Prediction API")

# Allows your Next.js server to call this API. In production, replace "*"
# with your actual dashboard origin (e.g. "https://your-dashboard.com").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Put your .pkl files in a "models" folder next to this file, e.g.:
#   python-api/models/logistic_regression.pkl
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_model_cache: dict[str, object] = {}  # filled lazily by load_model(), do not pre-seed with paths


def load_model(model_name: str):
    """Load a model by filename, caching it after first load.
    Tries joblib first (the common format for sklearn models), falls
    back to plain pickle for files saved that way instead.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

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


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model: str
    customer: CustomerData


# Exact columns/order the model was trained on. TotalCharges is NOT used.
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

CLEAN_SAMPLE_PATH = Path(__file__).parent / "data" / "clean_sample_api.csv"


def _yes_no(value: str) -> int:
    # Mirrors: .replace({"No phone service": "No", "No internet service": "No"})
    # followed by .replace({"Yes": 1, "No": 0}) in the batch cleaning script.
    return 1 if value == "Yes" else 0


def clean_customer_data(data: CustomerData) -> pd.DataFrame:
    """Reproduces the batch cleaning script's transformations for one
    incoming request, writes the result to clean_sample_api.csv, then
    reads it back for prediction (matching the requested workflow).

    The batch script fits a fresh OneHotEncoder on the whole dataset —
    that can't be replicated for a single row, so the same fixed one-hot
    columns (with the same baseline categories dropped: gender_Male,
    InternetService_Fiber optic, Contract_Month-to-month) are hardcoded
    here instead of fitting an encoder per request.
    """
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

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    CLEAN_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_SAMPLE_PATH, index=False)

    # Read back from disk, per the requested workflow.
    return pd.read_csv(CLEAN_SAMPLE_PATH)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def prediction(data: CustomerData):
    model = load_model(data.Model)
    df = clean_customer_data(data)

    try:
        proba = model.predict_proba(df)[0]
        # Assumes index 1 = "churn" class. Flip to [0] if your encoder
        # mapped churn to 0 instead of 1.
        churn_probability = float(proba[1])
    except AttributeError:
        pred = model.predict(df)[0]
        churn_probability = 1.0 if pred in (1, "Yes", "CHURN") else 0.0
    except ValueError as e:
        # Most common cause: the model expects numeric/encoded columns but
        # got raw strings ("Male", "Fiber optic", ...). See note below.
        raise HTTPException(
            status_code=500,
            detail=f"Model input mismatch — check preprocessing: {e}",
        )

    label = "CHURN" if churn_probability >= 0.5 else "NO CHURN"

    return PredictionResponse(
        prediction=label,
        probability=churn_probability,
        model=data.Model,
        customer=data,
    )