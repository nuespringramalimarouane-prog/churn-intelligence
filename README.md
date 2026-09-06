# Telecom Churn Intelligence

project notebook on kaggle : https://www.kaggle.com/code/nuespring/churn-intelligence-comparing-4-ml-classification

Telecom Churn Intelligence is a customer-risk dashboard for predicting whether a telecom customer is likely to leave. It combines:

- A Next.js frontend in `frontend/`.
- A FastAPI prediction service in `python-api/api/`.
- Four pre-trained scikit-learn models in `python-api/api/models/`.
- Cleaning, preprocessing, and training scripts in `python-api/src/`.

The normal workflow is to start the FastAPI service first, then start the Next.js dashboard. The dashboard sends prediction requests through its own server-side route, so the browser does not call FastAPI directly.

## Requirements

Install the following before starting:

- Windows 10 or later
- Python 3.10 or newer
- Node.js 20 or newer and npm
- Git, if cloning the repository

The checked-in model files must remain in `python-api/api/models/`. They are loaded lazily when a model is selected for the first prediction.

## Project Layout

```text
.
├── frontend/
│   ├── app/
│   │   ├── api/predict/route.ts    # Next.js proxy to FastAPI
│   │   ├── api/compare/route.ts    # Currently empty in this checkout
│   │   └── page.tsx
│   ├── components/dashboard.tsx   # Dashboard and prediction form
│   ├── lib/api.ts                  # Frontend types and API calls
│   ├── package.json
│   └── package-lock.json
└── python-api/
    ├── api/
    │   ├── server.py               # FastAPI application
    │   ├── requirements.txt
    │   ├── data/clean_sample_api.csv
    │   └── models/*.pkl
    └── src/
        ├── clean.py                # Exploratory/manual cleaning
        ├── pipeline_cleaning.py    # ColumnTransformer preprocessing
        ├── main.py                 # Model training and export
        ├── data/                   # Source and generated CSV files
        └── requirements.txt
```

## Installation

Open PowerShell in the repository root.

### 1. Create the Python environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python-api\api\requirements.txt
```

If PowerShell blocks activation, allow local scripts for your user account and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install frontend dependencies

Open a second PowerShell window, return to the repository root, and run:

```powershell
cd frontend
npm ci
```

`npm ci` uses the checked-in `package-lock.json`. Use `npm install` only when intentionally changing dependencies.

## Run Locally

Use two terminals.

### Terminal 1: FastAPI

From the repository root, activate the environment and start the API from the directory containing `server.py`:

```powershell
.\.venv\Scripts\Activate.ps1
cd python-api\api
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

The API is now available at `http://127.0.0.1:8000`. Check it before opening the dashboard:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

### Terminal 2: Next.js

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000` in a browser. Use **Predict Customer** to enter a customer profile and run a model.

## Connecting To Another API Host

The Next.js proxy defaults to `http://127.0.0.1:8000`. To use a different FastAPI host or port, create `frontend/.env.local`:

```env
FASTAPI_URL=http://127.0.0.1:8000
```

Restart `npm run dev` after changing this value. `FASTAPI_URL` is read by the Next.js server route and is not exposed as a browser-side `NEXT_PUBLIC_` variable.

## API Reference

### `GET /health`

Returns a service status object:

```json
{ "status": "ok" }
```

### `POST /predict`

Accepts one customer record and a model filename. The `Model` value must match a file in `python-api/api/models/`:

- `logistic_regression.pkl`
- `gradient_boosting.pkl`
- `random_forest.pkl`
- `voting_classifier.pkl`

Example request:

```json
{
  "customerID": "4821",
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 8,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 604,
  "Model": "logistic_regression.pkl"
}
```

The response contains `prediction` (`CHURN` or `NO CHURN`), a numeric `probability` between 0 and 1, the selected model, and the submitted customer record. The API does not use `TotalCharges` as a model feature, but it still requires that field in the request schema.

The API converts Yes/No service fields to numeric values and creates the fixed one-hot columns expected by the checked-in models. Its feature order and dropped baseline categories must stay aligned with the models if you replace or retrain them.

## Data And Model Pipeline

The source data and generated CSV files are under `python-api/src/data/`:

- `dataset-telecom.csv`: source telecom dataset for the pipeline cleaner.
- `real_samples.csv`: samples used by the manual cleaner and training script.
- `data_cleaned.csv` and `real_samples_cleaned.csv`: cleaned data consumed by `main.py`.
- `pipeline_cleaned_data.csv`: output from `pipeline_cleaning.py`.

The scripts use paths relative to the current working directory. Run them from `python-api/src`, not from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd python-api\src
python pipeline_cleaning.py
python clean.py
python main.py
```

`main.py` trains Logistic Regression, Gradient Boosting, Random Forest, and Voting Classifier models, prints evaluation metrics, and writes `.pkl` files to a local `models/` directory. Because the API reads from `python-api/api/models/`, create or copy the exports there before serving them:

```powershell
New-Item -ItemType Directory -Force ..\api\models
Copy-Item models\*.pkl ..\api\models\
```

The training script expects `data_cleaned.csv` and `real_samples_cleaned.csv` to already exist. Run the appropriate cleaning scripts first or provide those files yourself. The training requirements are listed in `python-api/src/requirements.txt`; install them in the same virtual environment if you are retraining:

```powershell
python -m pip install -r requirements.txt
```

The training code exports scikit-learn objects directly. Keep the Python and scikit-learn environment compatible with the versions used to create the model files, or retrain all models after changing versions.

## Current Limitations

- The frontend calls `/api/compare` for **Compare Models**, but `frontend/app/api/compare/route.ts` is empty in this checkout. Single-model prediction uses `/api/predict` and is the supported working flow; comparison will not return model results until that route is implemented.
- The dashboard contains static KPI, distribution, and factor values. They are presentation values, not calculated from live prediction history.
- `server.py` allows requests from all origins for local development. Restrict `allow_origins` before deploying publicly.
- Pickle/joblib files should be loaded only from trusted sources. Do not place untrusted model files in the models directory.

## Production Build

Build and start the frontend with:

```powershell
cd frontend
npm run build
npm run start
```

Run FastAPI with a production process configuration instead of `--reload`, for example:

```powershell
cd python-api\api
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Set `FASTAPI_URL` in the deployment environment to the reachable API URL, configure CORS for the real dashboard origin, and keep the model files available at `python-api/api/models/` or update `MODELS_DIR` in `server.py`.

## Troubleshooting

**Prediction service cannot be reached**

Confirm Terminal 1 is running and that `http://127.0.0.1:8000/health` returns `{"status":"ok"}`. If FastAPI uses another port, update `frontend/.env.local` and restart Next.js.

**Unknown model**

Check that the selected filename exists in `python-api/api/models/` and exactly matches one of the names in `frontend/lib/api.ts`.

**Model input mismatch**

The model and API preprocessing must agree on feature names, order, encoding, and scaling. Retrain/export the models using compatible preprocessing rather than passing raw categorical values to a model trained on encoded data.

**Training script cannot find data**

Change directory to `python-api/src` before running the script. Its paths are relative, so running it from another directory looks for `data/` in the wrong place.
