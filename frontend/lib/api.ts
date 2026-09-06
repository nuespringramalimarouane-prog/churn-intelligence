export type CustomerData = {
  customerID: string;
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges: number;
  Model: string;
};

export type ModelResult = {
  model: string;
  prediction: "CHURN" | "NO CHURN" | "ERROR";
  probability: number;
  error?: string;
};

export type Prediction = {
  prediction: "CHURN" | "NO CHURN" | "ERROR" ;
  probability: number;
  model: string;
  customer: CustomerData;
  all_models: ModelResult[];
};

// Update these to match your actual .pkl filenames in python-api/models/
export const models = [
  "logistic_regression.pkl",
  "gradient_boosting.pkl",
  "random_forest.pkl",
  "voting_classifier.pkl",
];

export const labels: Record<string, string> = {
  "logistic_regression.pkl": "Logistic Regression",
  "gradient_boosting.pkl": "Gradient Boosting",
  "random_forest.pkl": "Random Forest",
  "voting_classifier.pkl": "Voting Classifier",
};

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const message = await res.text().catch(() => "");
    throw new Error(`Request to ${url} failed (${res.status}): ${message}`);
  }

  return res.json();
}

export async function predictCustomer(data: CustomerData): Promise<Prediction> {
  return postJSON<Prediction>("/api/predict", data);
}

// No network call: /api/predict already returned every model's result,
// this just reshapes that data for the comparison view.
export function compareModels(prediction: Prediction): Prediction[] {
  return prediction.all_models.map((m) => ({
    prediction: m.prediction === "ERROR" ? "NO CHURN" : m.prediction,
    probability: m.probability,
    model: m.model,
    customer: prediction.customer,
    all_models: prediction.all_models,
  }));
}