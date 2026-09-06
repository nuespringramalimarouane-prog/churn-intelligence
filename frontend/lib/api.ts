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

export type Prediction = {
  prediction: "CHURN" | "NO CHURN";
  probability: number;
  model: string;
  customer: CustomerData;
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

export async function compareModels(data: CustomerData): Promise<Prediction[]> {
  return postJSON<Prediction[]>("/api/compare", data);
}