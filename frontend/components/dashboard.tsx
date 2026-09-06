"use client";
import { useState } from "react";
import {
  compareModels,
  CustomerData,
  Prediction,
  predictCustomer,
  labels,
  models,
} from "@/lib/api";

const initial: CustomerData = {
  customerID: "4821",
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  tenure: 8,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 75.5,
  TotalCharges: 604,
  Model: "logistic_regression.pkl",
};
const choices: Record<string, string[]> = {
  gender: ["Male", "Female"],
  SeniorCitizen: ["No", "Yes"],
  Partner: ["No", "Yes"],
  Dependents: ["No", "Yes"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["No", "Yes", "No phone service"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["Yes", "No", "No internet service"],
  OnlineBackup: ["Yes", "No", "No internet service"],
  DeviceProtection: ["Yes", "No", "No internet service"],
  TechSupport: ["Yes", "No", "No internet service"],
  StreamingTV: ["Yes", "No", "No internet service"],
  StreamingMovies: ["Yes", "No", "No internet service"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ],
};
const fieldLabels: Record<string, string> = {
  customerID: "Customer ID",
  gender: "Gender",
  SeniorCitizen: "Senior Citizen",
  Partner: "Partner",
  Dependents: "Dependents",
  tenure: "Tenure (months)",
  PhoneService: "Phone Service",
  MultipleLines: "Multiple Lines",
  InternetService: "Internet Service",
  OnlineSecurity: "Online Security",
  OnlineBackup: "Online Backup",
  DeviceProtection: "Device Protection",
  TechSupport: "Tech Support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming Movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless Billing",
  PaymentMethod: "Payment Method",
  MonthlyCharges: "Monthly Charges",
  TotalCharges: "Total Charges",
};
function Header() {
  return (
    <header>
      <div className="eyebrow">ML / CUSTOMER RISK</div>
      <h1>
        CHURN <span>INTELLIGENCE</span>
      </h1>
      <p>Customer churn prediction &amp; risk analytics</p>
    </header>
  );
}
function Kpis() {
  return (
    <section className="kpis">
      {[
        ["Total Customers", "7,043", "Customers analyzed"],
        ["Churn Rate", "26.5%", "1,869 customers churned"],
        ["High Risk", "842", "Probability &gt; 70%"],
      ].map(([a, b, c]) => (
        <div className="kpi" key={a}>
          <small>{a}</small>
          <strong>{b}</strong>
          <span>{c}</span>
        </div>
      ))}
    </section>
  );
}
function Distribution() {
  const bars = [28, 51, 72, 44, 22];
  return (
    <section className="panel distribution">
      <div className="section-head">
        <div>
          <h2>Churn Probability Distribution</h2>
          <p>Predicted customer churn probability across the dataset</p>
        </div>
        <span className="tag">7,043 SAMPLES</span>
      </div>
      <div className="chart">
        {bars.map((v, i) => (
          <div className="bar-wrap" key={i}>
            <div className="bar" style={{ height: `${v * 2.1}px` }}>
              <span>{[1892, 2640, 1578, 671, 262][i].toLocaleString("en-US")}</span>
            </div>
            <span>
              {i * 20}–{i * 20 + 20}%
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
function Factors() {
  return (
    <section className="panel factors">
      <div className="section-head">
        <div>
          <h2>Top Churn Factors</h2>
          <p>Features with the strongest influence</p>
        </div>
      </div>
      {[
        ["Month-to-month contract", 92],
        ["High monthly charges", 76],
        ["Low tenure", 63],
        ["Fiber optic internet", 51],
        ["No technical support", 38],
      ].map(([name, value], i) => (
        <div className="factor" key={name}>
          <div>
            <b>0{i + 1}</b>
            <span>{name}</span>
            <em>{value}%</em>
          </div>
          <div className="track">
            <i style={{ width: `${value}%` }} />
          </div>
        </div>
      ))}
    </section>
  );
}
function ModelSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="model-grid">
      {models.map((m) => (
        <button
          type="button"
          className={value === m ? "selected" : ""}
          onClick={() => onChange(m)}
          key={m}
        >
          <span>{labels[m]}</span>
          <small>{m.replace(".pkl", "")}</small>
        </button>
      ))}
    </div>
  );
}
function CustomerForm({
  onClose,
  onResult,
}: {
  onClose: () => void;
  onResult: (p: Prediction) => void;
}) {
  const [data, setData] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const update = (k: string, v: string) =>
    setData((d) => ({
      ...d,
      [k]: ["SeniorCitizen"].includes(k)
        ? v === "Yes"
          ? 1
          : 0
        : ["tenure"].includes(k)
          ? Number(v)
          : ["MonthlyCharges", "TotalCharges"].includes(k)
            ? Number(v)
            : v,
    }));
  const submit = async (e: any) => {
    e.preventDefault();
    if (
      !data.customerID ||
      data.tenure < 0 ||
      Number.isNaN(data.MonthlyCharges) ||
      Number.isNaN(data.TotalCharges)
    ) {
      setError("Please complete all required fields with valid values.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      onResult(await predictCustomer(data));
      onClose();
    } catch {
      setError("Prediction failed. Unable to process this customer sample.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={submit}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">NEW SAMPLE</div>
            <h2>Predict a customer</h2>
            <p>Enter profile signals and choose a model.</p>
          </div>
          <button type="button" className="close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="form-scroll">
          {[
            [
              "Customer Information",
              [
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
              ],
            ],
            [
              "Services",
              [
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
              ],
            ],
            [
              "Contract & Billing",
              [
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
              ],
            ],
          ].map(([group, keys]) => (
            <fieldset key={group as string}>
              <legend>{group as string}</legend>
              <div className="form-grid">
                {(keys as string[]).map((k) => (
                  <label key={k}>
                    {fieldLabels[k]}
                    {choices[k] ? (
                      <select
                        value={
                          String(data[k as keyof CustomerData]) === "1"
                            ? "Yes"
                            : String(data[k as keyof CustomerData])
                        }
                        onChange={(e) => update(k, e.target.value)}
                      >
                        {choices[k].map((x) => (
                          <option key={x}>{x}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={["tenure"].includes(k) ? "number" : "text"}
                        value={String(data[k as keyof CustomerData])}
                        onChange={(e) => update(k, e.target.value)}
                      />
                    )}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
          <fieldset>
            <legend>Choose Prediction Model</legend>
            <ModelSelector
              value={data.Model}
              onChange={(v) => update("Model", v)}
            />
          </fieldset>
          {error && <div className="error">{error}</div>}
        </div>
        <button className="primary" disabled={busy}>
          {busy ? (
            <>
              <span className="spinner" />
              Analyzing customer...
              <small>Running {labels[data.Model]}...</small>
            </>
          ) : (
            "Run prediction"
          )}
        </button>
      </form>
    </div>
  );
}
function PredictionCard({
  result,
  onOpen,
  onCompare,
}: {
  result: Prediction;
  onOpen: () => void;
  onCompare: () => void;
}) {
  const c = result.customer;
  const pct = Math.round(result.probability * 100);
  return (
    <section className="panel prediction">
      <div className="section-head">
        <div>
          <h2>Customer Prediction</h2>
          <p>Latest model inference</p>
        </div>
        <span className="live">● LIVE</span>
      </div>
      <div className="customer-id">
        Customer <b>#{c.customerID}</b>
        <span>{result.model}</span>
      </div>
      <div className="result-main">
        <div
          className="meter"
          style={{ "--pct": `${pct}%` } as React.CSSProperties}
        >
          <strong>{pct}%</strong>
          <span>probability</span>
        </div>
        <div>
          <div className={result.prediction === "CHURN" ? "risk high" : "risk"}>
            {result.prediction}
          </div>
          <small>
            {pct >= 70 ? "HIGH RISK" : pct >= 40 ? "MEDIUM RISK" : "LOW RISK"}
          </small>
        </div>
      </div>
      <div className="features">
        <span>
          <small>Contract</small>
          {c.Contract}
        </span>
        <span>
          <small>Tenure</small>
          {c.tenure} months
        </span>
        <span>
          <small>Monthly charges</small>${c.MonthlyCharges.toFixed(2)}
        </span>
        <span>
          <small>Internet</small>
          {c.InternetService}
        </span>
      </div>
      <div className="actions">
        <button className="primary" onClick={onOpen}>
          Predict Customer
        </button>
        <button className="secondary" onClick={onCompare}>
          Compare Models
        </button>
      </div>
    </section>
  );
}
function Comparison({
  results,
  onClose,
}: {
  results: Prediction[];
  onClose: () => void;
}) {
  return (
    <div className="comparison">
      <div className="section-head">
        <div>
          <h2>Model Comparison</h2>
          <p>Same customer, four classification models</p>
        </div>
        <button className="close" onClick={onClose}>
          ×
        </button>
      </div>
      <div className="table">
        {results.map((r) => (
          <div className="row" key={r.model}>
            <b>{r.model}</b>
            <strong className={r.prediction === "CHURN" ? "danger" : ""}>
              {r.prediction}
            </strong>
            <span>{Math.round(r.probability * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
export default function Dashboard() {
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<Prediction>({
    prediction: "CHURN",
    probability: 0.78,
    model: "Logistic Regression",
    customer: initial,
  });
  const [comparison, setComparison] = useState<Prediction[] | null>(null);
  return (
    <main className="shell">
      <Header />
      <Kpis />
      <Distribution />
      <div className="bottom">
        <Factors />
        <PredictionCard
          result={result}
          onOpen={() => setOpen(true)}
          onCompare={async () => {
            try {
              setComparison(await compareModels(result.customer));
            } catch {
              setComparison(null);
            }
          }}
        />
      </div>
      {comparison && (
        <Comparison results={comparison} onClose={() => setComparison(null)} />
      )}{" "}
      {open && (
        <CustomerForm onClose={() => setOpen(false)} onResult={setResult} />
      )}
      <footer>
        CHURN INTELLIGENCE <span>MODEL MONITORING / 2026</span>
      </footer>
    </main>
  );
}
