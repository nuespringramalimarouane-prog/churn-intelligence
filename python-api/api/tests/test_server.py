import pandas as pd

from server import ALL_MODEL_FILES, CustomerData, clean_customer_data, _score_model


def sample_customer():
    return CustomerData(
        customerID="test-1",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No internet service",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="Yes",
        Contract="One year",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=55.0,
        TotalCharges=660.0,
        Model=ALL_MODEL_FILES[0],
    )


def test_cleaning_returns_ordered_features_without_disk_side_effects():
    features = clean_customer_data(sample_customer())

    assert list(features.columns) == [
        "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
        "MonthlyCharges", "gender_Female", "InternetService_DSL",
        "InternetService_No", "Contract_One year", "Contract_Two year",
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    ]
    assert features.loc[0, "MultipleLines"] == 0
    assert features.loc[0, "OnlineSecurity"] == 0
    assert features.loc[0, "gender_Female"] == 1
    assert features.loc[0, "InternetService_DSL"] == 1
    assert isinstance(features, pd.DataFrame)


def test_model_error_is_preserved_for_comparison(monkeypatch):
    def fail_to_load(_):
        raise RuntimeError("model missing")

    monkeypatch.setattr("server.load_model", fail_to_load)
    result = _score_model(ALL_MODEL_FILES[0], clean_customer_data(sample_customer()))

    assert result.prediction == "ERROR"
    assert result.error == "model missing"
