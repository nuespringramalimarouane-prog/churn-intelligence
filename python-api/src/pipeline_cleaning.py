
# ----------------------------
# Import preprocessing modules
# ---------------------------- 

from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
from pathlib import Path
import logging

logging.getLogger(__name__)


# ------------------------
# load dataset with pandas
# ------------------------

data_path = Path('./data/dataset-telecom.csv')
df = None
try : 
    if data_path.exists():
        logging.info("data loading ...")
        df = pd.read_csv(data_path)
    else :
        raise FileNotFoundError('file not found')
        
except FileNotFoundError as e:
    logging.warning(f"{e} : {data_path}")

logging.info('data loading completed')

# Only run cleaning if df was loaded
if df is not None:
    df = df.drop(columns=["customerID", "TotalCharges"])
# ----------------------
# Custom pipeline steps
# ----------------------

def clean_no_service(col):
    return pd.DataFrame(col).replace({"No internet service": 0, "No phone service": 0,"No":0,"Yes":1}).to_numpy()


Cleaner = FunctionTransformer(clean_no_service)

logging.info("custom pipeline steps created")

# ----------------------
# Identify Column Types
# ----------------------

numeric_columns = ["MonthlyCharges","tenure"]

categorical_columns = ["gender","InternetService","Contract","PaymentMethod"]

convertToNumeric_columns = ["Partner","StreamingMovies","Dependents","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","PaperlessBilling","Churn","PhoneService"]

logging.info("identifying column types completed")

# ----------------------------------------------
# Handle Missing Values 
# Scale Numeric Features
# Encode Categorical Features 
# ----------------------------------------------

numeric_pipeline =  Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
])

categorical_pipeline =  Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore",drop='first')),
])

convertToNumeric_pipeline =  Pipeline(steps=[
    ("imputer",SimpleImputer(strategy='most_frequent')),
    ("cleaner",Cleaner),
])

logging.info("pipelines initialized")

# ------------------------------
# Combine with ColumnTransformer
# ------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("num",numeric_pipeline,numeric_columns),
    ("cat",categorical_pipeline,categorical_columns),
    ("conv",convertToNumeric_pipeline,convertToNumeric_columns),
    ("senior", "passthrough", ["SeniorCitizen"])  # keep as is
])

logging.info("column transformer initialized")

# Fit the preprocessor
data_cleaned_array = preprocessor.fit_transform(df)

print(preprocessor.named_transformers_["num"].fit_transform(df[numeric_columns]).shape)
print(preprocessor.named_transformers_["cat"].fit_transform(df[categorical_columns]).shape)
print(preprocessor.named_transformers_["conv"].fit_transform(df[convertToNumeric_columns]).shape)


# Get names for numeric + categorical parts
feature_names = []
feature_names.extend(preprocessor.named_transformers_["num"].get_feature_names_out(numeric_columns))
feature_names.extend(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns))

# For conv pipeline, just reuse the original column names
feature_names.extend(convertToNumeric_columns)

# SeniorCitizen passthrough
feature_names.append("SeniorCitizen")

# Wrap into DataFrame
data_cleaned = pd.DataFrame(data_cleaned_array, columns=feature_names)

# Wrap the array into a DataFrame with proper column names
data_cleaned = pd.DataFrame(data_cleaned_array, columns=feature_names)

print(data_cleaned.info())
print(data_cleaned.head())

save_cleaned_data = Path("./data/pipeline_cleaned_data.csv") 

data_cleaned.to_csv(save_cleaned_data,index=False)

logging.info("cleaned dataset saved as csv file")