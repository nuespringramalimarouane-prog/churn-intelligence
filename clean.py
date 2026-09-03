from pathlib import Path
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logging.getLogger(__name__)


dataset = Path("./data/dataset-telecom.csv")
df = None
try: 
    if dataset.exists():
        df = pd.read_csv(dataset)
    else:
        raise FileNotFoundError("file not found")
except FileNotFoundError as e:
    logging.error(f"{e} : {dataset}")

""" --drop useless columns ["customerID","TotalCharges"] """
df = df.drop(columns=["customerID","TotalCharges"])

""" --encode gender classes ['Female','Male'] """
""" --encode InternetService ['DSL', 'Fiber optic', 'No'] """
"""  --encode strings of contract  ['Month-to-month', 'One year', 'Two year']"""

ohe = OneHotEncoder(sparse_output=False)

#gender encoding
encoded_gender = ohe.fit_transform(df[["gender"]])
encoded_cols_gender = ohe.get_feature_names_out(["gender"])
df_gender = pd.DataFrame(encoded_gender,columns=encoded_cols_gender,index=df.index,dtype=int)
#InternetService encoding
encoded_intserv = ohe.fit_transform(df[["InternetService"]]) 
encoded_cols_intserv = ohe.get_feature_names_out(["InternetService"])
df_intserv = pd.DataFrame(encoded_intserv,columns=encoded_cols_intserv,index=df.index,dtype=int)

#Contract encoding
encoded_Contract = ohe.fit_transform(df[["Contract"]]) 
encoded_cols_Contract = ohe.get_feature_names_out(["Contract"])
df_Contract = pd.DataFrame(encoded_Contract,columns=encoded_cols_Contract,index=df.index,dtype=int)

#Contract encoding
encoded_PaymentMethod = ohe.fit_transform(df[["PaymentMethod"]]) 
encoded_cols_PaymentMethod = ohe.get_feature_names_out(["PaymentMethod"])
df_PaymentMethod = pd.DataFrame(encoded_PaymentMethod,columns=encoded_cols_PaymentMethod,index=df.index,dtype=int)


""" --update gender InternetService"""
df = df.drop(["gender","InternetService","Contract","PaymentMethod"],axis=1).join([df_gender,df_intserv,df_Contract,df_PaymentMethod])

"""  --convert string culomns to numeric values ["Yes","No","No phone service"]  """

df["MultipleLines"] = pd.DataFrame(df["MultipleLines"]).replace({"No phone service":"No"})
df[["OnlineSecurity","StreamingMovies","OnlineBackup","DeviceProtection","TechSupport","StreamingTV"]] = pd.DataFrame(df[["OnlineSecurity","StreamingMovies","OnlineBackup","DeviceProtection","TechSupport","StreamingTV"]]).replace({"No internet service":"No"})

df[["Partner","StreamingMovies","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","PaperlessBilling","Churn"]] = pd.DataFrame(df[["Partner","StreamingMovies","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","PaperlessBilling","Churn"]].replace({"Yes":1,"No":0}),dtype=int)
df.info()
df.dropna()
df.shape
df["Churn"]
save_data_path = Path("./data") / 'data_cleaned.csv'
df.to_csv(save_data_path,index=False)