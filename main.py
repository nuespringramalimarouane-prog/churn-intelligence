from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import pandas as pd
logging.getLogger(__name__)


dataset = Path("./data/data_cleaned.csv")
df = None
try: 
    if dataset.exists():
        df = pd.read_csv(dataset)
    else:
        raise FileNotFoundError("file not found")
except FileNotFoundError as e:
    logging.error(f"{e} : {dataset}")

y_data = pd.DataFrame(df[["Churn"]])
X_data = df.drop(columns=["Churn"])



X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.3,random_state=25)

model =  LogisticRegression()

model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train,y_pred_train)
test_acc = accuracy_score(y_test,y_pred_test)

print(f"train accu:{train_acc}")
print(f"test accu:{test_acc}")


