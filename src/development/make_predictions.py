import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/clean/predictive_data.csv")
model = joblib.load("src/development/model.sav")

# PART I: preprocess data
# drop columns
submission = df[["ID", "Pred"]]
df = df.drop(["year", "ID", "Pred", "Season", "T_x", "T_y"], axis=1)

# scale data
scl = MinMaxScaler()
df = scl.fit_transform(df)

# run predictions
submission["Pred"] = model.predict_proba(df)[:, 1]

# save predictions
submission.to_csv("data/clean/prediction_submissions.csv", index=False)
