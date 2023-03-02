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

# create rectangular prediction matrix
# TODO: Test predictions with Kaggle late submission(?)
rect = submission["ID"]\
    .str.split("_", expand=True)\
    .rename(columns={0: "year", 1: "T1", 2: "T2"})\
    .join(submission) \
    .drop(["year", "ID"], axis=1)

rect2 = rect\
    .copy()\
    .rename(columns={"T1": "T2", "T2": "T1"})\
    .assign(Pred=lambda x: 1-x["Pred"])

rect = pd.concat([rect, rect2], axis=0)\
    .pivot(index="T1", columns="T2", values="Pred")

rect.to_csv("data/clean/rectangular_predictions.csv", index=True)
