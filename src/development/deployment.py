import pandas as pd
from joblib import load

submission = pd.read_csv("data/raw/MDataFiles_Stage2/MSampleSubmissionStage2.csv")
rankings = pd.read_csv("data/clean/modeling_2/rankings.csv")
model = load("src/development3/model.sav")

# Build submission
submission = submission["ID"]\
    .str.split("_", expand=True)\
    .rename(columns={0: "season", 1: "team1", 2: "team2"})\
    .join(submission)

# Build ranking data
rankings = rankings\
    .query("season==2022")\
    [["team_id", "day_num", "season"]]\
    .groupby(["team_id", "season"])\
    .max()\
    .reset_index()\
    .assign(last_game=True)\
    .merge(rankings, how="right", on=["team_id", "day_num", "season"])\
    .query("last_game==True")\
    [["team_id", "dok", "mas", "mor", "pom", 'sag']]

rankings["team_id"] = rankings["team_id"].astype("str")

# build prediction data
x_pred = submission\
    [["team1", "team2"]]\
    .merge(rankings, how="left", left_on=["team1"], right_on=["team_id"])\
    .merge(rankings, how="left", left_on=["team2"], right_on=["team_id"])\
    .drop(["team1", "team2", "team_id_x", "team_id_y"], axis=1)

y_pred = model.predict_proba(x_pred)[:,1]

# save data
submission["Pred"] = y_pred
submission = submission[["ID", "Pred"]]
submission.to_csv("data/clean/modeling_2/submission.csv", index=False)
