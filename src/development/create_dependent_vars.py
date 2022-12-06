import pandas as pd

df = pd.read_csv("data/clean/inputs/game_results.csv")
df = df[["year", "day", "winner_team_id", "looser_team_id"]]

# rename fields
df.columns = ["year", "day", "team_1", "team_2"]

# create win_ind field
df["win_ind"] = True

# create game_id field
df = df.reset_index().rename(columns={"index": "game_id"})

# add duplicate layers with team_1 and team_2 switched
df = df\
    .rename(columns={"team_1": "team_2", "team_2": "team_1"})\
    .assign(win_ind=False)\
    .append(df)\
    .sort_values(["game_id", "team_1"])

# drop duplicate fields so there's only one row per game
df["first_team"] = df.groupby("game_id").cumcount()
df = df.query("first_team==0")

# select only relevant fields
df = df[["year", "day", "team_1", "team_2", "win_ind"]]

# save data
df.to_csv("data/clean/inputs/dependent_vars.csv", index=False)