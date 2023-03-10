import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv")
rankings = pd.read_csv("data/raw/MDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv")

# ---- convert from long to wide
rankings = rankings.pivot(index=["Season", "RankingDayNum", "TeamID"],
                          columns="SystemName",
                          values="OrdinalRank")

# ---- handle null values
rankings = rankings\
    .reset_index()\
    [["Season", "RankingDayNum", "TeamID", "DOK", "MAS", "MOR", "POM", "SAG"]]\
    .query("Season>=2014")

# ---- rename columns
rankings.columns = ['season', "day_num", "team_id", "dok", "mas", "mor", "pom", "sag"]

# ---- extend data out to all possible days
years = pd.DataFrame(data={"season": np.arange(2014, 2023), "key": 0})
games = pd.DataFrame(data={"day_num": np.arange(0, 134), "key": 0})
teams = rankings[["team_id"]].drop_duplicates()
teams["key"] = 0

rankings = years\
    .merge(games, on="key")\
    .merge(teams, on="key")\
    .drop(["key"], axis=1)\
    .merge(rankings,
           how="left",
           left_on=["season", "day_num", "team_id"],
           right_on=["season", "day_num", "team_id"])\
    .sort_values(["season", "team_id", "day_num"])\
    .set_index(["season", "team_id"])\
    .groupby(["season", "team_id"])\
    .fillna(method="ffill")\
    .reset_index()\
    .dropna()

# ---- Format games
winners = df\
    [["Season", "DayNum", "WTeamID", "LTeamID"]]\
    .rename(columns={"WTeamID": "team_1", "LTeamID": "team_2",
                     "Season": "season", "DayNum": "day_num"})\
    .assign(Win=True)

loosers = df\
    [["Season", "DayNum", "WTeamID", "LTeamID"]]\
    .rename(columns={"WTeamID": "team_2", "LTeamID": "team_1",
                     "Season": "season", "DayNum": "day_num"})\
    .assign(Win=False)

games = pd.concat([winners, loosers], axis=0)\
    .query("season>=2014")

modeling_data = games\
    .merge(rankings, how="left",
           left_on=["season", "team_1", "day_num"],
           right_on=["season", "team_id", "day_num"])\
    .merge(rankings, how="left",
           left_on=["season", "team_2", "day_num"],
           right_on=["season", "team_id", "day_num"])\
    .dropna()

# ---- Save data
rankings.to_csv("data/clean/modeling_2/rankings.csv", index=False)
modeling_data.to_csv("data/clean/modeling_2/modeling_data.csv", index=False)

