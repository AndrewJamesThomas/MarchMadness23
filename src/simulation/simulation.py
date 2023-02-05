import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

pred = pd.read_csv("data/clean/rectangular_predictions.csv")
df = pd.read_csv("data/simulation/tournament_structure.csv")


def select_winner(team_1, team_2, pred=pred):
    """provides a winner from the given two teams. This is based on the predictions and a random number"""
    proba = pred.loc[pred["T1"] == int(team_1), str(int(team_2))].values[0]
    random_number = np.random.random()
    if proba >= random_number:
        return team_1
    else:
        return team_2


def run_tourney(input_data):
    data = input_data.copy()
    for i in range(0, len(data)):
        print(i)
        row = data.iloc[i, :].copy()
        if row[["team_1"]].isnull().values[0]:
            row[["team_1"]] = data[data["ID"] == row["prev_game_1"]]["winner"]
            # insert winner from "prev_game_1" insert winner
        if row[["team_2"]].isnull().values[0]:
            row[["team_2"]] = data[data["ID"] == row["prev_game_2"]]["winner"]
        if row[["winner"]].isnull().values[0]:
            # select winner
            row["winner"] = select_winner(row["team_1"], row["team_2"])
        data.iloc[i, :] = row
    return data[['ID', "winner"]]


outcome = run_tourney(df)

# return dataframe with all game IDs and winners

# repeat function N times, append output together

