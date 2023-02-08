# TODO: Produce bonus matrix

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore')

SEASON = 2022

pred = pd.read_csv("data/clean/rectangular_predictions.csv")
df = pd.read_csv("data/simulation/tournament_structure.csv")
seeds = pd.read_csv("data/raw/MDataFiles_Stage2/MNCAATourneySeeds.csv")
seeds["Seed"] = seeds["Seed"].str.extract("(\d+)").astype("int")


def seed_bonus(winner, looser, seed_data=seeds):
    """accepts 2 team IDs, one winner and one looser. returns the seed bonus that the winner recieves (if any)
    bonus = high seed - low seed; only applies if low seed wins"""
    winner_seed = seeds[(seeds["TeamID"] == winner) & (seeds["Season"] == SEASON)]["Seed"].values[0]
    looser_seed = seeds[(seeds["TeamID"] == looser) & (seeds["Season"] == SEASON)]["Seed"].values[0]
    if winner_seed > looser_seed:
        return winner_seed - looser_seed
    else:
        return 0


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
        row = data.iloc[i, :].copy()
        if row[["team_1"]].isnull().values[0]:
            row[["team_1"]] = data[data["ID"] == row["prev_game_1"]]["winner"]
        if row[["team_2"]].isnull().values[0]:
            row[["team_2"]] = data[data["ID"] == row["prev_game_2"]]["winner"]
        if row[["winner"]].isnull().values[0]:
            row["winner"] = select_winner(row["team_1"], row["team_2"])
        data.iloc[i, :] = row
    return data.set_index("ID")

# I feel like this should return the winner and the looser, rather than team_1 and team_2;
# that would be more compact and easier for the optimization model/creating the bonus matrix
def repeat_simulation(n, data=df):
    for i in tqdm(range(n)):
        if i == 0:
            outcome = run_tourney(data)
            winners = outcome.loc[:, "winner"]
            team_1 = outcome.loc[:, "team_1"]
            team_2 = outcome.loc[:, "team_2"]
            base_points = outcome.loc[:, "base_points"]
        else:
            outcome = run_tourney(data)
            winners = pd.concat([winners, outcome.loc[:, "winner"]], axis=1)
            team_1 = pd.concat([team_1, outcome.loc[:, "team_1"]], axis=1)
            team_2 = pd.concat([team_2, outcome.loc[:, "team_2"]], axis=1)
            base_points = pd.concat([base_points, outcome.loc[:, "base_points"]], axis=1)

    return winners.T, team_1.T, team_2.T, base_points.T


if __name__ == "__main__":
   final_outcome = repeat_simulation(5)
   for d in final_outcome:
       file_name = d.index[0]
       d.to_csv(f"data/simulation/simulated_tournament_{file_name}.csv", index=True)
