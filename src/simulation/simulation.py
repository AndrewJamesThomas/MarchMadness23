import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pred = pd.read_csv("data/clean/rectangular_predictions.csv")
df = pd.read_csv("data/simulation/tournament_structure.csv")


def select_prev_game_winner(game_id):
    """Select the winner of the given game"""
    print(game_id)


def select_winner(team_1, team_2):
    """provides a winner from the given two teams. This is based on the predictions and a random number"""
    pass


# create function:
def run_tourney(data=df):
    for i in range(0, len(data)):
        row = data.iloc[i, :]
        if row[["team_1"]].isnull().values[0]:
            lookup_game_id = data[data["ID"] == row["prev_game_1"]]["winner"]
            select_prev_game_winner(lookup_game_id)
            # insert winner from "prev_game_1" insert winner
        if row[["team_2"]].isnull().values[0]:
            lookup_game_id = data[data["ID"] == row["prev_game_2"]]["winner"]
            select_prev_game_winner(lookup_game_id)
        if row[["winner"]].isnull().values[0]:
            select_winner(11, 22)
            # select winner
        data.iloc[i, :] = row
    return data[['ID', "winner"]]


# return dataframe with all game IDs and winners

# repeat function N times, append output together
