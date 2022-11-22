import pandas as pd
import numpy as np
from itertools import product

# import team names
teams = pd.read_csv("data/raw/MDataFiles_Stage2/MTeams.csv")\
    .rename(columns={"TeamID": "team_id", "TeamName": "team_name"})\
    [["team_id", "team_name"]]

teams.to_csv("data/clean/inputs/team_names.csv", index=False)

# import compact season results
season_results = pd.read_csv("data/raw/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv")\
    .rename(columns={"Season": "year", "DayNum": "day", "WTeamID": "winner_team_id",
                     "LTeamID": "looser_team_id", "WScore": "winner_score",
                     "LScore": "looser_score", "WLoc": "winner_home_ind"})\
    .drop(["NumOT"], axis=1)

season_results["winner_home_ind"] = season_results["winner_home_ind"]\
    .replace(["N", "H", "A"], ["NA", "Home", "Away"])

season_results["game_type"] = "Regular"

# import compact tourney results
tourney_results = pd.read_csv("data/raw/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv")\
    .rename(columns={"Season": "year", "DayNum": "day", "WTeamID": "winner_team_id",
                     "LTeamID": "looser_team_id", "WScore": "winner_score",
                     "LScore": "looser_score", "WLoc": "winner_home_ind"})\
    .drop(["NumOT"], axis=1)

tourney_results["winner_home_ind"] = tourney_results["winner_home_ind"]\
    .replace(["N", "H", "A"], ["NA", "Home", "Away"])

tourney_results["game_type"] = "Tourney"

# combine all results
results = season_results.append(tourney_results)

# rename fields
results = results\
    .rename(columns={
        'WFGM': "winner_field_goals_made",
        'WFGA': "winner_field_goals_attempted",
        'WFGM3': "winner_threes_made",
        'WFGA3': "winner_threes_attempted",
        'WFTM': "winner_free_throws_made",
        'WFTA': "winner_free_thres_attempted",
        'WOR': "winner_offensive_rebounds",
        'WDR': "winner_defensive_rebounds",
        'WAst': "winner_assists",
        'WTO': "winner_turnovers",
        'WStl': "winner_steals",
        'WBlk': "winner_blocks",
        'WPF': "winner_personal_fouls",
        'LFGM': "looser_field_goals_made",
        'LFGA': "looser_field_goals_attempted",
        'LFGM3': "looser_threes_made",
        'LFGA3': "looser_threes_attempted",
        'LFTM': "looser_free_throws_made",
        'LFTA': "looser_free_throws_attempts",
        'LOR': "looser_offensive_rebounds",
        'LDR': "looser_defensive_robounds",
        'LAst': "looser_assists",
        'LTO': "looser_turnovers",
        'LStl': "looser_steals",
        'LBlk': "looser_blocks",
        'LPF': "looser_personal_fouls"})

# Save results
results.to_csv("data/clean/inputs/game_results.csv", index=False)

# process seeds
seeds = pd.read_csv("data/raw/MDataFiles_Stage2/MNCAATourneySeeds.csv")
seeds.columns = ["year", "seed", "team_id"]
seeds.to_csv("data/clean/inputs/seeds.csv", index=False)

# process rankings
rankings = pd.read_csv("data/raw/MDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv")
rankings = rankings\
    .pivot(index=['Season', 'RankingDayNum', 'TeamID'],
           columns="SystemName",
           values="OrdinalRank")\
    .reset_index()\
    [['Season', 'RankingDayNum', 'TeamID', "DII", "MAS", "MOR", "POM", "SAG"]]\
    .query("Season > 2016")\
    .rename(columns={
        "Season": "year", "RankingDayNum": "day", "TeamID": "team_id",
        "DII": "dii", "MAS": "mas", "MOR": "mor", "POM": "pom", "SAG": "sag"})

all_years = np.arange(rankings.year.min(),
                      rankings.year.max() + 1,
                      1)

all_days = np.arange(rankings.day.min(),
                     rankings.day.max() + 1,
                     1)

all_teams = rankings.team_id.unique()

rankings_full = pd.DataFrame([*product(all_years, all_days, all_teams)])\
    .rename(columns={0: "year", 1: "day", 2: "team_id"})\
    .merge(rankings, how="left", on=["year", "day", "team_id"])\
    .sort_values(["team_id", "year", "day"])\
    .query("day>=16")

# impute forward
rankings_full = rankings_full.ffill()

# save data
rankings_full.to_csv("data/clean/inputs/ordinals.csv", index=False)
