import pandas as pd

stats = pd.read_csv("data/clean/inputs/independent_vars.csv")
df = pd.read_csv("data/clean/inputs/game_results.csv")

# Part 1: create two dataframes:
# df1: is just the normal df, but all the winner/looser fields are changed to "x_"/"y_" fields.
df1 = df[['year', 'day', 'winner_team_id', 'winner_score',
          'looser_score', 'winner_field_goals_made',
          'winner_field_goals_attempted', 'winner_threes_made',
          'winner_threes_attempted', 'winner_free_throws_made',
          'winner_free_thres_attempted', 'winner_offensive_rebounds',
          'winner_defensive_rebounds', 'winner_assists', 'winner_turnovers',
          'winner_steals', 'winner_blocks', 'winner_personal_fouls',
          'looser_field_goals_made', 'looser_field_goals_attempted',
          'looser_threes_made', 'looser_threes_attempted',
          'looser_free_throws_made', 'looser_free_throws_attempts',
          'looser_offensive_rebounds', 'looser_defensive_robounds',
          'looser_assists', 'looser_turnovers', 'looser_steals', 'looser_blocks',
          'looser_personal_fouls']]

df1.columns = ['year', 'day', 'team_id', 'x_score',
               'y_score', 'x_field_goals_made',
               'x_field_goals_attempted', 'x_threes_made',
               'x_threes_attempted', 'x_free_throws_made',
               'x_free_throws_attempted', 'x_offensive_rebounds',
               'x_defensive_rebounds', 'x_assists', 'x_turnovers',
               'x_steals', 'x_blocks', 'x_personal_fouls',
               'y_field_goals_made', 'y_field_goals_attempted',
               'y_threes_made', 'y_threes_attempted',
               'y_free_throws_made', 'y_free_throws_attempted',
               'y_offensive_rebounds', 'y_defensive_rebounds',
               'y_assists', 'y_turnovers', 'y_steals', 'y_blocks',
               'y_personal_fouls']

# df2: the same thing but the fields are all switched. (so looser is no "y_", etc)
df2 = df[['year', 'day', 'looser_team_id', 'winner_score',
          'looser_score', 'winner_field_goals_made',
          'winner_field_goals_attempted', 'winner_threes_made',
          'winner_threes_attempted', 'winner_free_throws_made',
          'winner_free_thres_attempted', 'winner_offensive_rebounds',
          'winner_defensive_rebounds', 'winner_assists', 'winner_turnovers',
          'winner_steals', 'winner_blocks', 'winner_personal_fouls',
          'looser_field_goals_made', 'looser_field_goals_attempted',
          'looser_threes_made', 'looser_threes_attempted',
          'looser_free_throws_made', 'looser_free_throws_attempts',
          'looser_offensive_rebounds', 'looser_defensive_robounds',
          'looser_assists', 'looser_turnovers', 'looser_steals', 'looser_blocks',
          'looser_personal_fouls']]

df2.columns = ['year', 'day', 'team_id', 'y_score',
               'x_score', 'y_field_goals_made',
               'y_field_goals_attempted', 'y_threes_made',
               'y_threes_attempted', 'y_free_throws_made',
               'y_free_throws_attempted', 'y_offensive_rebounds',
               'y_defensive_rebounds', 'y_assists', 'y_turnovers',
               'y_steals', 'y_blocks', 'y_personal_fouls',
               'x_field_goals_made', 'x_field_goals_attempted',
               'x_threes_made', 'x_threes_attempted',
               'x_free_throws_made', 'x_free_throws_attempted',
               'x_offensive_rebounds', 'x_defensive_rebounds',
               'x_assists', 'x_turnovers', 'x_steals', 'x_blocks',
               'x_personal_fouls']

# append these two dataframes together. So each team appears as both an "x_" and a "y_" for each game
df_main = df1.append(df2)

# Part 2: Loop through the 'stats' data.
# for each year/day/team get aggregate stats up until that game.


# agg stats should include: avg, std, min, max, win_ratio, etc. for both the team and their opponent

# join these aggregates to the stats dataframe and the location they were extracted from

# part 3: save
# save back and remodel
