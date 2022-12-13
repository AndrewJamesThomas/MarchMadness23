import pandas as pd

stats = pd.read_csv("data/clean/inputs/independent_vars.csv")
df = pd.read_csv("data/clean/inputs/game_results.csv")

# Part 1: create two dataframes:
# df1: is just the normal df, but all the winner/looser fields are changed to "x_"/"y_" fields.

# df2: the same thing but the fields are all switched. (so looser is no "y_", etc)

# append these two dataframes together. So each team appears as both an "x_" and a "y_" for each game

# Part 2: Loop through the 'stats' data.
# for each year/day/team get aggregate stats up until that game.

# agg stats should include: avg, std, min, max, win_ratio, etc. for both the team and their opponent

# join these aggregates to the stats dataframe and the location they were extracted from

# part 3: save
# save back and remodel
