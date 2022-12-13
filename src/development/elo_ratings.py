import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# PART I: Functions
def get_elo(year, day, team_id):
    val = elo[(elo['year'] == year) &
                 (elo['day'] == day) &
                 (elo['team_id'] == team_id)]\
        ["elo"]\
        .values
    if len(val) > 0:
        return val[0]
    else:
        return -1


def get_expected_value(ra, rb):
    return 1/(1+10**((rb-ra)/400))


def update_ratings(init_rating, win_proba, win_ind, k):
    return init_rating + k * (win_ind - win_proba)


def run_elo(year, day, winner, looser, k=10):
    # get elo values
    el1 = get_elo(year, day, winner)
    el2 = get_elo(year, day, looser)
    if el1 != -1 and el2 != -1:
        # convert to probabilities
        winner_proba = get_expected_value(el1, el2)

        # update ratings
        el1_new = update_ratings(el1, winner_proba, True, k)
        el2_new = update_ratings(el2, 1-winner_proba, False, k)

        # update data
        elo.loc[(elo["year"] == year) &
                (elo["day"] > day) &
                (elo["team_id"] == winner),
                "elo"] = el1_new

        elo.loc[(elo["year"] == year) &
                (elo["day"] > day) &
                (elo["team_id"] == looser),
                "elo"] = el2_new

    return elo


# PART II: import data
stats = pd.read_csv("data/clean/inputs/independent_vars.csv")

df = pd.read_csv("data/clean/inputs/game_results.csv")
df = df[["year", "day", "winner_team_id", "looser_team_id"]]
df = df.query("year >= 2017").reset_index(drop=True)

# initialize elo ratings
elo = stats[["year", "day", "team_id"]]
elo["elo"] = 1000

# PART III: run calculations
for i in range(0, max(df.index)-1):
    completion = round(i/max(df.index)*100, 2)
    print(f"Progress: {completion}% Complete", end="\r")
    row = df.iloc[i]
    elo = run_elo(row["year"], row["day"], row["winner_team_id"], row["looser_team_id"], k=24)

# PART IV: Save results back to hard drive
stats = stats.merge(elo, on=["year", "day", "team_id"])
stats.to_csv("data/clean/inputs/independent_vars.csv", index=False)

