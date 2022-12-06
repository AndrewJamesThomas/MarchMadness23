import pandas as pd
import networkx as nx
import warnings

# turn off pandas warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/clean/inputs/game_results.csv")
df = df[["year", "day", "winner_team_id", "looser_team_id"]]

ordinals = pd.read_csv("data/clean/inputs/ordinals.csv")


# function for getting page rank
def get_page_rank(year, game, data=df):
    # POC
    data = data.query(f"year=={year}").query(f"day<={game}")
    network = nx.from_pandas_edgelist(data,
                                      source="winner_team_id",
                                      target="looser_team_id",
                                      edge_attr=True)
    page_rank = nx.pagerank(network)
    page_rank = pd.DataFrame(data={"team_id": page_rank.keys(),
                                   "year": year,
                                   "day": game,
                                   "pagerank": page_rank.values()})
    return page_rank


first_dataframe = True
output = None
# Loop through years and add page rank
for y in ordinals["year"].unique():
    # loop through games
    for g in ordinals["day"].unique():
        print(f"Working on: year {y}/2022 game {g}/133.", end="\r")
        page_rank = get_page_rank(year=y, game=g)
        if first_dataframe:
            output = page_rank.copy()
            first_dataframe = False
        else:
            output = output.append(page_rank)

ordinals = ordinals.merge(output, on=["team_id", "year", "day"], how="left")
ordinals["pagerank"] = ordinals["pagerank"]*10000

# export
ordinals.to_csv("data/clean/inputs/independent_vars.csv", index=False)

