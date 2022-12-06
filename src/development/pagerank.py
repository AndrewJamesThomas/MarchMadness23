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


# Loop through years and add page rank
for y in ordinals["year"].unique():
    # loop through games
    for g in ordinals["day"].unique():
        print(f"Working on: year {y}/2022 game {g}/133.", end="\r")
        page_rank = get_page_rank(year=y, game=g)
        ordinals = ordinals\
            .merge(page_rank, on=["year", "day", "team_id"], how="left")
        for t in page_rank["team_id"].unique():
            ordinals\
                [(ordinals['year'] == y) &
                 (ordinals["day"] == g) *
                 (ordinals['team_id'] == t)]\
                ["pagerank"] = page_rank[page_rank["team_id"] == t]["pagerank"]

# explore relationships between ordinal rankings and page ranks
# we expect them to be very similiar but not exactly the same
# unclear if the relationship will be positive or negative




