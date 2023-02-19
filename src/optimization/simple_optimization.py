import pandas as pd
import numpy as np
from pyomo.environ import *
import warnings
warnings.simplefilter(action='ignore')

# TODO Update paths
points = pd.read_csv("data/simulation/simulated_tournament_base_points.csv")
points = points.drop(['Unnamed: 0'], axis=1)

winners = pd.read_csv("data/simulation/simulated_tournament_winner.csv")
winners = winners.drop(['Unnamed: 0'], axis=1)

details = pd.read_csv("data/simulation/team_details.csv")

teams = details["team_id"]
games = winners.columns
sims = points.index

# initialize optimization model
model = ConcreteModel()

# establish DVs and objective
model.dv = Var(teams, games, domain=Binary)
model.points = Objective(expr=sum([sum([points.loc[i, g] * model.dv[(winners.loc[i, g], g)] for g in games]) for i in sims]),
                         sense=maximize)

# set up constraints
model.cons = ConstraintList()
for g in games:
    model.cons.add(sum([model.dv[t, g] for t in teams]) == 1)


def tournament_rule(model, team, primary_game, dependent_game_1, dependent_game_2):
    return model.dv[team, primary_game] <= (model.dv[team, dependent_game_1] + model.dv[team, dependent_game_2])


for t in teams:
    model.cons.add(tournament_rule(model, t, "E69", "E51", "E52"))

    model.cons.add(tournament_rule(model, t, "E51", "A41", "B41"))
    model.cons.add(tournament_rule(model, t, "E52", "C41", "D41"))

    model.cons.add(tournament_rule(model, t, "A41", "A31", "A32"))
    model.cons.add(tournament_rule(model, t, "B41", "B31", "B32"))
    model.cons.add(tournament_rule(model, t, "C41", "C31", "C32"))
    model.cons.add(tournament_rule(model, t, "D41", "D31", "D32"))

    model.cons.add(tournament_rule(model, t, "A31", "A21", "A22"))
    model.cons.add(tournament_rule(model, t, "A32", "A23", "A24"))
    model.cons.add(tournament_rule(model, t, "B31", "B21", "B22"))
    model.cons.add(tournament_rule(model, t, "B32", "B23", "B24"))
    model.cons.add(tournament_rule(model, t, "C31", "C21", "C22"))
    model.cons.add(tournament_rule(model, t, "C32", "C23", "C24"))
    model.cons.add(tournament_rule(model, t, "D31", "D21", "D22"))
    model.cons.add(tournament_rule(model, t, "D32", "D23", "D24"))

    model.cons.add(tournament_rule(model, t, "A21", "A11", "A12"))
    model.cons.add(tournament_rule(model, t, "A22", "A13", "A14"))
    model.cons.add(tournament_rule(model, t, "A23", "A15", "A16"))
    model.cons.add(tournament_rule(model, t, "A24", "A17", "A18"))
    model.cons.add(tournament_rule(model, t, "B21", "B11", "B12"))
    model.cons.add(tournament_rule(model, t, "B22", "B13", "B14"))
    model.cons.add(tournament_rule(model, t, "B23", "B15", "B16"))
    model.cons.add(tournament_rule(model, t, "B24", "B17", "B18"))
    model.cons.add(tournament_rule(model, t, "C21", "C11", "C12"))
    model.cons.add(tournament_rule(model, t, "C22", "C13", "C14"))
    model.cons.add(tournament_rule(model, t, "C23", "C15", "C16"))
    model.cons.add(tournament_rule(model, t, "C24", "C17", "C18"))
    model.cons.add(tournament_rule(model, t, "D21", "D11", "D12"))
    model.cons.add(tournament_rule(model, t, "D22", "D13", "D14"))
    model.cons.add(tournament_rule(model, t, "D23", "D15", "D16"))
    model.cons.add(tournament_rule(model, t, "D24", "D17", "D18"))

# Solver the problem
SolverFactory("glpk").solve(model)
# without constraints we would expect this to return all ones


# save output
pd.DataFrame([(model.dv[i](), i[0], i[1]) for i in model.dv])\
    .pivot(index=1, columns=2)[0]\
    .reset_index()\
    .merge(details, left_on=1, right_on="team_id")\
    .drop([1, 'seed', 'team_id'], axis=1)\
    .set_index("team_name")\
    .to_csv("data/results/optimal_results.csv", index=True)

# export results distribution
pd.DataFrame([sum([points.loc[i, g] * model.dv[(winners.loc[i, g], g)]() for g in games]) for i in sims])\
    .to_csv("data/results/optimal_distribution.csv", index=False)
