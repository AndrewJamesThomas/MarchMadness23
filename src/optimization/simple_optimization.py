
import pandas as pd
import numpy as np
from pyomo.environ import *
import warnings
warnings.simplefilter(action='ignore')

points = pd.read_csv("data/simulation/simulated_tournament_base_points.csv")
points = points.drop(['Unnamed: 0'], axis=1)

winners = pd.read_csv("data/simulation/simulated_tournament_winner.csv")
winners = winners.drop(['Unnamed: 0'], axis=1)

details = pd.read_csv("data/simulation/team_details.csv")

bonus = pd.read_csv("data/simulation/simulated_tournament_bonus.csv")
bonus = bonus.drop(['Unnamed: 0'], axis=1)
bonus = bonus + 1

looser = pd.read_csv("data/simulation/simulated_tournament_looser.csv")
looser = looser.drop(['Unnamed: 0'], axis=1)

teams = details["team_id"]
games = winners.columns
games = list([games + "_W"][0]) + list([games + "_L"][0])

sims = points.index


def objective():
    base_points = sum([sum([points.loc[i, g[:3]] *
                            model.dv[(winners.loc[i, g[:3]], g)]
                            for g in games if g[3:] != "_L"]) for i in sims])

    bonus_points = sum([sum([bonus.loc[i, g[:3]] *
                             model.dv[(looser.loc[i, g[:3]], g)]
                             for g in games if g[3:] != "_W"]) for i in sims])

    return base_points + bonus_points


# initialize optimization model
model = ConcreteModel()

# establish DVs and objective
model.dv = Var(teams, games, domain=Binary)
model.points = Objective(expr=objective(),
                         sense=maximize)

# set up constraints
model.cons = ConstraintList()
for g in games:
    model.cons.add(sum([model.dv[t, g] for t in teams]) == 1)


def tournament_rule_winner(model, team, primary_game, dependent_game_1, dependent_game_2):
    return model.dv[team, primary_game + "_W"] <= (model.dv[team, dependent_game_1 + "_W"] + model.dv[team, dependent_game_2 + "_W"])


def tournament_rule_looser(model, team, primary_game, dependent_game_1, dependent_game_2):
    return model.dv[team, primary_game + "_L"] <= (model.dv[team, dependent_game_1 + "_W"] + model.dv[team, dependent_game_2 + "_W"])


def winner_looser_rule(model, team, game):
    w = game + "_W"
    l = game + "_L"
    return (model.dv[team, w] + model.dv[team, l]) <= 1


for t in teams:
    for g in set([i[:3] for i in games]):
        model.cons.add(winner_looser_rule(model, t, g))


for t in teams:
    model.cons.add(tournament_rule_winner(model, t, "E69", "E51", "E52"))

    model.cons.add(tournament_rule_winner(model, t, "E51", "A41", "B41"))
    model.cons.add(tournament_rule_winner(model, t, "E52", "C41", "D41"))

    model.cons.add(tournament_rule_winner(model, t, "A41", "A31", "A32"))
    model.cons.add(tournament_rule_winner(model, t, "B41", "B31", "B32"))
    model.cons.add(tournament_rule_winner(model, t, "C41", "C31", "C32"))
    model.cons.add(tournament_rule_winner(model, t, "D41", "D31", "D32"))

    model.cons.add(tournament_rule_winner(model, t, "A31", "A21", "A22"))
    model.cons.add(tournament_rule_winner(model, t, "A32", "A23", "A24"))
    model.cons.add(tournament_rule_winner(model, t, "B31", "B21", "B22"))
    model.cons.add(tournament_rule_winner(model, t, "B32", "B23", "B24"))
    model.cons.add(tournament_rule_winner(model, t, "C31", "C21", "C22"))
    model.cons.add(tournament_rule_winner(model, t, "C32", "C23", "C24"))
    model.cons.add(tournament_rule_winner(model, t, "D31", "D21", "D22"))
    model.cons.add(tournament_rule_winner(model, t, "D32", "D23", "D24"))

    model.cons.add(tournament_rule_winner(model, t, "A21", "A11", "A12"))
    model.cons.add(tournament_rule_winner(model, t, "A22", "A13", "A14"))
    model.cons.add(tournament_rule_winner(model, t, "A23", "A15", "A16"))
    model.cons.add(tournament_rule_winner(model, t, "A24", "A17", "A18"))
    model.cons.add(tournament_rule_winner(model, t, "B21", "B11", "B12"))
    model.cons.add(tournament_rule_winner(model, t, "B22", "B13", "B14"))
    model.cons.add(tournament_rule_winner(model, t, "B23", "B15", "B16"))
    model.cons.add(tournament_rule_winner(model, t, "B24", "B17", "B18"))
    model.cons.add(tournament_rule_winner(model, t, "C21", "C11", "C12"))
    model.cons.add(tournament_rule_winner(model, t, "C22", "C13", "C14"))
    model.cons.add(tournament_rule_winner(model, t, "C23", "C15", "C16"))
    model.cons.add(tournament_rule_winner(model, t, "C24", "C17", "C18"))
    model.cons.add(tournament_rule_winner(model, t, "D21", "D11", "D12"))
    model.cons.add(tournament_rule_winner(model, t, "D22", "D13", "D14"))
    model.cons.add(tournament_rule_winner(model, t, "D23", "D15", "D16"))
    model.cons.add(tournament_rule_winner(model, t, "D24", "D17", "D18"))


for t in teams:
    model.cons.add(tournament_rule_looser(model, t, "E69", "E51", "E52"))

    model.cons.add(tournament_rule_looser(model, t, "E51", "A41", "B41"))
    model.cons.add(tournament_rule_looser(model, t, "E52", "C41", "D41"))

    model.cons.add(tournament_rule_looser(model, t, "A41", "A31", "A32"))
    model.cons.add(tournament_rule_looser(model, t, "B41", "B31", "B32"))
    model.cons.add(tournament_rule_looser(model, t, "C41", "C31", "C32"))
    model.cons.add(tournament_rule_looser(model, t, "D41", "D31", "D32"))

    model.cons.add(tournament_rule_looser(model, t, "A31", "A21", "A22"))
    model.cons.add(tournament_rule_looser(model, t, "A32", "A23", "A24"))
    model.cons.add(tournament_rule_looser(model, t, "B31", "B21", "B22"))
    model.cons.add(tournament_rule_looser(model, t, "B32", "B23", "B24"))
    model.cons.add(tournament_rule_looser(model, t, "C31", "C21", "C22"))
    model.cons.add(tournament_rule_looser(model, t, "C32", "C23", "C24"))
    model.cons.add(tournament_rule_looser(model, t, "D31", "D21", "D22"))
    model.cons.add(tournament_rule_looser(model, t, "D32", "D23", "D24"))

    model.cons.add(tournament_rule_looser(model, t, "A21", "A11", "A12"))
    model.cons.add(tournament_rule_looser(model, t, "A22", "A13", "A14"))
    model.cons.add(tournament_rule_looser(model, t, "A23", "A15", "A16"))
    model.cons.add(tournament_rule_looser(model, t, "A24", "A17", "A18"))
    model.cons.add(tournament_rule_looser(model, t, "B21", "B11", "B12"))
    model.cons.add(tournament_rule_looser(model, t, "B22", "B13", "B14"))
    model.cons.add(tournament_rule_looser(model, t, "B23", "B15", "B16"))
    model.cons.add(tournament_rule_looser(model, t, "B24", "B17", "B18"))
    model.cons.add(tournament_rule_looser(model, t, "C21", "C11", "C12"))
    model.cons.add(tournament_rule_looser(model, t, "C22", "C13", "C14"))
    model.cons.add(tournament_rule_looser(model, t, "C23", "C15", "C16"))
    model.cons.add(tournament_rule_looser(model, t, "C24", "C17", "C18"))
    model.cons.add(tournament_rule_looser(model, t, "D21", "D11", "D12"))
    model.cons.add(tournament_rule_looser(model, t, "D22", "D13", "D14"))
    model.cons.add(tournament_rule_looser(model, t, "D23", "D15", "D16"))
    model.cons.add(tournament_rule_looser(model, t, "D24", "D17", "D18"))


# Solver the problem
SolverFactory("glpk").solve(model)


# save output
output = pd.DataFrame([(model.dv[i](), i[0], i[1]) for i in model.dv])\
    .pivot(index=1, columns=2)[0]\
    .reset_index()\
    .merge(details, left_on=1, right_on="team_id")\
    .drop([1, 'seed', 'team_id'], axis=1)\
    .set_index("team_name")

output[[i for i in output.keys() if i[3:]=="_W"]].to_csv("data/results/optimal_results.csv", index=True)

# export results distribution
"""
pd.DataFrame([sum([points.loc[i, g] * model.dv[(winners.loc[i, g], g)]() for g in games]) for i in sims])\
    .to_csv("data/results/optimal_distribution.csv", index=False)
"""