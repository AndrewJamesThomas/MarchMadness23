import pandas as pd

# Part I: Import Data
regresults = pd.read_csv("data/raw/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv")
results = pd.read_csv("data/raw/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv")
sub = pd.read_csv("data/raw/MDataFiles_Stage2/MSampleSubmissionStage2.csv")
seeds = pd.read_csv("data/raw/MDataFiles_Stage2/MNCAATourneySeeds.csv")
seeds["Seed"] = seeds["Seed"].str.extract("(\d+)").astype("int")

ordinals = pd.read_csv("data/raw/MDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv")

# Part II: double data by swapping positions
# do this first for the regular season
df1 = regresults[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "NumOT", "WFGA", "WAst", "WBlk", "LFGA",
                 "LAst", "LBlk"]]
df1.columns = ["Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT", "T1_fga", "T1_ast", "T1_blk",
               "T2_fga", "T2_ast", "T2_blk"]

df2 = regresults[["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "LFGA", "LAst", "LBlk", "WFGA",
                 "WAst", "WBlk"]]
df2.columns = ["Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT", "T1_fga", "T1_ast", "T1_blk",
               "T2_fga", "T2_ast", "T2_blk"]

regular_season_results = pd.concat([df1, df2], axis=0)

# now do it for the tournament
df1 = results\
    [["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]]\
    .assign(ResultDiff=lambda x: x["WScore"] - x["LScore"])
df1.columns = ["Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff"]

df2 = results\
    [["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore"]]\
.assign(ResultDiff=lambda x: x["LScore"] - x["WScore"])
df2.columns = ["Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff"]

tourney_results = pd.concat([df1, df2], axis=0)

# Part III: Assign Team Quality
# In this case we are going to use ordinals, but could be Elo, GLMM Random FX, or some other assesment
# select most recent only
ordinals = ordinals\
    .pivot(index=["Season", "TeamID", "RankingDayNum"], columns="SystemName", values="OrdinalRank")\
    [["DII", "MAS", "MOR", "POM", "SAG"]]\
    .reset_index()\
    .query("Season >= 2013")\
    .dropna()\
    .groupby(["Season", "TeamID"])\
    .max("RankingDayNum")\
    .reset_index()

# Part IV: Create Season Aggregates
regular_season_aggregates = regular_season_results\
    .assign(win14days=lambda x: (x["DayNum"] > 118) & (x["T1_Points"] > x["T2_Points"]),
            last14days=lambda x: (x["DayNum"] > 118),
            PointsDiff=lambda x: x["T1_Points"] - x["T2_Points"])\
    .groupby(["Season", "T1"])\
    .agg({
        "T1_Points": ["mean", "median"],
        "T1_fga": ["mean", "median", "min", "max"],
        "T1_ast": ["mean"],
        "T1_blk": ["mean"],
        "T2_fga": ["mean", "min"],
        "PointsDiff": ["mean"],
        "win14days": ["sum"],
        "last14days": ["sum"]
         })\
    .reset_index()\
    .assign(WinRatio14d=lambda x: x[('win14days', 'sum')]/x[('last14days', 'sum')])

regular_season_aggregates.columns = [
    "Season", "TeamID", "T1PntsMean", "T1PntsMedian", "T1FgaMean", "T1FgaMedian", "T1FgaMin",
    "T1FgaMax", "T1AstMean", "T1BlkMean", "T2FgaMean", "T2FgaMin", "PointsDiffMean", "Win14Days",
    "Last14Days", "WinRatio14D"
]

regular_season_aggregates = regular_season_aggregates.drop(["Win14Days", "Last14Days"], axis=1)

# Part V: Merge Everything Together
regular_season_aggregates = regular_season_aggregates\
    .merge(seeds, on=["Season", "TeamID"])\
    .merge(ordinals, on=["Season", "TeamID"])\
    .drop(["RankingDayNum"], axis=1)

full_data = tourney_results\
    .merge(regular_season_aggregates,
           left_on=["Season", "T1"],
           right_on=["Season", "TeamID"],
           how="inner")\
    .merge(regular_season_aggregates,
           left_on=["Season", "T2"],
           right_on=["Season", "TeamID"],
           how="inner")\
    .assign(SeedDiff=lambda x: x["Seed_x"] - x["Seed_y"])\
    .drop(["TeamID_x", "TeamID_y"], axis=1)\
    .rename(columns={"T1": "T_x", "T2": "T_y"})

# Part VI: Save Data
full_data.to_csv("data/clean/modeling_data.csv", index=False)
