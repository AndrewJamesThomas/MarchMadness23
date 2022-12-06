import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from scipy.stats import uniform

# import data
x_df = pd.read_csv("data/clean/inputs/independent_vars.csv")
y_df = pd.read_csv("data/clean/inputs/dependent_vars.csv")

# merge data together
df = y_df\
    .merge(x_df, left_on=["year", "day", "team_1"], right_on=["year", "day", "team_id"], how="inner")\
    .merge(x_df, left_on=["year", "day", "team_2"], right_on=["year", "day", "team_id"], how="inner",
           suffixes=("_team_1", "_team_2"))\
    .drop(['year', 'day', 'team_1', 'team_2', 'team_id_team_1', 'team_id_team_2'], axis=1)

# last minute data processing
X = df.drop(["win_ind"], axis=1)
y = df["win_ind"]

# train test split
RANDOM_SEED = 4206969
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RANDOM_SEED)

# model 1: unregularized logistic regression
model_1 = LogisticRegression()
model_1.fit(x_train, y_train)
proba_1 = model_1.predict_proba(x_test)[:, 1]
results_1 = log_loss(y_test, proba_1)

# model 2: regularized lasso regression
grid_2 = {"C": uniform()}
model_2 = LogisticRegression(penalty="l1", solver="liblinear")
srch_2 = RandomizedSearchCV(model_2, grid_2, scoring="neg_log_loss", cv=10, verbose=9)
srch_2.fit(x_train, y_train)
proba_2 = srch_2.predict_proba(x_test)[:, 1]
results_2 = log_loss(y_test, proba_2)

# model 3: regularized lasso regression
grid_3 = {"C": uniform()}
model_3 = LogisticRegression(penalty="l2", solver="liblinear")
srch_3 = RandomizedSearchCV(model_3, grid_3, scoring="neg_log_loss", cv=10, verbose=9)
srch_3.fit(x_train, y_train)
proba_3 = srch_3.predict_proba(x_test)[:, 1]
results_3 = log_loss(y_test, proba_3)

# model 4: elasticnet
grid_4 = {"C": uniform(), "l1_ratio": uniform()}
model_4 = LogisticRegression(penalty="elasticnet", solver="saga")
srch_4 = RandomizedSearchCV(model_4, grid_4, scoring="neg_log_loss", cv=10, verbose=9)
srch_4.fit(x_train, y_train)
proba_4 = srch_4.predict_proba(x_test)[:, 1]
results_4 = log_loss(y_test, proba_4)

# model 5: decision tree
grid_5 = {"max_depth": [3, 4, 5, 6, 7, 8]}
model_5 = DecisionTreeClassifier()
srch_5 = RandomizedSearchCV(model_5, grid_5, scoring="neg_log_loss", cv=10, verbose=9)
srch_5.fit(x_train, y_train)
proba_5 = srch_5.predict_proba(x_test)[:, 1]
results_5 = log_loss(y_test, proba_5)