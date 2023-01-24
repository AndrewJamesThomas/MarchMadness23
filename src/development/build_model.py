###########
# TODO: rerun all models, reselect best model, update final model training with new best model
###########
import warnings
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import joblib

warnings.filterwarnings("ignore")
df = pd.read_csv("data/clean/modeling_data.csv")

# train/test split
train = df.query("Season<2021")
test = df.query("Season==2021")

y_train = train["ResultDiff"] > 0
y_test = test["ResultDiff"] > 0

x_train = train.drop(['Season', 'DayNum', 'T_x', 'T_y', 'T1_Points', 'T2_Points', 'ResultDiff'], axis=1)
x_test = test.drop(['Season', 'DayNum', 'T_x', 'T_y', 'T1_Points', 'T2_Points', 'ResultDiff'], axis=1)

# Scale data
scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)


# Model 1, simple logistic regression; 0.63
model_1 = LogisticRegression()
model_1.fit(x_train, y_train)
proba_1 = model_1.predict_proba(x_test)[:, 1]
log_loss_1 = log_loss(y_test, proba_1)
print(f"Log Loss of Model #1 is {log_loss_1:.4f}")

# Model 2, Lasso Regression; 0.61
grid_2 = {
    "C": uniform()
}
model_2 = LogisticRegression(solver="liblinear", penalty="l1")
srch_2 = RandomizedSearchCV(model_2, grid_2, cv=10, scoring="neg_log_loss", verbose=99)
srch_2.fit(x_train, y_train)
proba_2 = srch_2.predict_proba(x_test)[:, 1]
log_loss_2 = log_loss(y_test, proba_2)
print(f"Log Loss of Model #2 is {log_loss_2:.4f}")

# Model 3, Ridge Regression 0.63
grid_3 = {
    "C": uniform()
}
model_3 = LogisticRegression(solver="liblinear", penalty="l2")
srch_3 = RandomizedSearchCV(model_3, grid_3, cv=10, scoring="neg_log_loss", verbose=99)
srch_3.fit(x_train, y_train)
proba_3 = srch_3.predict_proba(x_test)[:, 1]
log_loss_3 = log_loss(y_test, proba_3)
print(f"Log Loss of Model #3 is {log_loss_3:.4f}")

# Model 4, ElasticNet; 0.63
grid_4 = {
    "C": uniform(),
    "l1_ratio": uniform()
}
model_4 = LogisticRegression(solver="saga", penalty="elasticnet")
srch_4 = RandomizedSearchCV(model_4, grid_4, cv=10, scoring="neg_log_loss", verbose=99)
srch_4.fit(x_train, y_train)
proba_4 = srch_4.predict_proba(x_test)[:, 1]
log_loss_4 = log_loss(y_test, proba_4)
print(f"Log Loss of Model #4 is {log_loss_4:.4f}")

# Model 6: Decision Trees; 0.69
grid_6 = {"max_depth": [1, 2, 3, 4]}
model_6 = DecisionTreeClassifier()
srch_6 = RandomizedSearchCV(model_6, grid_6, cv=10, scoring="neg_log_loss", verbose=99)
srch_6.fit(x_train, y_train)
proba_6 = srch_6.predict_proba(x_test)[:, 1]
log_loss_6 = log_loss(y_test, proba_6)
print(f"Log Loss of Model #4 is {log_loss_6:.4f}")

# Model 7: Random Foresst; 0.63
grid_7 = {'bootstrap': [True, False],
          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
          'max_features': ['auto', 'sqrt'],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

model_7 = RandomForestClassifier()
srch_7 = RandomizedSearchCV(model_7, grid_7, cv=10, scoring="neg_log_loss", verbose=99)
srch_7.fit(x_train, y_train)
proba_7 = srch_7.predict_proba(x_test)[:, 1]
log_loss_7 = log_loss(y_test, proba_7)
print(f"Log Loss of Model #7 is {log_loss_7:.4f}")

# Model 8, XGBoost; 0.62
grid_8 = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1
    "max_depth": [2, 3, 4, 5, 1], # default 3
    "n_estimators": [80, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94], # default 100
    "subsample": uniform(0.6, 0.4)
}
model_8 = xgb.XGBClassifier(objective="binary:logistic", random_state=4206969)
srch_8 = RandomizedSearchCV(model_8, grid_8, scoring="neg_log_loss", cv=10, verbose=9)
srch_8.fit(x_train, y_train)

proba_8 = srch_8.predict_proba(x_test)[:, 1]
results_8 = log_loss(y_test, proba_8)
print(results_8)

# as always, LASSO regression wins
# rerun model on all data, with hyperparameter
final_x = df.drop(['Season', 'DayNum', 'T_x', 'T_y', 'T1_Points', 'T2_Points', 'ResultDiff'], axis=1)
final_y = df["ResultDiff"] > 0

scl2 = MinMaxScaler()
final_x = scl2.fit_transform(final_x)

final_model = LogisticRegression(solver="liblinear", penalty="l1", C=0.5795772119388047)
final_model.fit(final_x, final_y)

# export model here
joblib.dump(final_model, "src/development/model.sav")
