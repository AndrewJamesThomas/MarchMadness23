import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.stats import uniform
from joblib import dump

df = pd.read_csv("data/clean/modeling_2/modeling_data.csv")

X = df[['dok_x', 'mas_x', 'mor_x', 'pom_x', 'sag_x', 'dok_y', 'mas_y', 'mor_y', 'pom_y', 'sag_y']]
y = df[["Win"]]

# Pre-Process Data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=4206969)

# Build Models
# Model 1: Basic Regression <- winner
model_1 = LogisticRegression()
model_1.fit(xtrain, ytrain)
pred_1 = model_1.predict_proba(xtest)
ll1 =log_loss(ytest, pred_1) # 0.5583997063860974
print(ll1)

# model 2: Lasso
grid_2 = {"C": uniform}
model_2 = LogisticRegression(penalty="l1", solver="liblinear")
srch = RandomizedSearchCV(model_2, grid_2, scoring="neg_log_loss", cv=10, random_state=4206969, verbose=99)
srch.fit(xtrain, ytrain)
pred_2 = srch.predict_proba(xtest)
ll2 = log_loss(ytest, pred_2)
print(ll2)

# model 3: Ridge
grid_3 = {"C": uniform}
model_3 = LogisticRegression(penalty="l2")
srch_3 = RandomizedSearchCV(model_3, grid_3, scoring="neg_log_loss", cv=10, random_state=4206969, verbose=99)
srch_3.fit(xtrain, ytrain)
pred_3 = srch_3.predict_proba(xtest)
ll3 = log_loss(ytest, pred_3)
print(ll3)

# model 3: ElasticNet
grid_4 = {"C": uniform, "l1_ratio": uniform}
model_4 = LogisticRegression(penalty="elasticnet", solver="saga")
srch_4 = RandomizedSearchCV(model_4, grid_4, scoring="neg_log_loss", cv=10, random_state=4206969, verbose=99)
srch_4.fit(xtrain, ytrain)
pred_4 = srch_4.predict_proba(xtest)
ll4 = log_loss(ytest, pred_4)
print(ll4)

# Ridge Regression wins
model_final = LogisticRegression(penalty="l2", C=0.0415737798183381)
model_final.fit(X, y)
dump(model_final, "src/development3/model.sav")
