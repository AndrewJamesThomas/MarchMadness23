# March Madness 2023
The 2023 March Madness Model is based on the 2022 Model but will hopefully improve on many of the weaknesses that were
presented by that model. However, the basic structure should be the same. There will be four major components: 1) An ML
model that predicts the likelihood that any team will beat any other team 2) a monte carlo simulation that simulates
the entire tournament based on the ML model 3) an optimization model that selects the best bracket based on those
simulations and 4) some kind of reporting that exposes and tracks these decisions.

## Data
The data is primarily sourced from the NCAA Tournament Kaggle Competition. While the 2023 competition is not live at this
time, we can still use the previous year's data to build our model.

The 2022 link can be found here: https://www.kaggle.com/competitions/mens-march-mania-2022

## Part I: Predictive Model
The predictive model is a fairly straightforward ML model built with Scikit-Learn. We will try a series of different
options (ie Regularized Regression, Random Forrest, Etc.) and optimize for log-loss. The more difficult aspect will focus
on the extraction of relevant features. These features fall into three broad categories: Team PageRanks, 
SEM Adjusted Aggregate Stats, and Expert Ratings.

### Feature Set a) PageRanks

### Feature Set b) SEM Adj. Stats

### Feature Set C) Expert Ratings

## Part II: Monte Carlo Simulation

## Part III: Decision Model

## Part IV: Reporting

# Results and Reflections:
TBD