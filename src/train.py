import sys, os, io, time   
import pandas as pd
import numpy as np 
from numpy import random 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import contextlib

from src.functions import data_setup, data_setup_asec, full_model, full_model, kfold_cv
from src.functions import post_data, time_graph, time_graph_by, out_data, update_best_params

## Structure for each retirement transition 
# Bring in data 
# kfold_cv, saving scores and params 
# run model using best params 
# save results using out_data function 

### Outcomes to do: 
## CPS 
# R, ER, RE, UR, RU, NR, RN 
## ASEC
# R (all, workly, noworkly)

#------------------------ CPS: Cross-section retired predict --------------------------
# load data 
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1, seed=42)

# run kfold_cv to find best params
results_df, best_params = kfold_cv(full_model, data_dict, 5,
                                   {"n_hidden1": [16, 24, 32],
                                    "lr": [0.005, 0.01, 0.05],
                                    "epochs": [1000, 1500]},
                                   {"report_every": 500},
                                   path = "data/cv_cps_R.csv")

# run model using best params
best_params_df = pd.read_csv("data/cv_cps_R.csv")
best_params = best_params_df.loc[best_params_df['score'].idxmax()].to_dict()
for key in ["model_name", "avg_loss","avg_f1","model"]:
    best_params.pop(key)
model, evals = full_model(data_dict, **best_params, seed=42, weight=True, report_every=1)

# save results  
out_data(model, data_dict, "data/pred_cps_R")


#------------------------ CPS: Loop through transitions --------------------------
for trans in ["ER", "RE", "UR", "RU", "NR", "RN"]:
    # load data 
    data_path = 'data/covid_long.dta'
    data_dict = data_setup(data_path, trans, test_size=0.2, samp=1, seed=42)
    
    #kfold_cv to find best params
    path_cv = "data/cv_cps_"+trans+".csv"
    results_df, best_params = kfold_cv(full_model, data_dict, 5,
                                    {"n_hidden1": [16, 24, 32],
                                        "lr": [0.005, 0.01, 0.05],
                                        "epochs": [1000, 1500]},
                                    {"report_every": 500},
                                    path = path_cv)

    # run model using best params
    best_params_df = pd.read_csv(path_cv)
    best_params = best_params_df.loc[best_params_df['score'].idxmax()].to_dict()
    for key in ["model_name", "avg_loss","avg_f1","model"]:
        best_params.pop(key)
    model, evals = full_model(data_dict, **best_params, seed=42, weight=True, report_every=1)

    # save results  
    path_pred = "data/pred_cps_"+trans
    out_data(model, data_dict, path_pred)

#------------------------ ASEC: Cross-section retired predict --------------------------
# load data 
data_path = 'data/asec_data.dta'
data_dict = data_setup(data_path, pred="R",  work_sample="all", test_size=0.2, samp=1, seed=42)

# run kfold_cv to find best params
results_df, best_params = kfold_cv(full_model, data_dict, 5,
                                   {"n_hidden1": [16, 24, 32],
                                    "lr": [0.005, 0.01, 0.05],
                                    "epochs": [500, 1000]},
                                   {"report_every": 500},
                                   path = "data/cv_asec_R.csv")

# run model using best params
best_params_df = pd.read_csv("data/cv_asec_R.csv")
best_params = best_params_df.loc[best_params_df['score'].idxmax()].to_dict()
for key in ["model_name", "avg_loss","avg_f1","model"]:
    best_params.pop(key)
model, evals = full_model(data_dict, **best_params, seed=42, weight=True, report_every=1)