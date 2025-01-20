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
from torch.utils.data import DataLoader, TensorDataset
import contextlib

from src.functions import run_model, data_setup, post_data, time_graph, time_graph_by, out_data, data_setup_asec, kfold_cv, update_best_params

#------------------------ cross-section retired predict --------------------------
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1)
model_w, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=500, seed=40, weight=True)
data_dict_w = post_data(data_dict, model_w, weight=True)

time_graph(data_dict_w, "retired", pvar="py2", smooth=True, weight=True).show()
time_graph(data_dict_w, "retired", pvar="py2", smooth=False, weight=True).show()

time_graph_by(data_dict_w, "retired", "diffphys", pvar="py2", smooth=True, test_train="test", weight=True).show()

out_data(data_dict_w,"nn","retired_share_nn")

for col in data_dict["Xdf"].columns: # determine if any missings in data
    print(col, data_dict["Xdf"][col].isna().sum())


#------------------------ transitional E-R predict --------------------------
data_path = 'data/covid_long.dta'
data_dict_er = data_setup(data_path, pred="ER", test_size=0.2, samp=1)
model, evals = run_model(data_dict_er, 24, 12, lr=0.05, epochs=500, seed=41, weight=True, report_every=10) 
data_dict_er = post_data(data_dict_er, model)
p = time_graph(data_dict_er, "f12_retired", pvar="py2", smooth=False, weight="wtfinl")
p.show()
p = time_graph(data_dict_er, "f12_retired", pvar="py2", smooth=True, diff=False, weight="wtfinl")
p.show()

v="occ_maj"
time_graph_by(data_dict_er, "f12_retired", v, pvar="py2", smooth=True,
              weight="wtfinl", test_train="test").show() 

out_data(data_dict_er,"nn","trans_er_nn")


#------------------------ transitional UR predict --------------------------
data_path = 'data/covid_long.dta'
data_dict_ur = data_setup(data_path, pred="UR", test_size=0.2, samp=1)
model, evals = run_model(data_dict_ur, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
data_dict_ur = post_data(data_dict_ur, model)
p = time_graph(data_dict_ur, "f12_retired", pvar="py2", smooth=False, weight=True)
p.show()
p = time_graph(data_dict_ur, "f12_retired", pvar="py2", smooth=True, weight=True, diff=False)
p.show()

time_graph_by(data_dict_ur, "f12_retired", "educ", pvar="py2", smooth=True, test_train="test").show() 

out_data(data_dict_ur,"nn","trans_ur_nn")


#------------------------ transitional NR predict --------------------------
data_path = 'data/covid_long.dta'
data_dict_nr = data_setup(data_path, pred="NR", test_size=0.2, samp=1)
model, evals = run_model(data_dict_nr, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
data_dict_nr = post_data(data_dict_nr, model)
p = time_graph(data_dict_nr, "f12_retired", pvar="py2", smooth=False, weight=True)
p.show()
p = time_graph(data_dict_nr, "f12_retired", pvar="py2", smooth=True, weight=True, diff=False)
p.show()

time_graph_by(data_dict_nr, "f12_retired", "educ", pvar="py2", smooth=True, test_train="test").show() 

out_data(data_dict_nr,"nn","trans_nr_nn")

#------------------------ transitional RE predict --------------------------
data_path = 'data/covid_long.dta'
data_dict_re = data_setup(data_path, pred="RE", test_size=0.2, samp=1)
model, evals = run_model(data_dict_re, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
data_dict_re = post_data(data_dict_re, model)
p = time_graph(data_dict_re, "f12_employed", pvar="py2", smooth=True, weight="wtfinl")
p.show()

time_graph_by(data_dict_re, "f12_employed", "educ", pvar="py2", smooth=True, test_train="test").show() 

out_data(data_dict_re,"nn","trans_re_nn")


# -------- RUN ALL! ---------
for pred in ["UR"]: #["ER", "RE", "UR", "RU", "NR", "RN"]: # 
    print(f"{pred} starting")
    data_path = 'data/covid_long.dta'
    data_dict = data_setup(data_path, pred=pred, test_size=0.2, samp=1)
    model, evals = run_model(data_dict, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
    data_dict = post_data(data_dict, model)
    #p = time_graph(data_dict, "f12_employed", pvar="py2", smooth=True, weight=True)
    #p.show()
    pred = pred.lower() # lowercase version of pred  for export 
    out_data(data_dict,"nn","trans_"+pred+"_nn")
    print(f"{pred} done")

# do R separately 
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1)
model, evals = run_model(data_dict, 26, 12, lr=0.05, epochs=1100, seed=41, weight=True) 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "retired", pvar="py2", smooth=True, weight="wtfinl")
p.show()
out_data(data_dict,"nn","retired_share_nn2")

start_time = time.time()
model, evals = run_model(data_dict, 64, 32, lr=0.05, epochs=100, seed=41, weight=True, report_every=1) 
end_time = time.time()
time1 = end_time - start_time
print(time1)
evals 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "retired", pvar="py2", smooth=True, weight="wtfinl")
p.show()


start_time = time.time()
model2, evals2 = run_model2(data_dict, 32, 16, lr=0.005, epochs=10, seed=41, weight=True, report_every=1) 
end_time = time.time()
time2 = end_time - start_time
print(time2)
evals2

data_dict = post_data(data_dict, model2)
time_graph(data_dict, "retired", pvar="py2", smooth=True, weight="wtfinl").show()
time_graph_by(data_dict, "retired", "educ", pvar="py2", smooth=True, test_train="train", weight="wtfinl").show()
p.show()

# ---------- asec data ------------
# all workers
data_path = 'data/asec_data.dta'
data_dict = data_setup_asec(data_path, pred="R", work_sample="all", test_size=0.2, samp=1)
results_df, model_results, best_params = kfold_cv(data_dict, 5, [400,600,800], [0.01, 0.05], [16, 24, 32])
update_best_params("data/generated/best_params.csv", best_params, "asec_all_R")

# workly workers
data_path = 'data/asec_data.dta'
data_dict = data_setup_asec(data_path, pred="R", work_sample="workly", test_size=0.2, samp=1)
results_df, model_results, best_params = kfold_cv(data_dict, 5, [400,600,800], [0.01, 0.05], [16, 24, 32])
update_best_params("data/generated/best_params.csv", best_params, "asec_workly_R")

# noworkly workers
data_path = 'data/asec_data.dta'
data_dict = data_setup_asec(data_path, pred="R", work_sample="noworkly", test_size=0.2, samp=1)
results_df, model_results, best_params = kfold_cv(data_dict, 5, [400,600,800], [0.01, 0.05], [16, 24, 32])
update_best_params("data/generated/best_params.csv", best_params, "asec_noworkly_R")
best_params


model_w, evals = run_model(data_dict, 
                           int(best_params["neurons"]), 
                           int(best_params["neurons"]*0.6), 
                           lr=best_params["lr"], 
                           epochs=int(best_params["epochs"]), seed=43, weight=True)

data_dict_w = post_data(data_dict, model_w, weight=True)
time_graph(data_dict_w, "retired", pvar="py2", smooth=False, weight="asecwt").show()
time_graph_by(data_dict_w, "retired", "sex", pvar="py2", test_train="test", smooth=False, weight="asecwt").show()



num_cores = os.cpu_count()
num_cores






