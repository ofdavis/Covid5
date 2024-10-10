import sys, os 
import pandas as pd
import numpy as np 
from numpy import random 
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import torch
from torch import nn
import contextlib
import io 
from patsy import dmatrices, dmatrix, demo_data 

from src.functions import run_model, data_setup, post_data, time_graph, time_graph_by, out_data 

#------------------------ cross-section retired predict --------------------------
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1)
model_u, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=500, seed=40, weight=False) 
model_w, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=500, seed=40, weight=True) 
data_dict_u = post_data(data_dict, model_u, weight=False)
data_dict_u["data"].py2.head()
data_dict_w = post_data(data_dict, model_w, weight=True)
data_dict_w["data"].py2.head()
data_dict_u["data"].py2.head()
data_dict_w["data"].py2==data_dict_u["data"].py2

p = time_graph(data_dict_u, "retired", pvar="py2", smooth=False)
p.show()
p = time_graph(data_dict_w, "retired", pvar="py2", smooth=True, weight=True)
p.show()
p = time_graph(data_dict_u, "retired", pvar="py2", smooth=True)
p.show()


test_data = data_dict["data"][data_dict["data"]["data_type"] == "test"]
coll_test = test_data.groupby("mo", as_index=False).apply(lambda x: pd.Series({
            "retired":   np.average(x["retired"],   weights=x['wtfinl']),
            "py2":       np.average(x["py2"],       weights=x['wtfinl']) }))

out_data(data_dict,"_nn","retired_share_nn")


#------------------------ transitional E-R predict --------------------------
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="ER", test_size=0.5, samp=1)
model, accs, losses, figs = run_model(data_dict, 26, 13, lr=0.05, epochs=250, seed=41) 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "f_retired", pvar="py2", smooth=True)
p.show()
p = time_graph(data_dict, "f_retired", pvar="py2", smooth=False)
p.show()

time_graph_by(data_dict, "f_retired", "month", pvar="py2", smooth=True, test_train="test").show() 
time_graph_by(data_dict, "f_retired", "mish", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "sex", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "vet", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "nativity", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffrem", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffphys", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffmob", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "race", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "married", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "agegrp_sp", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_any", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_yng", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_adt", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "educ", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "metro", pvar="py2", smooth=True, test_train="test").show()



#------------------------ transitional UN-R predict --------------------------
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="UNR", test_size=0.5, samp=1)
model, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=250, seed=41) 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "f_retired", pvar="py2", smooth=True)
p.show()
p = time_graph(data_dict, "f_retired", pvar="py2", smooth=False)
p.show()

time_graph_by(data_dict, "f_retired", "month", pvar="py2", smooth=True, test_train="test").show() 
time_graph_by(data_dict, "f_retired", "mish", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "sex", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "vet", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "nativity", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffrem", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffphys", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "diffmob", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "race", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "married", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "agegrp_sp", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_any", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_yng", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "child_adt", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "educ", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "f_retired", "metro", pvar="py2", smooth=True, test_train="test").show()

time_graph_by(data_dict, "f_retired", "nlf_oth", pvar="py2", smooth=True, test_train="test").show()