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
model_w, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=750, seed=40, weight=True)
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
model, accs, losses, figs = run_model(data_dict_er, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
data_dict_er = post_data(data_dict_er, model)
p = time_graph(data_dict_er, "f12_retired", pvar="py2", smooth=False, weight=True)
p.show()
p = time_graph(data_dict_er, "f12_retired", pvar="py2", smooth=True, diff=False, weight=True)
p.show()

v="educ"
time_graph_by(data_dict_er, "f12_retired", v, pvar="py2", smooth=False, weight=True, test_train="test").show() 

out_data(data_dict_er,"nn","trans_er_nn")


#------------------------ transitional UR predict --------------------------
data_path = 'data/covid_long.dta'
data_dict_ur = data_setup(data_path, pred="UR", test_size=0.2, samp=1)
model, accs, losses, figs = run_model(data_dict_ur, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
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
model, accs, losses, figs = run_model(data_dict_nr, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
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
model, accs, losses, figs = run_model(data_dict_re, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
data_dict_re = post_data(data_dict_re, model)
p = time_graph(data_dict_re, "f12_employed", pvar="py2", smooth=True, weight=True)
p.show()
p = time_graph(data_dict_re, "f12_employed", pvar="py2", diff=True, smooth=False, weight=True)
p.show()

time_graph_by(data_dict_re, "f12_retired", "educ", pvar="py2", smooth=True, test_train="test").show() 

out_data(data_dict_re,"nn","trans_re_nn")


# -------- RUN ALL! ---------
for pred in ["UR"]: #["ER", "RE", "UR", "RU", "NR", "RN"]: # 
    print(f"{pred} starting")
    data_path = 'data/covid_long.dta'
    data_dict = data_setup(data_path, pred=pred, test_size=0.2, samp=1)
    model, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=500, seed=41, weight=True) 
    data_dict = post_data(data_dict, model)
    #p = time_graph(data_dict, "f12_employed", pvar="py2", smooth=True, weight=True)
    #p.show()
    pred = pred.lower() # lowercase version of pred  for export 
    out_data(data_dict,"nn","trans_"+pred+"_nn")
    print(f"{pred} done")

# do R separately 
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1)
model, accs, losses, figs = run_model(data_dict, 26, 12, lr=0.05, epochs=1100, seed=41, weight=True) 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "retired", pvar="py2", smooth=True, weight=True)
p.show()
out_data(data_dict,"nn","retired_share_nn2")