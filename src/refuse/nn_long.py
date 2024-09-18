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

from src.functions import run_model, data_setup, post_data, time_graph, time_graph_by

#------------------------ run --------------------------
data_path = 'data/covid_long.dta'
data_dict = data_setup(data_path, pred="R", test_size=0.2, samp=1)
model, accs, losses, figs = run_model(data_dict, 24, 12, lr=0.05, epochs=1000, seed=41) 
data_dict = post_data(data_dict, model)
p = time_graph(data_dict, "retired", pvar="py2", smooth=True)
p.show()
p = time_graph(data_dict, "retired", pvar="py2", smooth=False)
p.show()

time_graph_by(data_dict, "retired", "month", pvar="py2", smooth=True, test_train="test").show() 
time_graph_by(data_dict, "retired", "mish", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "sex", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "vet", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "nativity", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "diffrem", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "diffphys", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "diffmob", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "race", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "married", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "agegrp_sp", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "child_any", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "child_yng", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "child_adt", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "educ", pvar="py2", smooth=True, test_train="test").show()
time_graph_by(data_dict, "retired", "metro", pvar="py2", smooth=True, test_train="test").show()