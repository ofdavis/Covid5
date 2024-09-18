import sys 
import pandas as pd
import numpy as np 
from numpy import random 
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
import contextlib
import io 
from patsy import dmatrices, dmatrix, demo_data 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import time 
import os 

# collapse plot 
def coll_plot(df, byvar: str, ylist: list):
    vars = ylist.copy()
    vars.append(byvar)
    df = df[vars]
    df = df.groupby(df[byvar], as_index=False).mean()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for y in ylist: 
        ax.plot(df[byvar],df[y], label=y)
    ax.legend()
    return fig 

# collapse plot by category 
def coll_plot_split(df, svar, byvar: str, ylist: list):
    vars = ylist.copy()
    vars.append(byvar)
    vars.append(svar)
    df = df[vars]
    df = df.groupby([byvar,svar], as_index=False).mean()

    # set up subplots 
    svals = data[svar].unique().tolist()
    svals.sort()
    numplot = len(svals)
    #        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    gridr = [1, 1, 1, 2, 2, 2, 3, 3, 3,4, 4, 4, 4, 4, 4, 4]
    gridc = [1, 2, 3, 2, 3, 3, 3, 3, 3,3, 3, 3, 4, 4, 4, 4]
    fig, axs = plt.subplots(gridr[numplot-1], gridc[numplot-1], 
                            figsize=(3*gridc[numplot-1], 3*gridr[numplot-1]))

    # run through ylist 
    for i in range(numplot):
        for y in ylist: 
            axs.flatten()[i].plot(df[df[svar]==svals[i]][byvar],df[df[svar]==svals[i]][y], label=y)
            axs.flatten()[i].set_title(f"{svar}: {svals[i]}")
    plt.tight_layout()
    return fig 

# function for post-testing
def post_data_lrcv(xtest,xtrain,xpost,ytest,ytrain,ypost,model,xcols,round=False): 
    # get predictions 
    if round==True:
        p_test   = model.predict(xtest)
        p_train  = model.predict(xtrain)
        p_post   = model.predict(xpost)
    else: 
        p_test   = model.predict_proba(xtest)[:,1]
        p_train  = model.predict_proba(xtrain)[:,1]
        p_post   = model.predict_proba(xpost)[:,1]

    # combine pred (test & post, train & post)
    p_test_post =  pd.DataFrame({"py" : np.append(p_test,  p_post) })
    p_train_post = pd.DataFrame({"py" : np.append(p_train, p_post) })

    # combine y (test & post, train & post)
    y_test_post  = pd.DataFrame({"y" : pd.concat([ytest,  ypost],ignore_index=True) })
    y_train_post = pd.DataFrame({"y" : pd.concat([ytrain, ypost],ignore_index=True) })

    # create full data frames 
    data_test_post  = pd.concat([xtest,  xpost],ignore_index=True)
    data_train_post = pd.concat([xtrain, xpost],ignore_index=True)
    data_test_post.columns  = xcols 
    data_train_post.columns = xcols 

    # add predicted and actual y to data frames 
    data_test_post['py']  = p_test_post
    data_train_post['py'] = p_train_post
    data_test_post['y']   = y_test_post
    data_train_post['y']  = y_train_post

    return data_test_post, data_train_post


# collapse test/train/predicted data by byvar and plot 
def coll_graph(df_test_post, df_train_post, byvar):
    coll_test_post = df_test_post.groupby(byvar, as_index=False).agg({
        'py': 'mean', 
        'y': 'mean'
    })

    coll_train_post = df_train_post.groupby(byvar, as_index=False).agg({
        'py': 'mean',
        'y': 'mean'  
    })

    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    
    # Plot the first column vs the second column in the first subplot
    axs[0].plot(coll_test_post[byvar], coll_test_post["py"])
    axs[0].plot(coll_test_post[byvar], coll_test_post["y"])
    axs[0].set_title("Test data: Prediction vs actual")
    axs[0].set_xlabel(byvar)
    
    # Plot the second column vs the third column in the second subplot
    axs[1].plot(coll_train_post[byvar], coll_train_post["py"])
    axs[1].plot(coll_train_post[byvar], coll_train_post["y"])
    axs[1].set_title("Train data: Prediction vs actual")
    axs[1].set_xlabel(byvar)

    # Plot the third column vs the first column in the third subplot
    axs[2].plot(coll_test_post[byvar],  coll_test_post["py"])
    axs[2].plot(coll_train_post[byvar], coll_train_post["py"])
    axs[2].set_title("Test predictions vs train predictions")
    axs[2].set_xlabel(byvar)
    
    # Adjust layout
    plt.tight_layout()
    
    # Return the figure
    return fig

"""" ------------------------------------------------------------------------------------------------
                                       main code 
------------------------------------------------------------------------------------------------"""
# data load 
samp = 1
data = pd.read_csv('data/covid_ml.csv').sample(frac=samp)
cols = data.columns

# fix mo 
data.mo     = pd.DataFrame({"mo" : (data.year - 2010)*12 + data.month.astype("float")}) # months since 2009m12

"""
categorial: 'year', 'month', 'mish', 'statefip', 'race', 'educ', 'agegrp_sp'
binary: 'sex', 'vet', 'married', 'metro', 'retired', 'disable', 'ssa', 'covid'
continuous: 'mo', 'age', 'agesq', 'agecub', 'pia', 'ur', 'urhat', 
other/util: 'wtfinl', 'cpsidp', 

interactions: sex:(educ + married + race + agegrp_sp + age), race(educ + married + age),  educ(married + age + pia),     
"""

datax = dmatrix(
    "-1 + C(month) + C(mish) + C(statefip) + C(race) + C(educ) + C(agegrp_sp) + " + 
    "sex + vet + married + metro + disable + " + 
    "cpsidp + covid + retired + " + 
    "mo + age + agesq + agecub + pia + ssa:pia + " + 
    "C(sex):(C(educ) + C(race) + C(agegrp_sp) + married + age) + " + 
    "C(race):(C(educ) + married + age) + " + 
    "C(educ):(married + age + pia)",
    data, return_type="dataframe"
)
datax.mo.head(10) 
datax.columns

# set up pre and post X and y 
X_pre  = datax[datax.covid==0].drop(columns=["cpsidp","retired","covid"])
X_post = datax[datax.covid==1].drop(columns=["cpsidp","retired","covid"])
y_pre =  datax[datax.covid==0].retired
y_post = datax[datax.covid==1].retired

# split into test and train -- only need to spit pre 
X_train, X_test, y_train, y_test = train_test_split(
    X_pre, 
    y_pre, 
    test_size=0.2, # 20% test, 80% train
    random_state=42) # make the random split reproducible 

# assign data groups (test/train/post) to main data 
data["train"] = data.index.isin(X_train.index)
data["test"] = data.index.isin(X_test.index)
data["post"] = data.index.isin(X_post.index)

# standardize variables 
sc = StandardScaler()
for i in range(X_pre.shape[1]): # test if binary (even tho all floats)
    if ((X_train.iloc[:,i]==0) | (X_train.iloc[:,i]==1)).all()==False: 
        print(f"{X_pre.columns[i]} is not binary")
        X_train.iloc[:,i] = sc.fit_transform(X_train.iloc[:,i].to_numpy().reshape(-1,1), y=None)
        X_test.iloc[:,i]  = sc.transform(X_test.iloc[:,i].to_numpy().reshape(-1,1))
        X_post.iloc[:,i]  = sc.transform(X_post.iloc[:,i].to_numpy().reshape(-1,1))

# Define models 
num_cores = os.cpu_count()
mdict = {
    "lr_1sa" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='saga', penalty="l1"),
    "lr_1ll" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='liblinear', penalty="l1"),
    "lr_2sa" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='saga', penalty="l2"),
    "lr_2lb" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='lbfgs', penalty="l2"),
    "lr_e"   : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='saga', penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9],max_iter=1000)
}

# Train the models
times = []
for model in ["lr_1sa","lr_1ll","lr_2sa","lr_2lb","lr_e"]: #
    tic = time.perf_counter()
    mdict[model].fit(X_train, y_train)
    toc = time.perf_counter() 
    times.append(toc-tic)
    print(model, " time: ", toc-tic)

    # get predictions 
    p_test   = mdict[model].predict_proba(X_test)[:,1]
    p_train  = mdict[model].predict_proba(X_train)[:,1]
    p_post   = mdict[model].predict_proba(X_post)[:,1]

    # append to Xs 
    colname = "py_" + model
    X_test_tmp = X_test.copy()
    X_test_tmp[colname] = p_test
    X_test_tmp = X_test_tmp[[colname]]

    X_train_tmp = X_train.copy()
    X_train_tmp[colname] = p_train
    X_train_tmp = X_train_tmp[[colname]]

    X_post_tmp = X_post.copy()
    X_post_tmp[colname] = p_post
    X_post_tmp = X_post_tmp[[colname]]

    # append temps 
    X_temp = pd.concat([X_test_tmp,X_train_tmp,X_post_tmp])
    data = data.join(X_temp, how="left")
times 

# generate plots

p_lr_1sa = coll_plot(data[(data.train==0)], "mo", ["retired","py_lr_1sa","py_lr_1ll","py_lr_2sa","py_lr_2lb"])
p_lr_1sa.show()

p_lr_1ll = coll_plot(data[(data.train==0) & (data.sex==1)], "mo", ["retired","py_lr_1ll"])
p_lr_1ll.show()

p_lr_2sa = coll_plot(data[(data.train==0) & (data.sex==1)], "mo", ["retired","py_lr_2lb"])
p_lr_2sa.show()

p_lr_2lb = coll_plot(data[(data.train==0) & (data.sex==1)], "mo", ["retired","py_lr_2lb"])
p_lr_2lb.show()

p_lr_e   = coll_plot(data[(data.train==0) & (data.sex==1)], "mo", ["retired","py_lr_e"])
p_lr_e.show()

#p_lr_e   = coll_plot(data[(data.train==0) & (data.sex==1)], "mo", ["retired","py_lr_e"])
#p_lr_e.show()

pp = coll_plot_split(data,"race","mo",["retired","py_lr_2lb"])
pp.show()

# Test the models
#print("Logistic Regression: {} || Elasticnet: {}".format(logreg_0.score(X_test, y_test), logreg_0.score(X_test, y_test)))

# Print out some more metrics
#print(classification_report(y_test, logreg_0.predict(X_test)))
#print(classification_report(y_test, logreg_e.predict(X_test)))



