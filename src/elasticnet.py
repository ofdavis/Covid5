import sys 
import pandas as pd
import numpy as np 
from numpy import random 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from patsy import dmatrices, dmatrix, demo_data 
import time, os 

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

# program for bringing in and formatting data 
def data_setup_sk(path, test_size, samp):
    # data load 
    data = pd.read_stata(path,convert_dates=True,convert_categoricals=False).sample(frac=samp)

    # fix mo 
    data["modate"]= pd.DataFrame({"modate" : (data.year - 2010)*12 + data.month.astype("float")}) # months since 2009m12

    """
    categorical: 'year', 'month', 'mish', 'statefip', 'race', 'educ', 'agegrp_sp'
    binary: 'sex', 'vet', 'married', 'metro', 'retired', 'diffphys', 'diffmob', 'ssa', 'covid'
    continuous: 'modate', 'age', 'agesq', 'agecub', 'pia', 'ur', 'urhat', 
    other/util: 'wtfinl', 'cpsidp', 

    interactions: sex:(educ + married + race + agegrp_sp + age), race(educ + married + age),  educ(married + age + pia),     
    """
    datax = dmatrix(
        "-1 + C(month) + C(mish) + C(statefip) + C(race) + C(educ) + C(agegrp_sp) + " + 
        "sex + vet + married + metro + diffmob + diffphys + " + 
        "cpsidp + wtfinl + covid + retired + " + 
        "modate + age + agesq + agecub + pia + ssa:pia + urhat + " + 
        "C(sex):(C(educ) + C(race) + C(agegrp_sp) + married + age) + " + 
        "C(race):(C(educ) + married + age) + " + 
        "C(educ):(married + age + pia)",
        data, return_type="dataframe"
    )

    # set up pre and post X and y 
    Xdf_pre  = datax[datax.covid==0]
    Xdf_post = datax[datax.covid==1]
    ydf_pre =  datax[datax.covid==0].retired
    ydf_post = datax[datax.covid==1].retired

    # split into test and train -- only need to spit pre 
    Xdf_train, Xdf_test, ydf_train, ydf_test = train_test_split(
        Xdf_pre, 
        ydf_pre, 
        test_size=test_size, # 20% test, 80% train
        random_state=42) # make the random split reproducible 
    
    # assign data groups (test/train/post) to main data 
    data["data_type"] = np.nan
    data.loc[data.index.isin(Xdf_train.index), "data_type"] = "train"
    data.loc[data.index.isin(Xdf_test.index), "data_type"] = "test"
    data.loc[data.index.isin(Xdf_post.index), "data_type"] = "post"

    # create weight arrays 
    wt_train = Xdf_train["wtfinl"].to_numpy(dtype=float).copy()
    wt_test  = Xdf_test["wtfinl"].to_numpy(dtype=float).copy()
    wt_post  = Xdf_post["wtfinl"].to_numpy(dtype=float).copy()

    # drop unneeded columns 
    Xdf_train = Xdf_train.drop(["cpsidp","covid","retired","wtfinl"], axis=1)
    Xdf_test  =  Xdf_test.drop(["cpsidp","covid","retired","wtfinl"], axis=1)
    Xdf_post  =  Xdf_post.drop(["cpsidp","covid","retired","wtfinl"], axis=1)

    # standardize variables 
    sc = StandardScaler()
    for i in range(Xdf_train.shape[1]): # test if binary (even tho all floats)
        if (((Xdf_train.iloc[:,i]==0) | (Xdf_train.iloc[:,i]==1)).all()==False):  
            print(f"{Xdf_train.columns[i]} is not binary")
            Xdf_train.iloc[:,i] = sc.fit_transform(Xdf_train.iloc[:,i].to_numpy().reshape(-1,1), y=None)
            Xdf_test.iloc[:,i]  = sc.transform(Xdf_test.iloc[:,i].to_numpy().reshape(-1,1))
            Xdf_post.iloc[:,i]  = sc.transform(Xdf_post.iloc[:,i].to_numpy().reshape(-1,1))

    # preallocate numpy array for X_train, X_test, y_train, y_test 
    Xnp_train = Xdf_train.to_numpy(dtype=float).copy()
    Xnp_test  = Xdf_test.to_numpy(dtype=float).copy()
    Xnp_post  = Xdf_post.to_numpy(dtype=float).copy()
    ynp_train = ydf_train.to_numpy(dtype=float).copy()
    ynp_test  = ydf_test.to_numpy(dtype=float).copy()
    ynp_post  = ydf_post.to_numpy(dtype=float).copy()

    data_dict = {
        "Xdf_train": Xdf_train,
        "Xdf_test": Xdf_test,
        "Xdf_post": Xdf_post,
        "Xnp_train": Xnp_train,
        "Xnp_test": Xnp_test,
        "Xnp_post": Xnp_post,
        "ydf_train": ydf_train,
        "ydf_test": ydf_test,
        "ydf_post": ydf_post,
        "ynp_train": ynp_train,
        "ynp_test": ynp_test,
        "ynp_post": ynp_post,
        "wt_train": wt_train,
        "wt_test": wt_test,
        "wt_post": wt_post,
        "data": data
    }
    return data_dict


def run_logistic(data_dict_in, model_dict, model):
    data_dict_out = data_dict_in.copy() 

    # run model 
    tic = time.perf_counter()
    model_dict[model].fit(data_dict_out["Xnp_train"], data_dict_out["ynp_train"], sample_weight=data_dict_out["wt_train"])
    toc = time.perf_counter()
    print(f"{model} time: ", toc-tic)

    # get predictions 
    p_test   = model_dict[model].predict_proba(data_dict_out["Xnp_test"])[:,1]
    p_train  = model_dict[model].predict_proba(data_dict_out["Xnp_train"])[:,1]
    p_post   = model_dict[model].predict_proba(data_dict_out["Xnp_post"])[:,1]

    # append to Xs 
    colname = "py_" + model
    X_test_tmp = data_dict_out["Xdf_test"].copy()
    X_test_tmp[colname] = p_test
    X_test_tmp = X_test_tmp[[colname]]

    X_train_tmp = data_dict_out["Xdf_train"].copy()
    X_train_tmp[colname] = p_train
    X_train_tmp = X_train_tmp[[colname]]

    X_post_tmp = data_dict_out["Xdf_post"].copy()
    X_post_tmp[colname] = p_post
    X_post_tmp = X_post_tmp[[colname]]

    # append temps 
    X_temp = pd.concat([X_test_tmp,X_train_tmp,X_post_tmp])
    data_dict_out["data"] = data_dict_out["data"].join(X_temp, how="left")

    return data_dict_out


def out_data_sk(data_dict_in, suffix, filename):
    # export cpsidp, mo, data_type, py, py2 to stata 
    outvar = "py_" + suffix
    data_out =data_dict_in["data"][["cpsidp", "mo", "data_type", outvar]]
    data_out.to_stata(f"data/generated/{filename}.dta", convert_dates={'mo':'%tm'})
    print("Data exported to " + f"data/generated/{filename}.dta")



# Define models 
num_cores = os.cpu_count()
mdict = {
    "lr_1sa" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='saga', penalty="l1"),
    "lr_1ll" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='liblinear', penalty="l1"),
    "lr_2sa" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='sag', penalty="l2"),
    "lr_2lb" : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='lbfgs', penalty="l2"),
    "lr_e"   : LogisticRegressionCV(cv=3, n_jobs=num_cores, solver='saga', penalty='elasticnet', l1_ratios=[0.1, 0.5, 0.9],max_iter=1000)
}


data_dict_sk = data_setup_sk("data/covid_long.dta", test_size=0.5 , samp=1)
data_dict_sk = run_logistic(data_dict_sk, mdict, "lr_2lb")

p_lr_2lb = coll_plot(data_dict_sk["data"][(data_dict_sk["data"].data_type=="train") | (data_dict_sk["data"].data_type=="post")], "mo", ["retired","py_lr_2lb"])
p_lr_2lb.show()

out_data_sk(data_dict_sk, "lr_2lb", "retired_share_ridge")



data_dict_e = data_setup_sk("data/covid_long.dta", test_size=0.5 , samp=1)
data_dict_e = run_logistic(data_dict_e, mdict, "lr_e")

p_lr_2lb = coll_plot(data_dict_e["data"][(data_dict_e["data"].data_type=="train") | (data_dict_e["data"].data_type=="post")], "mo", ["retired","py_lr_e"])
p_lr_2lb.show()

out_data_sk(data_dict_e, "lr_e", "retired_share_elast")
