import sys, os, io, time, itertools, contextlib, gc
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
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, TensorDataset

# from src.define_class import CustomDataset  # Import the CustomDataset class

# data load
def data_setup(path, pred, test_size=0.5, samp=1):
    if pred not in ["ER", "UR", "NR", "RE", "RU", "RN", "R"]:
        raise ValueError("pred must be ER, UR, NR, RE, RU, RN or R")

    data_in = pd.read_stata(path,convert_dates=True,convert_categoricals=False).sample(frac=samp)

    # for transitions (two-char pred), restrict sample
    if len(pred)==2:
        data_in.mo = data_in.f12_mo  # recast mo as outcome mo (f12) if transition
        if pred[0]=="E":
            data_in = data_in[(data_in.mish<=4) & (data_in.employed==1) & (data_in.wtf12.notna())]
        if pred[0]=="U":
            data_in = data_in[(data_in.mish<=4) & (data_in.unem==1) & (data_in.wtf12.notna())]
        if pred[0]=="N":
            data_in = data_in[(data_in.mish<=4) & (data_in.nlf==1) & (data_in.wtf12.notna())]
        if pred[0]=="R":
            data_in = data_in[(data_in.mish<=4) & (data_in.retired==1) & (data_in.wtf12.notna())]

    if pred=="R":
        data_in = data_in

    # data setup (one-hot encoding etc)
    mish     = pd.get_dummies(data_in['mish'], prefix='month')
    month    = pd.get_dummies(data_in['month'], prefix='month')
    state    = pd.get_dummies(data_in['statefip'], prefix='state')
    race     = pd.get_dummies(data_in['race'], prefix='race')
    nativity = pd.get_dummies(data_in['nativity'], prefix='nativity')
    educ     = pd.get_dummies(data_in['educ'], prefix='educ')
    agesp    = pd.get_dummies(data_in['agegrp_sp'], prefix='agesp')
    ind_maj  = pd.get_dummies(data_in['ind_maj'], prefix='ind_maj', dummy_na=True)
    occ_maj  = pd.get_dummies(data_in['occ_maj'], prefix='occ_maj', dummy_na=True)
    sex      = data_in['sex'].astype("bool")
    covid    = data_in['covid'].astype("bool")
    f12_covid = data_in['f12_covid'].astype("bool")
    marr     = data_in['married'].astype("bool")
    ssa      = data_in['ssa'].astype("bool")
    metro    = data_in['metro'].astype("bool")
    vet      = data_in['vet'].astype("bool")
    #diffmob  = data_in['diffmob'].astype("bool")
    #diffrem  = data_in['diffrem'].astype("bool")
    #diffphys = data_in['diffphys'].astype("bool")
    #unem     = data_in['unem'].astype("bool")
    untemp   = data_in['untemp'].astype("bool")
    unlose   = data_in['unlose'].astype("bool")
    unable   = data_in['unable'].astype("bool")
    nlf_oth  = data_in['nlf_oth'].astype("bool")
    self     = data_in['self'].astype("bool")
    govt     = data_in['govt'].astype("bool")
    ft       = data_in['ft'].astype("bool")
    absnt    = data_in['absnt'].astype("bool")
    child_any= data_in['child_any'].astype("bool")
    child_yng= data_in['child_yng'].astype("bool")
    child_adt= data_in['child_adt'].astype("bool")
    age      = data_in.age.astype("float")
    agesq    = data_in.agesq.astype("float")
    agecub   = data_in.agecub.astype("float")
    dur      = data_in.dur.astype("float")
    #wage     = data.wage.astype("float")
    #wageflag = data['wageflag'].astype("bool")
    pia      = data_in.pia.astype("float")
    urhat    = data_in.urhat.astype("float")
    ssapia   = pd.DataFrame({"ssapia" :(data_in.pia * data_in.ssa).astype("float")})
    mo       = pd.DataFrame({"mo" : (data_in.year - 2010)*12 + data_in.month.astype("float")}) # months since 2009m12

    # create x and y dataframes, pre and post, X and y; for each of the two transitions and the retired outcome
    if pred[0]=="E":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, data_in.wtf12, mish, mo, month, state, race, nativity, sex, educ, covid, f12_covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, age, agesq, agecub, pia, ssapia, urhat,
                            ind_maj, occ_maj, govt, ft, absnt, self], axis=1)
    if pred[0]=="U":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, data_in.wtf12, mish, mo, month, state, race, nativity, sex, educ, covid, f12_covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, age, agesq, agecub, pia, ssapia, urhat,
                            untemp, unlose, dur], axis=1)
    if pred[0]=="N":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, data_in.wtf12, mish, mo, month, state, race, nativity, sex, educ, covid, f12_covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, age, agesq, agecub, pia, ssapia, urhat,
                            unable, nlf_oth], axis=1)
    if pred[0]=="R":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, data_in.wtf12, mish, mo, month, state, race, nativity, sex, educ, covid, f12_covid, marr, agesp,
                            ssa, metro, child_any, child_yng, child_adt, vet, age, agesq, agecub, pia, ssapia, urhat], axis=1)
    if pred=="R": # drop uneeded cols if just R predict
        Xdf_out = Xdf_out.drop(["wtf12", "f12_covid"], axis=1)

    # for transitions (two-char pred), define pre/post dataframes
    if len(pred)==2:
        Xdf_pre_out  = Xdf_out[Xdf_out.f12_covid==0]
        Xdf_post_out = Xdf_out[Xdf_out.f12_covid==1]

        #define transition outcomes
        if pred[1]=="R":
            ydf_pre_out =  data_in[data_in.f12_covid==0].f12_retired
            ydf_post_out = data_in[data_in.f12_covid==1].f12_retired
        if pred[1]=="E":
            ydf_pre_out =  data_in[data_in.f12_covid==0].f12_employed
            ydf_post_out = data_in[data_in.f12_covid==1].f12_employed
        if pred[1]=="U":
            ydf_pre_out =  data_in[data_in.f12_covid==0].f12_unem
            ydf_post_out = data_in[data_in.f12_covid==1].f12_unem
        if pred[1]=="N":
            ydf_pre_out =  data_in[data_in.f12_covid==0].f12_nlf
            ydf_post_out = data_in[data_in.f12_covid==1].f12_nlf

    if pred=="R":
        Xdf_pre_out  = Xdf_out[Xdf_out.covid==0]
        Xdf_post_out = Xdf_out[Xdf_out.covid==1]
        ydf_pre_out =  data_in[data_in.covid==0].retired
        ydf_post_out = data_in[data_in.covid==1].retired

    if np.sum(Xdf_out.isnull().sum())>0:
        print("There are missing values in the data")

    # split into test and train -- only need to spit pre
    Xdf_train_out, Xdf_test_out, ydf_train_out, ydf_test_out = train_test_split(
        Xdf_pre_out,
        ydf_pre_out,
        test_size=test_size,
        shuffle=False,
        random_state=1) # make the random split reproducible

    # standardize variables
    sc = StandardScaler()
    for i in range(Xdf_pre_out.shape[1]):
        if (Xdf_train_out.iloc[:,i].dtype!="bool") & (Xdf_train_out.iloc[:,i].name not in ["cpsidp","wtfinl","wtf12"]):
            #print(f"{i} is not binary")
            Xdf_train_out.iloc[:,i] = sc.fit_transform(Xdf_train_out.iloc[:,i].to_numpy().reshape(-1,1), y=None)
            Xdf_test_out.iloc[:,i]  = sc.transform(Xdf_test_out.iloc[:,i].to_numpy().reshape(-1,1))
            Xdf_post_out.iloc[:,i]  = sc.transform(Xdf_post_out.iloc[:,i].to_numpy().reshape(-1,1))

    # turn data to tensors
    if (pred=="R"):
        wtvar   ="wtfinl"
        exclude_cols = ["wtfinl", "cpsidp", "covid"]
    else:
        wtvar   ="wtf12"
        exclude_cols = ["wtfinl", "wtf12", "cpsidp", "covid", "f12_covid"]

    Xtn_train_out = torch.tensor(Xdf_train_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_test_out  = torch.tensor( Xdf_test_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_post_out  = torch.tensor( Xdf_post_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    ytn_train_out = torch.tensor(ydf_train_out.values).type(torch.float32)
    ytn_test_out  = torch.tensor( ydf_test_out.values).type(torch.float32)
    ytn_post_out  = torch.tensor( ydf_post_out.values).type(torch.float32)
    wtn_train_out = torch.tensor(Xdf_train_out[wtvar].to_numpy(dtype=float)).type(torch.float32)
    wtn_test_out  = torch.tensor( Xdf_test_out[wtvar].to_numpy(dtype=float)).type(torch.float32)
    wtn_post_out  = torch.tensor( Xdf_post_out[wtvar].to_numpy(dtype=float)).type(torch.float32)

    data_dict_out = dict(
        data=data_in,
        Xdf=Xdf_out,
        Xdf_pre=Xdf_pre_out,
        Xdf_post=Xdf_post_out,
        Xdf_train=Xdf_train_out,
        Xdf_test=Xdf_test_out,
        ydf_train=ydf_train_out,
        ydf_test=ydf_test_out,
        ydf_pre=ydf_pre_out,
        ydf_post=ydf_post_out,
        Xtn_train=Xtn_train_out,
        Xtn_test=Xtn_test_out,
        Xtn_post=Xtn_post_out,
        ytn_train=ytn_train_out,
        ytn_test=ytn_test_out,
        ytn_post=ytn_post_out,
        wtn_train=wtn_train_out,
        wtn_test=wtn_test_out,
        wtn_post=wtn_post_out
    )

    return data_dict_out


def data_setup_asec(path, pred, work_sample="all", test_size=0.5, samp=1):
    if pred not in ["R"]:
        raise ValueError("pred must be R for ASEC ")
    #
    data_in = pd.read_stata(path,convert_dates=True,convert_categoricals=False).sample(frac=samp)\
    #
    if   work_sample=="all":
        data_in = data_in
    elif work_sample=="workly":
        data_in = data_in[data_in.workly==1]
    elif work_sample=="noworkly":
        data_in = data_in[data_in.workly==0]
    else:
        raise ValueError("work_sample must be workly, noworkly, or all")
    #
    # data setup (one-hot encoding etc)
    state    = pd.get_dummies(data_in['statefip'], prefix='state')
    race     = pd.get_dummies(data_in['race'], prefix='race')
    nativity = pd.get_dummies(data_in['nativity'], prefix='nativity')
    educ     = pd.get_dummies(data_in['educ'], prefix='educ')
    agesp    = pd.get_dummies(data_in['agegrp_sp'], prefix='agesp')
    ind_majly = pd.get_dummies(data_in['ind_majly'], prefix='ind_maj', dummy_na=True)
    occ_majly = pd.get_dummies(data_in['occ_majly'], prefix='occ_maj', dummy_na=True)
    fullpart = pd.get_dummies(data_in['fullpart'], prefix='fullpart')
    whynwly  = pd.get_dummies(data_in['whynwly'], prefix='whynwly')
    health   = pd.get_dummies(data_in['health'], prefix='health')
    incq     = pd.get_dummies(data_in['incq'], prefix='incq')
    sex      = data_in['sex'].astype("bool")
    covid    = data_in['covid'].astype("bool")
    marr     = data_in['married'].astype("bool")
    ssa      = data_in['ssa'].astype("bool")
    metro    = data_in['metro'].astype("bool")
    vet      = data_in['vet'].astype("bool")
    #diffmob  = data_in['diffmob'].astype("bool")
    #diffrem  = data_in['diffrem'].astype("bool")
    #diffphys = data_in['diffphys'].astype("bool")
    selfly   = data_in['selfly'].astype("bool")
    govtly   = data_in['govtly'].astype("bool")
    child_any= data_in['child_any'].astype("bool")
    child_yng= data_in['child_yng'].astype("bool")
    child_adt= data_in['child_adt'].astype("bool")
    own      = data_in['own'].astype("bool")
    workly   = data_in['workly'].astype("bool")
    unemly   = data_in['unemly'].astype("bool")
    incrd    = data_in['incrd'].astype("float")
    ssinc    = data_in['ssinc'].astype("float")
    retinc   = data_in['retinc'].astype("float")
    year     = data_in.year.astype("float")
    wksly    = data_in['wksly'].astype("float")
    age      = data_in.age.astype("float")
    agesq    = data_in.agesq.astype("float")
    agecub   = data_in.agecub.astype("float")
    pia      = data_in.pia.astype("float")
    urhat    = data_in.urhat.astype("float")
    ssapia   = pd.DataFrame({"ssapia" :(data_in.pia * data_in.ssa).astype("float")})
    mo       = pd.DataFrame({"mo" : (data_in.year - 2010)*12.0 + 3.0}) # months since 2009m12
    #
    # create x and y dataframes, pre and post, X and y; for each of the two transitions and the retired outcome
    if work_sample=="all":
        Xdf_out = pd.concat([data_in.asecidp, data_in.asecwt, state, race, nativity, educ, agesp,
                            health, sex, covid, marr, ssa, metro, vet,
                            child_any, child_yng, child_adt,
                            own, incrd, year, age, agesq, agecub, pia, urhat, ssapia, mo], axis=1)
    elif work_sample=="workly":
        Xdf_out = pd.concat([data_in.asecidp, data_in.asecwt, state, race, nativity, educ, agesp,
                            health, sex, covid, marr, ssa, metro, vet,
                            child_any, child_yng, child_adt,
                            own, incrd, year, age, agesq, agecub, pia, urhat, ssapia, mo,
                            ind_majly, occ_majly, selfly, govtly, wksly, # this row and below workly
                            unemly, ssinc, retinc, incq, fullpart], axis=1)
    elif work_sample=="noworkly":
        Xdf_out = pd.concat([data_in.asecidp, data_in.asecwt, state, race, nativity, educ, agesp,
                            health, sex, covid, marr, ssa, metro, vet,
                            child_any, child_yng, child_adt,
                            own, incrd, year, age, agesq, agecub, pia, urhat, ssapia, mo,
                            whynwly], axis=1)
    #
    Xdf_pre_out  = Xdf_out[Xdf_out.covid==0]
    Xdf_post_out = Xdf_out[Xdf_out.covid==1]
    ydf_pre_out =  data_in[data_in.covid==0].retired
    ydf_post_out = data_in[data_in.covid==1].retired
    #
    if np.sum(Xdf_out.isnull().sum())>0:
        print("There are missing values in the data")
    #
    # split into test and train -- only need to spit pre
    Xdf_train_out, Xdf_test_out, ydf_train_out, ydf_test_out = train_test_split(
        Xdf_pre_out,
        ydf_pre_out,
        test_size=test_size,
        shuffle=False,
        random_state=1) # make the random split reproducible
    #
    # standardize variables
    sc = StandardScaler()
    for i in range(Xdf_pre_out.shape[1]):
        if (Xdf_train_out.iloc[:,i].dtype!="bool") & (Xdf_train_out.iloc[:,i].name not in ["asecidp","asecwt"]):
            #print(f"{i} is not binary")
            Xdf_train_out.iloc[:,i] = sc.fit_transform(Xdf_train_out.iloc[:,i].to_numpy().reshape(-1,1), y=None)
            Xdf_test_out.iloc[:,i]  = sc.transform(Xdf_test_out.iloc[:,i].to_numpy().reshape(-1,1))
            Xdf_post_out.iloc[:,i]  = sc.transform(Xdf_post_out.iloc[:,i].to_numpy().reshape(-1,1))
    #
    # turn data to tensors
    exclude_cols = ["asecwt", "asecidp", "covid"]
    Xtn_train_out = torch.tensor(Xdf_train_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_test_out  = torch.tensor( Xdf_test_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_post_out  = torch.tensor( Xdf_post_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    ytn_train_out = torch.tensor(ydf_train_out.values).type(torch.float32)
    ytn_test_out  = torch.tensor( ydf_test_out.values).type(torch.float32)
    ytn_post_out  = torch.tensor( ydf_post_out.values).type(torch.float32)
    wtn_train_out = torch.tensor(Xdf_train_out["asecwt"].to_numpy(dtype=float)).type(torch.float32)
    wtn_test_out  = torch.tensor( Xdf_test_out["asecwt"].to_numpy(dtype=float)).type(torch.float32)
    wtn_post_out  = torch.tensor( Xdf_post_out["asecwt"].to_numpy(dtype=float)).type(torch.float32)
    #
    data_dict_out = dict(
        data=data_in,
        Xdf=Xdf_out,
        Xdf_pre=Xdf_pre_out,
        Xdf_post=Xdf_post_out,
        Xdf_train=Xdf_train_out,
        Xdf_test=Xdf_test_out,
        ydf_train=ydf_train_out,
        ydf_test=ydf_test_out,
        ydf_pre=ydf_pre_out,
        ydf_post=ydf_post_out,
        Xtn_train=Xtn_train_out,
        Xtn_test=Xtn_test_out,
        Xtn_post=Xtn_post_out,
        ytn_train=ytn_train_out,
        ytn_test=ytn_test_out,
        ytn_post=ytn_post_out,
        wtn_train=wtn_train_out,
        wtn_test=wtn_test_out,
        wtn_post=wtn_post_out
    )
    #
    return data_dict_out

# ------------------------ modeling ---------------------------------
def batch_model(data_dict_in, n_hidden1, lr=0.1, epochs=500, seed=42,
               weight=False, batch_size=128, report_every=100, num_workers=0):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available and has {os.cpu_count()} workers.") # Print a message to confirm GPU usage
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.") # Print a message if GPU is not available

    torch.manual_seed(seed)

    Xtn_train_in = data_dict_in["Xtn_train"].to(device)
    ytn_train_in = data_dict_in["ytn_train"].to(device)
    Xtn_test_in = data_dict_in["Xtn_test"].to(device)
    ytn_test_in = data_dict_in["ytn_test"].to(device)
    wtn_train_in = torch.squeeze(data_dict_in["wtn_train"]).to(device)
    wtn_test_in = torch.squeeze(data_dict_in["wtn_test"]).to(device)

    train_dataset = TensorDataset(Xtn_train_in, ytn_train_in, wtn_train_in)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    n_hidden2 = int(0.6*n_hidden1)
    class ChurnModel(nn.Module):
        def __init__(self):
            super(ChurnModel, self).__init__()
            self.layer_1 = nn.Linear(Xtn_train_in.shape[1], n_hidden1)
            self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
            self.layer_out = nn.Linear(n_hidden2, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
            self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)
            return x

    model_out = ChurnModel().to(device)

    def loss_fn(y_logits_in, ytn_in, weight_in, wtn_in):
        y_logits_in = torch.squeeze(y_logits_in)
        ytn_in = torch.squeeze(ytn_in)
        wtn_in = torch.squeeze(wtn_in)
        if weight_in == False:
            loss_fn_ = nn.BCEWithLogitsLoss()
            loss_out = loss_fn_(y_logits_in, ytn_in)
        else:
            loss_fn_ = nn.BCEWithLogitsLoss(reduction='none')
            loss_0 = loss_fn_(y_logits_in, ytn_in)
            loss_out = (wtn_in * loss_0 / torch.sum(wtn_in)).sum()
        return loss_out

    def f1_fn(y_true, y_pred):
        tp = (y_true * y_pred).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return 1 - f1

    optimizer = torch.optim.Adam(params=model_out.parameters(), lr=lr)

    train_losses = []
    for epoch in range(epochs):
        model_out.train()
        for batch_X, batch_y, batch_w in train_loader:
            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            optimizer.zero_grad()
            y_logits = model_out(batch_X)
            loss = loss_fn(y_logits, batch_y, weight, batch_w)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if (epoch+1) % report_every == 0:
            model_out.eval()
            with torch.no_grad():
                test_logits = model_out(Xtn_test_in).squeeze();
                test_pred = torch.round(torch.sigmoid(test_logits));
                test_loss = loss_fn(test_logits, ytn_test_in, weight, wtn_test_in);
                test_f1 = f1_fn(y_true=ytn_test_in, y_pred=test_pred);
                print(f"Epoch: {epoch+1} | Avg loss: {np.mean(train_losses):.5f}, F1: TK | Test Loss: {test_loss.item():.5f}, Test F1: {test_f1:.4f}")

        train_losses = []

    model_out.eval()
    with torch.no_grad():
        test_logits = model_out(Xtn_test_in).squeeze();
        test_pred = torch.round(torch.sigmoid(test_logits));
        test_loss = loss_fn(test_logits, ytn_test_in, weight, wtn_test_in);
        test_f1 = f1_fn(y_true=ytn_test_in, y_pred=test_pred);

    # clean up 
    del optimizer, train_dataset, train_loader
    gc.collect()

    evals = {"loss_test": test_loss.item(), "f1_test": test_f1.item()}
    return model_out, evals



def full_model(data_dict_in, n_hidden1, lr=0.1, epochs=500, seed=42, weight=False, report_every=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtn_train_in = data_dict_in["Xtn_train"].to(device)
    ytn_train_in = data_dict_in["ytn_train"].to(device)
    Xtn_test_in = data_dict_in["Xtn_test"].to(device)
    ytn_test_in = data_dict_in["ytn_test"].to(device)
    wtn_train_in = torch.squeeze(data_dict_in["wtn_train"]).to(device)
    wtn_test_in = torch.squeeze(data_dict_in["wtn_test"]).to(device)

    n_hidden2 = int(0.6*n_hidden1)
    num_in = Xtn_train_in.shape[1]
    class ChurnModel(nn.Module):
        def __init__(self):
            super(ChurnModel, self).__init__()
            self.layer_1 = nn.Linear(num_in, n_hidden1)
            self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
            self.layer_out = nn.Linear(n_hidden2, 1)

            self.relu       = nn.ReLU()
            self.leaky      = nn.LeakyReLU()
            self.sigmoid    = nn.Sigmoid()
            self.dropout    = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
            self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)

            return x

    #model_out = model0().to(device)
    model_out = ChurnModel().to(device)

    # define loss function
    def loss_fn(y_logits_in, ytn_in, weight_in, wtn_in):
        if weight_in==False:
            loss_fn_ = nn.BCEWithLogitsLoss();
            loss_out = loss_fn_(y_logits_in, ytn_in); # BCEWithLogitsLoss calculates loss using logits
        else:
            loss_fn_ = nn.BCEWithLogitsLoss(reduction='none')
            loss_0 = loss_fn_(y_logits_in, ytn_in);
            loss_out   = (wtn_in*loss_0/torch.sum(wtn_in)).sum()
        return loss_out

    # Create an optimizer
    optimizer = torch.optim.Adam(params=model_out.parameters(), lr=lr)

    def f1_fn(y_true, y_pred):
        tp = (y_true * y_pred).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return 1 - f1

    #--------------- TEST AND TRAIN LOOP -------------------
    torch.manual_seed(seed)

    # Put data to target device
    Xtn_train_in, ytn_train_in = Xtn_train_in.to(device), ytn_train_in.to(device)
    Xtn_test_in,  ytn_test_in =  Xtn_test_in.to(device),  ytn_test_in.to(device)

    # Training and evaluation loop
    for epoch in range(epochs):
        # Forward pass
        y_logits = model_out(Xtn_train_in).squeeze();
        y_pred = torch.round(torch.sigmoid(y_logits)); # logits -> prediction probabilities -> prediction labels

        # Calculate loss and f1, append to lists for plots; depends on whether using weights
        loss = loss_fn(y_logits, ytn_train_in, weight, wtn_train_in)
        f1  = f1_fn(y_true=ytn_train_in, y_pred=y_pred);

        # Optimizer zero grad, backward loss, optimizer
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        ### Test data
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model_out.eval()
        with torch.inference_mode():
            test_logits = model_out(Xtn_test_in).squeeze();
            test_pred = torch.round(torch.sigmoid(test_logits)); # logits -> prediction probabilities -> prediction labels
            test_loss = loss_fn(test_logits, ytn_test_in, weight, wtn_test_in);
            test_f1 = f1_fn(y_true=ytn_test_in, y_pred=test_pred);

        # Print out what's happening
        if (epoch+1) % report_every == 0:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.5f}, F1: {f1:.3f} | Test Loss: {test_loss.item():.5f}, Test F1: {test_f1:.3f}")

    # clean up 
    del optimizer 
    gc.collect()

    evals = {"f1_train": f1.item(), "loss_train": loss.item(), "f1_test": test_f1.item(), "loss_test": test_loss.item()}
    return model_out, evals


def kfold_cv(model_in, data_dict_in, k, kwargs_var_in, kwargs_const_in={}, path="results.csv"):
    start_time = time.time()

    Xtn_train_in = data_dict_in["Xtn_train"]
    ytn_train_in = data_dict_in["ytn_train"]
    wtn_train_in = data_dict_in["wtn_train"]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    # Generate all possible combinations of parameter values
    param_names  = kwargs_var_in.keys()
    param_values = kwargs_var_in.values()

    for param_combination in itertools.product(*param_values):
        kwargs_var = dict(zip(param_names, param_combination))
        kwargs_all = {**kwargs_var, **kwargs_const_in}

        # check if the csv at path contains a raw with cell values matching (key, value) pairs in kwargs_all
        kwargs_all_df = pd.DataFrame([kwargs_all])
        skip = False 
        if os.path.exists(path):
            prev_results = pd.read_csv(path)
            prev_results = prev_results[list(kwargs_all.keys())]
            if any(prev_results.eq(kwargs_all_df.iloc[0]).all(1)):
                print(f"Skipping parameter combination {kwargs_all} as it already exists in {path}.")
                skip = True
        
        if skip==False:
            fold_losses = []
            fold_f1s = []
            for train_index, val_index in skf.split(Xtn_train_in, ytn_train_in):
                fold_start_time = time.time()
                print(f"=======Running Fold: {len(fold_losses)+1} of {kwargs_var}==========")
                X_train_fold, X_val_fold = Xtn_train_in[train_index], Xtn_train_in[val_index]
                y_train_fold, y_val_fold = ytn_train_in[train_index], ytn_train_in[val_index]
                w_train_fold, w_val_fold = wtn_train_in[train_index], wtn_train_in[val_index]

                temp_data_dict = {
                    "Xtn_train": X_train_fold,
                    "ytn_train": y_train_fold,
                    "Xtn_test": X_val_fold,
                    "ytn_test": y_val_fold,
                    "wtn_train": w_train_fold,
                    "wtn_test": w_val_fold
                }

                if model_in==full_model:
                    model, evals = full_model(temp_data_dict, **kwargs_var, **kwargs_const_in, seed=42, weight=True)
                elif model_in==batch_model:
                    model, evals = batch_model(temp_data_dict,  **kwargs_var, **kwargs_const_in, seed=42, weight=True)
                else:
                    raise ValueError("model must be full_model or batch_model")

                fold_losses.append(evals["loss_test"])
                fold_f1s.append(evals["f1_test"])

                fold_end_time = time.time()
                print(f"fold took {np.round(fold_end_time - fold_start_time)} seconds")

                # clean up 
                del model, evals, temp_data_dict
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = np.mean(fold_losses)
            avg_f1 = np.mean(fold_f1s)

            result = {key: value for key, value in kwargs_all.items()}
            result.update({
                "avg_loss": avg_loss,
                "avg_f1": avg_f1
            })
            results.append(result)

            result_df = pd.DataFrame([result])
            result_df["model"] = model_in.__name__
            if os.path.exists(path):
                existing_df = pd.read_csv(path)
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
                result_df = result_df.drop_duplicates()
            result_df.to_csv(path, index=False)

    if len(results)>0:
        results_df = pd.DataFrame(results)
        results_df["model"] = model_in.__name__
        best_params = results_df.loc[results_df['avg_f1'].idxmin()]
        for col in ["epochs", "n_hidden1", "n_hidden2"]:
            try:
                best_params[col] = int(best_params[col])
            except:
                pass
    else: 
        results_df = pd.DataFrame()
        best_params = None
        print("Nothing run because all parameter combinations already exist in the results file.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Time to run: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    return results_df, best_params


# function for post-testing
def post_data(data_dict_in_, model_in, weight: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dict_in = data_dict_in_.copy()
    data_in = data_dict_in["data"].copy()

    for col in ["data_type","py","yhat","py2","yhat2"]:
        try:
            data_in = data_in.drop(col, axis=1)
            print(f"prior col {col} dropped")
        except:
            pass

    Xdf_to_merge = pd.DataFrame()
    for d in ["train", "test", "post"]:
        Xdf = data_dict_in["Xdf_" + d].copy()
        Xtn = data_dict_in["Xtn_" + d].to(device)
        ytn = data_dict_in["ytn_" + d].to(device)
        wtn = data_dict_in["wtn_" + d].to(device)

        Xdf["data_type"] = d

        # ---------------- BASIC PREDICTIONS -------------------
        py   = torch.sigmoid(model_in(Xtn).squeeze()).detach().cpu().numpy()
        yhat = torch.round(torch.sigmoid(model_in(Xtn).squeeze())).detach().cpu().numpy()

        Xdf["py"]   = py
        Xdf["yhat"] = yhat

        # ----------------- PLATT CALIBRATION -------------------
        with torch.no_grad():
            X_logits = model_in(Xtn).cpu().numpy().flatten()  # get logits for validation set

        if d=="train":
            # Fit Platt scaling (logistic regression on logits)
            platt_model = LogisticRegression(solver = 'lbfgs' )
            if weight:
                platt_model.fit(X_logits.reshape(-1, 1), ytn.cpu().numpy(), sample_weight=wtn.squeeze().numpy())
                print("weights used")
            else:
                platt_model.fit(X_logits.reshape(-1, 1), ytn.cpu().numpy())

        # Calibrate the predicted probabilities using Platt scaling
        Xdf["py2"]    = platt_model.predict_proba(X_logits.reshape(-1, 1))[:, 1]
        Xdf["yhat2"]  = (Xdf["py2"] >= 0.5).astype(int)

        Xdf_to_merge = pd.concat([Xdf_to_merge, Xdf])

    Xdf_to_merge = Xdf_to_merge[["data_type","py","yhat","py2","yhat2"]]

    # join Xdfs to data
    data_out = data_in.join(Xdf_to_merge, how="left")

    data_dict_in["data"] = data_out
    return data_dict_in

# collapse test/train/predicted data by byvar and plot
def coll_graph(data_dict_in, outvar, byvar, pvar="py"):
    df_test_post = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "test") | (data_dict_in["data"]["data_type"] == "post")]
    df_train_post = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "train") | (data_dict_in["data"]["data_type"] == "post")]

    coll_test_post = df_test_post.groupby(byvar, as_index=False).agg({
        pvar: 'mean',
        f'{outvar}': 'mean'
    })

    coll_train_post = df_train_post.groupby(byvar, as_index=False).agg({
        pvar: 'mean',
        f'{outvar}': 'mean'
    })

    fig, axs = plt.subplots(1, 3, figsize=(12,4))

    # Plot the first column vs the second column in the first subplot
    axs[0].plot(coll_test_post[byvar], coll_test_post[pvar], label="predicted")
    axs[0].plot(coll_test_post[byvar], coll_test_post[f'{outvar}'], label="actual")
    axs[0].set_title("Test data: Prediction vs actual")
    axs[0].set_xlabel(byvar)
    axs[0].legend()

    # Plot the second column vs the third column in the second subplot
    axs[1].plot(coll_train_post[byvar], coll_train_post[pvar], label="predicted")
    axs[1].plot(coll_train_post[byvar], coll_train_post[f'{outvar}'], label="actual")
    axs[1].set_title("Train data: Prediction vs actual")
    axs[1].set_xlabel(byvar)
    axs[1].legend()

    # Plot the third column vs the first column in the third subplot
    axs[2].plot(coll_test_post[byvar],  coll_test_post[pvar], label="test")
    axs[2].plot(coll_train_post[byvar], coll_train_post[pvar], label="train")
    axs[2].set_title("Test predictions vs train predictions")
    axs[2].set_xlabel(byvar)
    axs[2].legend()

    # add covid date lines if byvar is mo
    coviddate=pd.to_datetime("2020-03-01")
    if byvar == "mo":
        for ax in axs:
            ax.axvline(x=coviddate, color="red")

    # Adjust layout
    plt.tight_layout()

    # Return the figure
    return fig

# collapse test/train/predicted data by mo and plot
def time_graph(data_dict_in, outvar, pvar="py", smooth=False, diff=False, weight: str = None):
    df_test_post  = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "test")  | (data_dict_in["data"]["data_type"] == "post")]
    df_train_post = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "train") | (data_dict_in["data"]["data_type"] == "post")]

    if weight:
        coll_test_post = df_test_post.groupby("mo", as_index=False).apply(lambda x: pd.Series({
            pvar:   np.average(x[pvar],   weights=x[weight]),
            outvar: np.average(x[outvar], weights=x[weight]) }), include_groups=False)
        coll_train_post = df_train_post.groupby("mo", as_index=False).apply(lambda x: pd.Series({
            pvar:   np.average(x[pvar],   weights=x[weight]),
            outvar: np.average(x[outvar], weights=x[weight]), }), include_groups=False)
    else:
        coll_test_post = df_test_post.groupby("mo", as_index=False).agg({
            pvar: 'mean', outvar: 'mean'})
        coll_train_post = df_train_post.groupby("mo", as_index=False).agg({
            pvar: 'mean', outvar: 'mean'})

    coll_test_post["diff"]  = coll_test_post[f'{outvar}']  - coll_test_post[pvar]
    coll_train_post["diff"] = coll_train_post[f'{outvar}'] - coll_train_post[pvar]

    # create variables that reflect 12-mo average of py, outvar, and diff
    if smooth:
        for col in [pvar, f'{outvar}', "diff"]:
            coll_test_post[f'{col}'] = coll_test_post[col].rolling(12).mean()
            coll_train_post[f'{col}'] = coll_train_post[col].rolling(12).mean()

    fig, axs = plt.subplots(1, 3, figsize=(12,4))

    if diff==False:
        # Plot the first column vs the second column in the first subplot
        axs[0].plot(coll_test_post["mo"], coll_test_post[pvar], label="predicted")
        axs[0].plot(coll_test_post["mo"], coll_test_post[f'{outvar}'], label="actual")
        axs[0].set_title("Test data: Prediction vs actual")
        axs[0].set_xlabel("mo")
        axs[0].legend()

        # Plot the second column vs the third column in the second subplot
        axs[1].plot(coll_train_post["mo"], coll_train_post[pvar], label="predicted")
        axs[1].plot(coll_train_post["mo"], coll_train_post[f'{outvar}'], label="actual")
        axs[1].set_title("Train data: Prediction vs actual")
        axs[1].set_xlabel("mo")
        axs[1].legend()

        # Plot the third column vs the first column in the third subplot
        axs[2].plot(coll_test_post["mo"],  coll_test_post[pvar], label="test")
        axs[2].plot(coll_train_post["mo"], coll_train_post[pvar], label="train")
        axs[2].set_title("Test predictions vs train predictions")
        axs[2].set_xlabel("mo")
        axs[2].legend()

    else:
        # Plot the first column vs the second column in the first subplot
        axs[0].plot(coll_test_post["mo"], coll_test_post["diff"], label="diff")
        axs[0].set_title("Test data: Prediction vs actual")
        axs[0].set_xlabel("mo")
        axs[0].legend()

        # Plot the second column vs the third column in the second subplot
        axs[1].plot(coll_train_post["mo"], coll_train_post["diff"], label="diff")
        axs[1].set_title("Train data: Prediction vs actual")
        axs[1].set_xlabel("mo")
        axs[1].legend()

        # Plot the third column vs the first column in the third subplot
        axs[2].plot(coll_test_post["mo"],  coll_test_post["diff"], label="test diff")
        axs[2].plot(coll_train_post["mo"],  coll_train_post["diff"], label="train diff")
        axs[2].set_title("Test predictions vs train predictions")
        axs[2].set_xlabel("mo")
        axs[2].legend()

    # add covid date lines if byvar is mo
    coviddate=pd.to_datetime("2020-03-01")
    for ax in axs:
        ax.axvline(x=coviddate, color="red")

    # Adjust layout
    plt.tight_layout()

    # Return the figure
    return fig


# collapse test/train/predicted data by mo and plot
def time_graph_by(data_dict_in, outvar, byvar, pvar="py", test_train = "test", smooth=False, weight: str = None):
    if test_train == "test":
        df = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "test") | (data_dict_in["data"]["data_type"] == "post")]
    else:
        df = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "train") | (data_dict_in["data"]["data_type"] == "post")]

    if weight:
        df_coll = df.groupby(["mo",byvar], as_index=False).apply(lambda x: pd.Series({
            pvar:   np.average(x[pvar],   weights=x[weight]),
            outvar: np.average(x[outvar], weights=x[weight]) }), include_groups=False)
    else:
        df_coll = df.groupby(["mo",byvar], as_index=False).agg({pvar: 'mean', outvar: 'mean'})

    # create variables that reflect 12-mo average of py, outvar, and diff
    if smooth:
        for col in [pvar, f'{outvar}']:
            df_coll[f'{col}'] = df_coll.groupby(byvar)[col].transform(lambda x: x.rolling(12, 1).mean())

    def make_grid(num):
        if num<=6:
            cols = num
            rows = 1
        else:
            cols = 6
            rows = np.ceil(num/6).astype(int)
        return (rows,cols)
    numplots = len(df_coll[byvar].unique())
    rownum, colnum = make_grid(numplots)

    w=4 if rownum<=3 else 2

    fig, axs = plt.subplots(rownum, colnum, figsize=(w*colnum,w*rownum))
    axs = axs.flatten()

    # Plot the first column vs the second column in the first subplot
    for i in range(numplots):
        axs[i].plot(df_coll[df_coll[byvar] == df_coll[byvar].unique()[i]]["mo"], df_coll[df_coll[byvar] == df_coll[byvar].unique()[i]][pvar], label="predicted")
        axs[i].plot(df_coll[df_coll[byvar] == df_coll[byvar].unique()[i]]["mo"], df_coll[df_coll[byvar] == df_coll[byvar].unique()[i]][f'{outvar}'], label="actual")
        axs[i].set_title(f"{df_coll[byvar].unique()[i]}")
        axs[i].set_xlabel("mo")
        axs[i].legend()
        axs[i].axvline(x=pd.to_datetime("2020-03-01"), color="red")

    # Adjust layout
    plt.tight_layout()

    # give title
    fig.suptitle(f"{outvar} by {byvar}")

    # Return the figure
    return fig

def update_best_params(path, best_params_in, model_name):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()

    best_params_df = pd.DataFrame([best_params_in])
    best_params_df['model_name'] = model_name

    if 'model_name' in df.columns:
        df = df[df['model_name'] != model_name]

    df = pd.concat([df, best_params_df], ignore_index=True)
    df = df[['model_name'] + [col for col in df.columns if col != 'model_name']]
    df.to_csv(path, index=False)

def out_data(data_dict_in, suffix, path):
    # export cpsidp, mo, data_type, py, py2 to stata
    if "cpsidp" in data_dict_in["data"].columns:
        idvar = "cpsidp"
    elif "asecidp" in data_dict_in["data"].columns:
        idvar = "asecidp"
    else:
        raise ValueError("Neither 'cpsidp' nor 'asecidp' found in data_dict_in['data'].columns")
    data_out =data_dict_in["data"][[idvar, "mo", "data_type", "py2"]]
    data_out = data_out.rename(columns={"py2": f"p_{suffix}"})
    data_out.to_stata(f"{path}.dta", convert_dates={'mo':'%tm'}, write_index=False)
    print("Data exported to " + f"{path}.dta")
    
