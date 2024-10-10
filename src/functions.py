import sys 
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

# data load 
def data_setup(path, pred, test_size=0.5, samp=1):
    if pred not in ["ER", "UNR", "R"]:
        raise ValueError("pred must be ER or UNR or R")
    
    data_in = pd.read_stata(path,convert_dates=True,convert_categoricals=False).sample(frac=samp)
    cols = data_in.columns
    cols

    # keep only mish4 & employed
    if pred=="ER":
        data_in = data_in[(data_in.mish!=4) & (data_in.mish!=8) & (data_in.employed==1) & (data_in.f_retired.notna())]
    if pred=="UNR":
        data_in = data_in[(data_in.mish!=4) & (data_in.mish!=8) &  (data_in.retired==0) & (data_in.employed==0) & (data_in.f_retired.notna())]
    if pred=="R":
        data_in = data_in 

    # test for nans 
    for c in cols:
        s = sum(data_in[c].isna())
        if s>0: 
            print(s, c, "missing!")

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
    covid  = data_in['covid'].astype("bool")
    f_covid  = data_in['f_covid'].astype("bool")
    marr     = data_in['married'].astype("bool")
    ssa      = data_in['ssa'].astype("bool")
    metro    = data_in['metro'].astype("bool")
    vet      = data_in['vet'].astype("bool")
    diffmob  = data_in['diffmob'].astype("bool")
    diffrem  = data_in['diffrem'].astype("bool")
    diffphys = data_in['diffphys'].astype("bool")
    unem     = data_in['unem'].astype("bool")
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
    #wageflag = data['wageflag'].astype("bool")
    age      = data_in.age.astype("float")
    agesq    = data_in.agesq.astype("float")
    agecub   = data_in.agecub.astype("float")
    #wage     = data.wage.astype("float")
    mo       = pd.DataFrame({"mo" : (data_in.year - 2010)*12 + data_in.month.astype("float")}) # months since 2009m12
    pia      = data_in.pia.astype("float")
    ssapia   = pd.DataFrame({"ssapia" :(data_in.pia * data_in.ssa).astype("float")}) 

    # create x and y dataframes, pre and post, X and y; for each of the two transitions and the retired outcome 
    if pred=="ER":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, mish, mo, month, state, race, nativity, sex, educ, f_covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, diffmob, diffrem, diffphys, age, agesq, agecub, pia, ssapia, 
                            ind_maj, occ_maj, govt, ft, absnt, self], axis=1) 
        Xdf_pre_out  = Xdf_out[Xdf_out.f_covid==0]
        Xdf_post_out = Xdf_out[Xdf_out.f_covid==1]
        ydf_pre_out =  data_in[data_in.f_covid==0].f_retired
        ydf_post_out = data_in[data_in.f_covid==1].f_retired

    if pred=="UNR":  
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, mish, mo, month, state, race, nativity, sex, educ, f_covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, diffmob, diffrem, diffphys, age, agesq, agecub, pia, ssapia, 
                            unem, untemp, unlose, unable, nlf_oth], axis=1) 
        Xdf_pre_out  = Xdf_out[Xdf_out.f_covid==0]
        Xdf_post_out = Xdf_out[Xdf_out.f_covid==1]
        ydf_pre_out =  data_in[data_in.f_covid==0].f_retired
        ydf_post_out = data_in[data_in.f_covid==1].f_retired

    if pred=="R":
        Xdf_out = pd.concat([data_in.cpsidp, data_in.wtfinl, mish, mo, month, state, race, nativity, sex, educ, covid, marr,
                            agesp, ssa, metro, child_any, child_yng, child_adt, vet, diffmob, diffrem, diffphys, age, agesq, agecub, pia, ssapia], axis=1) 
        Xdf_pre_out  = Xdf_out[Xdf_out.covid==0]
        Xdf_post_out = Xdf_out[Xdf_out.covid==1]
        ydf_pre_out =  data_in[data_in.covid==0].retired
        ydf_post_out = data_in[data_in.covid==1].retired

    # split into test and train -- only need to spit pre 
    Xdf_train_out, Xdf_test_out, ydf_train_out, ydf_test_out = train_test_split(
        Xdf_pre_out, 
        ydf_pre_out, 
        test_size=test_size, 
        shuffle=False,
        random_state=40) # make the random split reproducible 
    
    # standardize variables 
    sc = StandardScaler()

    for i in range(Xdf_pre_out.shape[1]):
        if (Xdf_train_out.iloc[:,i].dtype!="bool") & (Xdf_train_out.iloc[:,i].name not in ["cpsidp","wtfinl"]): 
            print(f"{i} is not binary")
            Xdf_train_out.iloc[:,i] = sc.fit_transform(Xdf_train_out.iloc[:,i].to_numpy().reshape(-1,1), y=None)
            Xdf_test_out.iloc[:,i]  = sc.transform(Xdf_test_out.iloc[:,i].to_numpy().reshape(-1,1))
            Xdf_post_out.iloc[:,i]  = sc.transform(Xdf_post_out.iloc[:,i].to_numpy().reshape(-1,1))
    
    # turn data to tensors 
    if (pred=="R"): 
        covidvar="covid"
    else:
        covidvar="f_covid"
    
    exclude_cols = [covidvar, "cpsidp", "wtfinl"]
    Xtn_train_out = torch.tensor(Xdf_train_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_test_out  = torch.tensor( Xdf_test_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    Xtn_post_out  = torch.tensor( Xdf_post_out.drop(exclude_cols, axis=1).to_numpy(dtype=float)).type(torch.float32)
    ytn_train_out = torch.tensor(ydf_train_out.values).type(torch.float32)
    ytn_test_out  = torch.tensor( ydf_test_out.values).type(torch.float32)
    ytn_post_out  = torch.tensor( ydf_post_out.values).type(torch.float32)
    wtn_train_out = torch.tensor(Xdf_train_out[["wtfinl"]].to_numpy(dtype=float)).type(torch.float32)
    wtn_test_out  = torch.tensor( Xdf_test_out[["wtfinl"]].to_numpy(dtype=float)).type(torch.float32)
    wtn_post_out  = torch.tensor( Xdf_post_out[["wtfinl"]].to_numpy(dtype=float)).type(torch.float32)

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

# ------------------------ modeling ---------------------------------
def run_model(data_dict_in, n_hidden1, n_hidden2, lr=0.1, epochs=500, seed=42, weight=False):
    Xtn_train_in = data_dict_in["Xtn_train"]
    ytn_train_in = data_dict_in["ytn_train"]
    Xtn_test_in = data_dict_in["Xtn_test"]
    ytn_test_in = data_dict_in["ytn_test"]
    wtn_train_in = torch.squeeze(data_dict_in["wtn_train"])
    wtn_test_in = torch.squeeze(data_dict_in["wtn_test"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    
    num_in = Xtn_train_in.shape[1]
    #n_hidden1 = 20
    #n_hidden2 = 10
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
    optimizer = torch.optim.Adam(params=model_out.parameters(), 
                                lr=lr)

    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc

    #--------------- TEST AND TRAIN LOOP -------------------
    torch.manual_seed(seed)
    
    # device setup  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Put data to target device
    Xtn_train_in, ytn_train_in = Xtn_train_in.to(device), ytn_train_in.to(device)
    Xtn_test_in,  ytn_test_in =  Xtn_test_in.to(device),  ytn_test_in.to(device)

    # Build training and evaluation loop
    accs_out = [] 
    losses_out = []
    figdict = dict() 
    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model_out(Xtn_train_in).squeeze();
        y_pred = torch.round(torch.sigmoid(y_logits)); # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy, append to lists for plots; depends on whether using weights 
        loss = loss_fn(y_logits, ytn_train_in, weight, wtn_train_in)

        acc = accuracy_fn(y_true=ytn_train_in, 
                          y_pred=y_pred);
        
        accs_out.append(acc);
        losses_out.append(loss.item());
        
        # 3. Optimizer zero grad
        optimizer.zero_grad();

        # 4. Loss backward
        loss.backward();

        # 5. Optimizer step
        optimizer.step();

        ### Testing
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            model_out.eval()
        
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_out(Xtn_test_in).squeeze();
            test_pred = torch.round(torch.sigmoid(test_logits)); # logits -> prediction probabilities -> prediction labels
            
            # 2. Calcuate loss and accuracy
            test_loss = loss_fn(test_logits, ytn_test_in, weight, wtn_test_in);
            test_acc = accuracy_fn(y_true=ytn_test_in,
                                      y_pred=test_pred);

        # Print out what's happening
        print(f"Epoch: {epoch} | Loss: {loss.item():.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss.item():.5f}, Test Accuracy: {test_acc:.2f}%")
        
        #if epoch % 100 == 0:
        #    data_test_post_tmp, data_train_post_tmp = post_data(data_dict_in, model_out)
        #    p = coll_graph(data_test_post_tmp, data_train_post_tmp, "mo")
        #    name = "fig" + str(epoch)
        #    figdict[name] = p
    
    return model_out, accs_out, losses_out, figdict 

# function for post-testing 
def post_data(data_dict_in_, model_in, weight=False): 
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
        Xtn = data_dict_in["Xtn_" + d]
        ytn = data_dict_in["ytn_" + d]
        wtn = data_dict_in["wtn_" + d]

        Xdf["data_type"] = d

        # ---------------- BASIC PREDICTIONS -------------------
        Xdf["py"]   = torch.sigmoid(model_in(Xtn).squeeze()).detach().numpy()
        Xdf["yhat"] = torch.round(torch.sigmoid(model_in(Xtn).squeeze())).detach().numpy()

        # ----------------- PLATT CALIBRATION -------------------
        with torch.no_grad():
            X_logits = model_in(Xtn).numpy().flatten()  # get logits for validation set   

        if d=="train":
            # Fit Platt scaling (logistic regression on logits)
            platt_model = LogisticRegression(solver = 'lbfgs' )
            if weight==False:
                platt_model.fit(X_logits.reshape(-1, 1), ytn.numpy())
            else: 
                platt_model.fit(X_logits.reshape(-1, 1), ytn.numpy(), sample_weight=wtn.squeeze().numpy()) 
                print("weights used")

        # Calibrate the predicted probabilities using Platt scaling
        Xdf["py2"] = platt_model.predict_proba(X_logits.reshape(-1, 1))[:, 1]
        Xdf["yhat2"] = (Xdf["py2"] >= 0.5).astype(int)

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
def time_graph(data_dict_in, outvar, pvar="py", smooth=False, diff=False, weight=False):
    df_test_post = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "test") | (data_dict_in["data"]["data_type"] == "post")]
    df_train_post = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "train") | (data_dict_in["data"]["data_type"] == "post")]

    if weight==False:
        coll_test_post = df_test_post.groupby("mo", as_index=False).agg({
            pvar: 'mean', outvar: 'mean'})
        coll_train_post = df_train_post.groupby("mo", as_index=False).agg({
            pvar: 'mean', outvar: 'mean'})
    else:
        coll_test_post = df_test_post.groupby("mo", as_index=False).apply(lambda x: pd.Series({
            pvar:   np.average(x[pvar],   weights=x['wtfinl']),
            outvar: np.average(x[outvar], weights=x['wtfinl']) }), include_groups=False)
        coll_train_post = df_train_post.groupby("mo", as_index=False).apply(lambda x: pd.Series({
            pvar:   np.average(x[pvar],   weights=x['wtfinl']),
            outvar: np.average(x[outvar], weights=x['wtfinl']), }), include_groups=False)

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
def time_graph_by(data_dict_in, outvar, byvar, pvar="py", test_train = "test", smooth=False):
    if test_train == "test":
        df = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "test") | (data_dict_in["data"]["data_type"] == "post")]
    else: 
        df = data_dict_in["data"][(data_dict_in["data"]["data_type"] == "train") | (data_dict_in["data"]["data_type"] == "post")]

    df_coll = df.groupby(["mo",byvar], as_index=False).agg({
        pvar: 'mean', 
        f'{outvar}': 'mean'
    })

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

def out_data(data_dict_in, suffix, filename):
    # export cpsidp, mo, data_type, py, py2 to stata 
    data_out =data_dict_in["data"][["cpsidp", "mo", "data_type", "py", "py2"]]
    data_out = data_out.rename(columns={"py": f"py_{suffix}", "py2": f"py2_{suffix}"})
    data_out.to_stata(f"data/generated/{filename}.dta", convert_dates={'mo':'%tm'})
    print("Data exported to " + f"data/generated/{filename}.dta")

