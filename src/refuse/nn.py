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

# function for post-testing 
def post_data(xtest,xtrain,xpost,ytest,ytrain,ypost,model,xcols,round=False): 
    # get predictions 
    if round==True:
        p_test   = torch.round(torch.sigmoid(model(xtest).squeeze())).detach().numpy()
        p_train  = torch.round(torch.sigmoid(model(xtrain).squeeze())).detach().numpy()
        p_post   = torch.round(torch.sigmoid(model(xpost).squeeze())).detach().numpy()
    else: 
        p_test   = torch.sigmoid(model(xtest).squeeze()).detach().numpy()
        p_train  = torch.sigmoid(model(xtrain).squeeze()).detach().numpy()
        p_post   = torch.sigmoid(model(xpost).squeeze()).detach().numpy()

    # combine pred (test & post, train & post)
    p_test_post =  pd.DataFrame({"py" : np.append(p_test,  p_post) })
    p_train_post = pd.DataFrame({"py" : np.append(p_train, p_post) })

    # combine y (test & post, train & post)
    y_test_post  = pd.DataFrame({"y" : np.append(ytest.numpy(),  ypost.numpy()) })
    y_train_post = pd.DataFrame({"y" : np.append(ytrain.numpy(), ypost.numpy()) })

    # create full data frames 
    data_test_post  = pd.DataFrame(torch.cat((xtest,  xpost), dim=0))
    data_train_post = pd.DataFrame(torch.cat((xtrain, xpost), dim=0))
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

# data load 
samp = .5
data = pd.read_stata('data/covid_data.dta',convert_dates=True,convert_categoricals=False).sample(frac=samp)
cols = data.columns
cols

# data setup (one-hot encoding etc) 
month  = pd.get_dummies(data['month'], prefix='month')
mish   = pd.get_dummies(data['mish'], prefix='mish')
state  = pd.get_dummies(data['statefip'], prefix='state')
race   = pd.get_dummies(data['race'], prefix='race')
educ   = pd.get_dummies(data['educ'], prefix='educ')
agesp  = pd.get_dummies(data['agegrp_sp'], prefix='agesp')
sex    = data['sex'].astype("bool")
covid  = data['covid'].astype("bool")
marr   = data['married'].astype("bool")
ssa    = data['ssa'].astype("bool")
metro  = data['metro'].astype("bool")
vet    = data['vet'].astype("bool")
dis    = data['disable'].astype("bool")
age    = data.age.astype("float")
agesq  = data.agesq.astype("float")
agecub = data.agecub.astype("float")
mo     = pd.DataFrame({"mo" : (data.year - 2010)*12 + data.month.astype("float")}) # months since 2009m12
mosq   = pd.DataFrame({"mosq" : (mo.mo*mo.mo).astype("float") }) # months since 2009m12
pia    = data.pia.astype("float")
ssapia = (data.pia * data.ssa).astype("float")
ur     = data.ur.astype("float")
urhat  = data.urhat.astype("float")

# create x and y dataframes
Xdf = pd.concat([month,mish,state,race,sex,educ,covid,marr,agesp,ssa,metro,vet,dis,age,agesq,agecub,mo,pia,ssapia], axis=1) # leaving out ur,urhat
#Xdf = pd.concat([covid,age,agelog], axis=1) # leaving out ur,urhat
Xdf_pre  = Xdf[Xdf.covid==0]
Xdf_post = Xdf[Xdf.covid==1]

ydf_pre =  data[data.covid==0].retired 
ydf_post = data[data.covid==1].retired 

# split into test and train -- only need to spit pre 
Xdf_train, Xdf_test, ydf_train, ydf_test = train_test_split(
    Xdf_pre, 
    ydf_pre, 
    test_size=0.2, # 20% test, 80% train
    shuffle=False,
    random_state=42) # make the random split reproducible 

# standardize variables 
sc = StandardScaler()

for i in range(Xdf_pre.shape[1]):
    if Xdf_train.iloc[:,i].dtype!="bool": 
        print(f"{i} is not binary")
        Xdf_train.iloc[:,i] = sc.fit_transform(Xdf_train.iloc[:,i].to_numpy().reshape(-1,1), y=None)
        Xdf_test.iloc[:,i]  = sc.transform(Xdf_test.iloc[:,i].to_numpy().reshape(-1,1))
        Xdf_post.iloc[:,i]  = sc.transform(Xdf_post.iloc[:,i].to_numpy().reshape(-1,1))

# turn data to tensors 
X_train = torch.tensor(Xdf_train.to_numpy(dtype=float)).type(torch.float32)
X_test  = torch.tensor(Xdf_test.to_numpy(dtype=float)).type(torch.float32)
X_post  = torch.tensor(Xdf_post.to_numpy(dtype=float)).type(torch.float32)
y_train = torch.tensor(ydf_train.values).type(torch.float32)
y_test  = torch.tensor(ydf_test.values).type(torch.float32)
y_post  = torch.tensor(ydf_post.values).type(torch.float32)

# ------------------------ modeling ---------------------------------
# device setup  
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# 1. Construct a model class that subclasses nn.Module
num_in = X_train.shape[1]
n_hidden1 = 20
n_hidden2 = 10
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(num_in, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, 1) 
        
        self.relu       = nn.ReLU()
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

#model_0 = model0().to(device)
model_0 = ChurnModel().to(device)

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.Adam(params=model_0.parameters(), 
                            lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


#--------------- TEST AND TRAIN LOOP -------------------
torch.manual_seed(42)

# Set the number of epochs
epochs = 500

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
accs = [] 
losses = []
for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_0(X_train).squeeze();
    y_pred = torch.round(torch.sigmoid(y_logits)); # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy, append to lists for plots 
    loss = loss_fn(y_logits, y_train); # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred);
    
    accs.append(acc);
    losses.append(loss.item());
    
    # 3. Optimizer zero grad
    optimizer.zero_grad();

    # 4. Loss backward
    loss.backward();

    # 5. Optimizer step
    optimizer.step();

    ### Testing
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        model_0.eval()
    
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze();
        test_pred = torch.round(torch.sigmoid(test_logits)); # logits -> prediction probabilities -> prediction labels
        
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, y_test);
        test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred);

    # Print out what's happening
    #if epoch % 100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# look at path of accuracy and losses 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(np.arange(len(accs)), accs)
ax2.plot(np.arange(len(losses)), losses)
plt.show()

# post testing 
data_test_post, data_train_post = post_data(X_test,X_train,X_post,y_test,y_train,y_post,model_0,Xdf_post.columns)
p =  coll_graph(data_test_post, data_train_post, "age")
p.show()

p =  coll_graph(data_test_post, data_train_post, "mo")
p.show()

p =  coll_graph(data_test_post, data_train_post, "pia")
p.show()


p =  coll_graph(
      data_test_post[(data_test_post.educ_3==1)  | ( data_test_post.educ_4==1)], 
    data_train_post[(data_train_post.educ_3==1) |  (data_train_post.educ_4==1)], 
    "mo")
p.show()

p =  coll_graph(
      data_test_post[(data_test_post.race_1==0) ], 
    data_train_post[(data_train_post.race_1==0) ], 
    "mo")
p.show()

