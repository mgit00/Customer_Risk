import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler, PowerTransformer
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader   

import numpy as np 
from sklearn.utils.class_weight import compute_class_weight

from utils.schemas import ModelTrainingBasemodel
from utils.constants import ConstantFilenames, ModelType

from pipeline.utils import Utils 

### FIXME - can remove file 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

 # read data 
data = Utils.read_data(ConstantFilenames.data_filename.value)

# x,y = data.iloc[:, :-1], data.iloc[:, -1:] # df, and df (not series, makes mode code more complicated to switch between classificaiton and regression later on) 
y = data[['risk_level']]
x = data.drop(columns = ['risk_level'])

# y has to be long if using crossentropy loss (classification) or float32 for mse loss (regression)
x = torch.tensor(x.values, dtype = torch.float32) # float32 = default for pytorch models/model layers 
y = torch.tensor(y.values.ravel(), dtype= torch.long )

trainx, testx, trainy, testy = train_test_split(x, y, test_size=.3, random_state= 1, shuffle = True)
    # random state makes it replicable 
    # batching not done here, done in dataloader 
    # shuffle rows 

# 1. Ensure clean float32 scaling
scaler = StandardScaler()
X_tr = torch.tensor(scaler.fit_transform(trainx.numpy()), dtype=torch.float32)
X_te = torch.tensor(scaler.transform(testx.numpy()), dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(TensorDataset(X_tr, trainy), batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(X_te, testy), batch_size=256, shuffle=False)

# 2. Widen network for 36k rows
model = nn.Sequential(
    nn.Linear(X_tr.shape[1], 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)

# 3. Softened class weights (sqrt prevents Class 2's 28x weight from destabilizing gradients)
weights = torch.tensor([ 0.4397,  1.4364, 33.7553], dtype=torch.float32)
class_weights=weights.to(device)
# soft_weights = torch.sqrt(class_weights).to(device)
# catch classes 1 and 2 
soft_weights = class_weights.to(device) # thics change does help 
criterion = nn.CrossEntropyLoss(weight=soft_weights)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# learn more slowly 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

for epoch in range(15):
    model.train()
    curr_loss = 0 
    curr_loss2 = 0 
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        curr_loss+= loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            res = model(bx)
            loss = criterion(res, by)
            curr_loss2+= loss.item()
            preds = res.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)
    print(f"Epoch {epoch:02d} | Test Acc: {correct / total:.4f}, trainloss {curr_loss/len(train_loader)}, testloss {curr_loss2/len(test_loader)}")

# python -m src.testing