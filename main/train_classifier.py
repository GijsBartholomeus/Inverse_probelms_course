"""In this file we train a classifier to distinguish between two reconstruction algorithms: Algoritme A and Algoritme B.

The classifier is trained on a dataset of patches of reconstructed images from both algorithms. The dataset is created by the script `build_physical_dataset.py`, which generates a set of phantoms, reconstructs them using both algorithms, and saves the patches to disk.

The pathes are of a smaller size than the orginal images.
Just to make the training faster, we use ~ 30 by 30 pixels patchs.
We only want to prove of concept.
"""
from utils.cnn import SmallCNN
import torch 
from torch import nn
from torch.utils.data import DataLoader
# 1 Load in data



# 2. set hyperparamets
batch_size = 64
train_len  = int(0.8*len(ds))
train_ds, val_ds = torch.utils.data.random_split(ds,[train_len, len(ds)-train_len])
train_dl = DataLoader(train_ds,batch_size,shuffle=True)
val_dl   = DataLoader(val_ds,batch_size)
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. set model, optimizer and loss function
model = SmallCNN().to(device)                           # Set model
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)   # Set optimizer
lossf = nn.CrossEntropyLoss()                           # Set loss function   

for epoch in range(10):
    model.train()
    for xb,yb in train_dl:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward();  opt.step()

    model.eval();   correct = total = 0
    with torch.no_grad():
        for xb,yb in val_dl:
            out = model(xb.to(device));  pred = out.argmax(1)
            correct += (pred.cpu()==yb).sum().item();  total += yb.size(0)
    print(f"epoch {epoch+1:2d}: val-acc = {correct/total:.3f}")
