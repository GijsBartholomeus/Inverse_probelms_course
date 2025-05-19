import numpy as np, torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import datasets, transforms

# --- option A: you already have two NumPy lists A, B --------------------
class PatchMemDS(Dataset):
    def __init__(self, arr, label):
        self.x = torch.from_numpy(arr).float()          # (N,H,W) or (N,H,W,C)
        if self.x.ndim == 3:                            # make CHW
            self.x = self.x.unsqueeze(1)
        elif self.x.shape[-1] in (1,3):
            self.x = self.x.permute(0,3,1,2)
        self.y = torch.full((len(arr),), label, dtype=torch.long)
    def __len__(self):  return len(self.x)
    def __getitem__(self,i):  return self.x[i]/255.0, self.y[i]   # scale 0-1

# ds = ConcatDataset([PatchMemDS(A,0), PatchMemDS(B,1)])

# --- option B: patches saved as ./data/AlgoA/*.png  /AlgoB/*.png --------
tfm = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
ds  = datasets.ImageFolder(root="data", transform=tfm)          # 0=A, 1=B


import torch.nn as nn, torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(          # 20×20  →  6×6
            nn.Conv2d(in_ch,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 10×10
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 5×5
            nn.Conv2d(32,64,3), nn.ReLU())  # 3×3
        self.fc = nn.Linear(64*3*3, n_classes)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)
