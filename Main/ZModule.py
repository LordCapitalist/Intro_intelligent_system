import torch
import torch.nn as nn
import torch.optim as optim

class ZModule(nn.Module):
    def __init__(self, input_dim):
        super(ZModule, self).__init__()
        self.hidden = nn.Linear(input_dim, 64)
        self.output = nn.Linear(64, 1) # Dimentionality reduction
        self.sigmoid = nn.Sigmoid()           

    def forward(self, x):
        x = torch.relu(self.hidden(x)) # 1 => 1, 0 => 0, -1 => 0, -2 = 0, 4 = 4
        x = self.sigmoid(self.output(x)) # 0.1, 0.6, => 0, 1 (Aldrig 0.1, 0.9) 0.98, 0.01
        return x