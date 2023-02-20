#ehr model
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

class EHR_MLP(nn.Module):
    def __init__(self):
        super(EHR_MLP, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128,128)
        self.dropout1 = nn.Dropout(0.4)
        

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', 
                                 nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', 
                                 nonlinearity='relu')
        

    def forward(self, data):
        z1 = F.relu(self.fc1(data))
        z1 = self.dropout1(z1)
        z1 = (self.fc2(z1))
        
        return z1