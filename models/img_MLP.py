import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class IMG_MLP(nn.Module):
    def __init__(self):
        super(IMG_MLP, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512,256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256,128)
        
        

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', 
                                 nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', 
                                 nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', 
                                 nonlinearity='relu')

    def forward(self, data):
        z1 = F.relu(self.fc1(data))
        z1 = self.dropout1(z1)
        z1 = F.relu(self.fc2(z1))
        z1 = (self.fc3(z1))
        # print("z1.shape: ", z1.shape)
        
        return z1