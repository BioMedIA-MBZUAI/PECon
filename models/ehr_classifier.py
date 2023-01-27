from models.ehr_MLP import EHR_MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class EHR_classifier(nn.Module):
    def __init__(self, ehr_model):
        super(EHR_classifier, self).__init__()
        
        self.ehr_model = ehr_model
        self.classifier = nn.Linear(128,2)
    
        # nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', 
        #                          nonlinearity='relu')

    def forward(self, data):
        z1 = self.ehr_model(data)
        z1 = self.classifier(z1)
        
        return z1

