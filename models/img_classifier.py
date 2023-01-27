from models.img_MLP import IMG_MLP
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

class IMG_classifier(nn.Module):
    def __init__(self, img_model):
        super(IMG_classifier, self).__init__()
        
        self.img_model = img_model
        self.classifier = nn.Linear(128,2)
    
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', 
                                 nonlinearity='relu')

    def forward(self, data):
        z1 = self.img_model(data)
        z1 = self.classifier(z1)
        
        return z1

