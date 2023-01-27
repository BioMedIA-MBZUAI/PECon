import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    def __init__(self, img_model, ehr_model):
        super(CLIP,self).__init__()
        self.visual = img_model
        self.text = ehr_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()