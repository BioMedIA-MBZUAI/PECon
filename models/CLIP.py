import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    def __init__(self, img_model, ehr_model, penet_backbone, unfreeze_penet=True):
        super(CLIP,self).__init__()
        self.visual = img_model
        self.text = ehr_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.penetbackbone = penet_backbone
        if(unfreeze_penet == True):
            ct =0
            print("[INFO] Unfreeze few PeNet blocks...")
            for child in self.penetbackbone.children():
                ct+=1
                if ct<4:
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            print("[INFO] Freezing PeNet backbone...")
            for param in self.penetbackbone.parameters():
                param.requires_grad = False

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        output = self.penetbackbone(image)
        mean_image_features = output.mean(dim=0)
        image_features = self.encode_image(mean_image_features, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()