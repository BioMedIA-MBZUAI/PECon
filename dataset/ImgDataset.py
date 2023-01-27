import torch
torch.manual_seed(0)
from torchvision import transforms
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(self, img_data, targets, transform=None):
        self.img_data = img_data
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        x1 = self.img_data[index]
        y = self.targets[index]
        
        return x1, y
    
    def __len__(self):
        return len(self.img_data)