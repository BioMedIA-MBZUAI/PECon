from torchvision import transforms
from torch.utils.data import Dataset
import torch
torch.manual_seed(0)


class EhrDataset(Dataset):
    def __init__(self, ehr_data, targets, transform=None):
        self.ehr_data = ehr_data
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        x2 = self.ehr_data[index]
        y = self.targets[index]
        
        return x2, y
    
    def __len__(self):
        return len(self.ehr_data)