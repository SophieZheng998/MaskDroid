from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y