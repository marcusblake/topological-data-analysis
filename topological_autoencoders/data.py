import pandas as pd
import torch
from torch.utils.data import Dataset

class EMNISTDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath, header=None)
        self.labels = data.iloc[:,1].to_numpy()
        self.images = data.iloc[:,2:].to_numpy()

        # Normalize the images so that each index is between [0,1].
        self.images = self.images / self.images.max()
        self.images = torch.from_numpy(self.images)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
