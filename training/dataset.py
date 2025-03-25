import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TerrainDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_paths = sorted(os.listdir(low_res_dir))
        self.high_res_paths = sorted(os.listdir(high_res_dir))
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform

    def __len__(self):
        return len(self.low_res_paths)

    def __getitem__(self, idx):
        low_res = cv2.imread(os.path.join(self.low_res_dir, self.low_res_paths[idx]), cv2.IMREAD_GRAYSCALE)
        high_res = cv2.imread(os.path.join(self.high_res_dir, self.high_res_paths[idx]), cv2.IMREAD_GRAYSCALE)

        low_res = torch.tensor(low_res, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize
        high_res = torch.tensor(high_res, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize

        return low_res, high_res

# Create PyTorch DataLoader
def get_dataloader(low_res_dir, high_res_dir, batch_size=8):
    dataset = TerrainDataset(low_res_dir, high_res_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
