from torch.utils.data import Dataset
import sys
import torch
import os
import re
import numpy as np

class ProbDataset(Dataset):
    def __init__(self, x_path, yhat_path) -> None:
        super().__init__()
        self.x = np.load(x_path, allow_pickle=True)
        self.yhat = np.load(yhat_path, allow_pickle=True)
        assert len(self.x) == len(self.yhat)
        self.len = len(self.x)
   
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'yhat': np.array(self.yhat[idx]).reshape(-1)}

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    p_x = '/home/rzuo02/work/probility-regression/data/2023-02-11/train/x.npy'
    p_yhat = '/home/rzuo02/work/probility-regression/data/2023-02-11/train/yhat.npy'
    dataset = ProbDataset(p_x, p_yhat)
    print(dataset[0]['yhat'].shape)
    print(dataset[0]['yhat'])
    # print(dataset[0]['x'].shape)
    # print(dataset[0]['yhat'].shape)