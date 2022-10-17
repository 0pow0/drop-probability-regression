from torch.utils.data import Dataset
import sys
import torch
import os
import re
import numpy as np

class ProbDataset(Dataset):
    def __init__(self, x_path, yhat_path) -> None:
        super().__init__()
        self.x = np.load(x_path)
        self.yhat = np.load(yhat_path)
        assert len(self.x) == len(self.yhat)
        self.len = len(self.x)
   
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'yhat': self.yhat[idx]}

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    p_x = '/home/rui/work/prob-regression-model/data/2022-10-08/x.npy'
    p_yhat = '/home/rui/work/prob-regression-model/data/2022-10-08/yhat.npy'
    dataset = ProbDataset(p_x, p_yhat)
    print(dataset[0:10])