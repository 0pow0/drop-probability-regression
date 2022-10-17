import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import torch
from model import ProbModel
from data_loader import ProbDataset 
from scipy.optimize import curve_fit, nnls
import math
import pandas as pd
import os
import re

def plot():
    # plt.rcParams['text.usetex'] = True 

    p_x = '/home/rui/work/prob-regression-model/data/2022-10-08/x.npy'
    p_yhat = '/home/rui/work/prob-regression-model/data/2022-10-08/yhat.npy'
    dataset = ProbDataset(p_x, p_yhat)
    test_set, _ = torch.utils.data.random_split(dataset, [2000, 18000])

    test_dataloader  = torch.utils.data.DataLoader(test_set,
                                                    batch_size=None,
                                                    batch_sampler=None)

    model_path = "/home/rui/work/prob-regression-model/data/2022-10-08/check_points/2022-10-08 20:01:44/epoch_2379_1.2610507011413574"
    model = ProbModel()
    model.load_state_dict(torch.load(model_path))

    criterion = torch.nn.MSELoss(reduction='sum')

    model.eval()
    test_loss = 0.0

    ys = []
    yhats = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            x = data['x'].float()
            print(x.shape)
            return
            yhat = data['yhat'].float()
            y = model(x)
            loss = criterion(y, yhat.data)
            test_loss = test_loss + loss
            ys.append(y)
            yhats.append(yhat)

    print('Loss: ', test_loss)

    y = torch.stack(ys)
    yhat = torch.stack(yhats)
    
    y = y.detach().numpy()
    yhat = yhat.detach().numpy()

    def cor(y, yhat):
        ybar = y.mean()
        yhatbar = yhat.mean()
        numerator = np.sum((y - ybar) * (yhat - yhatbar))
        denominator = np.sqrt(np.sum(np.power(y - ybar, 2)) * np.sum(np.power(yhat - yhatbar, 2)))
        return numerator / denominator

    correlation = cor(y, yhat)

    print(y.shape)
    print(yhat.shape)

    def line_func(x, w, b):
        return x * w + b

    lopt, lcov = curve_fit(line_func, y.reshape(-1), yhat.reshape(-1))
    print(lopt)

    fig, ax = plt.subplots(facecolor=(1, 1, 1))
    ax.scatter(y, yhat)
    ax.plot(y, line_func(y, *lopt), c='red', ls='dotted', label=r'yhat='+str(lopt[0])+'y+'+str(lopt[1]))
    # ax.plot(y, y, c='gray', ls='dotted')#, label=r'\hat{y}='+str(lopt[0])+'y+'+str(lopt[1]))
    ax.set_xlabel(r'$y$', fontsize=15)
    ax.set_ylabel(r'$\hat{y}$', fontsize=15)
    ax.legend()
    # fig.suptitle("Loss (MSE) = " + str(test_loss) + " Corr = " + str(correlation))
    fig.suptitle("Corr = " + str(correlation))
    fig.tight_layout()
    fig.savefig('test.png', dpi=300)
    # plt.show()


def plot1():
    p_d = re.compile('\d+')
    p_rx = re.compile('Rx \d+')
    p_name = re.compile('-\d+-*')

    model_path = "/home/rui/work/prob-regression-model/data/2022-10-08/check_points/2022-10-08 20:01:44/epoch_2379_1.2610507011413574"
    model = ProbModel()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    ys = []

    with torch.no_grad():
        for x in range(1, 255):
            x = torch.Tensor([x])
            print(x.shape)
            y = model(x)
            ys.append(y)

    y = torch.stack(ys)
    y = y.detach().numpy()

    fig, ax = plt.subplots()
    # ax.plot(df['Number of Interference Ue'], df['y'])
    ax.plot(np.arange(1, 255), y)
    ax.set_xlabel('Number of Interference Ue', fontsize=15)
    # ax.set_ylabel(r'$\hat{y}$', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    # fig.savefig(prefix + 'y.png', dpi=300, bbox_inches='tight')
    fig.savefig('y.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # plot()
    plot1()
