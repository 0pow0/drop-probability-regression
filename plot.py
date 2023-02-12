from statistics import correlation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import torch
from scipy.optimize import curve_fit, nnls
import math
import os
import sys
import inspect

from data_loader import ProbDataset
from model import ProbModel 

# plt.rcParams['text.usetex'] = True 

p_test_x = '/home/rzuo02/work/probility-regression/data/2023-02-11/test/x.npy'
p_test_yhat = '/home/rzuo02/work/probility-regression/data/2023-02-11/test/yhat.npy'
test_dataset = ProbDataset(p_test_x, p_test_yhat)

test_dataloader  = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=None,
                                                batch_sampler=None)

model_path = '/home/rzuo02/work/probility-regression/data/2023-02-11/check_points/2023-02-11 13:34:45/epoch_1448_0.37413862347602844'
model = ProbModel()
model.load_state_dict(torch.load(model_path))

criterion = torch.nn.MSELoss(reduction='sum')

model.eval()
sum_test_loss = 0.0

ys = []
yhats = []

with torch.no_grad():
    for idx, data in enumerate(test_dataloader):
        x = data['x'].float()
        yhat = data['yhat'].float()
        y = model(x)
        loss = criterion(yhat, y)
        sum_test_loss = sum_test_loss + loss
        ys.append(y)
        yhats.append(yhat)

test_loss = sum_test_loss / len(test_dataloader)
print('Loss: ', test_loss)

ys = torch.stack(ys)
yhats = torch.stack(yhats)
 
ys = ys.detach().numpy()
yhats = yhats.detach().numpy()

def cor(y, yhat):
    ybar = y.mean()
    yhatbar = yhat.mean()
    numerator = np.sum((y - ybar) * (yhat - yhatbar))
    denominator = np.sqrt(np.sum(np.power(y - ybar, 2)) * np.sum(np.power(yhat - yhatbar, 2)))
    return numerator / denominator

correlation = cor(ys, yhats)

print(ys.shape)
print(yhats.shape)

def line_func(x, w, b):
    return x * w + b

lopt, lcov = curve_fit(line_func, ys.reshape(-1), yhats.reshape(-1))
print(lopt)

fig, ax = plt.subplots(facecolor=(1, 1, 1))
ax.scatter(yhats, ys)
ax.plot(ys, line_func(ys, *lopt), c='red', ls='dotted', label=r'yhat='+str(lopt[0])+'y+'+str(lopt[1]))
# ax.plot(y, y, c='gray', ls='dotted')#, label=r'\hat{y}='+str(lopt[0])+'y+'+str(lopt[1]))
ax.set_xlabel(r'$\hat{y}$', fontsize=15)
ax.set_ylabel(r'$y$', fontsize=15)
ax.legend()
fig.suptitle("Loss (MSE) = " + str(test_loss.item()) + " Corr = " + str(correlation))
# fig.suptitle("Loss (MSE)" + test_loss+ "Corr = " + str(correlation))
fig.tight_layout()
fig.savefig('test.png', dpi=300)
# plt.show()
