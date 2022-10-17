import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import os
import re
import math

p_d = re.compile('\d+')
p_numUe = re.compile('-\d+-*')

def gen_csv(folder):
    df = pd.DataFrame()
    df["Number of Interference Ue"] = ""
    df["yhat"] = ""
    files = natsorted(os.listdir(folder))
    df_index = 0
    for i in tqdm(range(0, len(files), 2)):
        f_intf = files[i]
        f_subj = files[i+1]
        numUe = int(p_d.search(p_numUe.search(f_intf).group()).group())
        df.loc[df_index, 'Number of Interference Ue'] = numUe
        df_subj = pd.read_csv(folder + '/' + f_subj)
        subj_seq = set(df_subj['Seq number'].unique())
        count_seq = 0
        for e in subj_seq:
            if p_d.match(e):
                count_seq = count_seq + 1
        df_intf = pd.read_csv(folder + '/' + f_intf)
        intf_seq = set(df_intf['Seq number'].unique())
        df.loc[df_index, 'yhat'] = (len(intf_seq) / count_seq)
        df_index = df_index + 1
    df.to_csv(folder + '/' + 'data.csv', index=False)

def save_file(folder):
    df = pd.read_csv(folder + '/' + 'data.csv')
    x = df[df.columns[0]].to_numpy()
    yhat = df[df.columns[1]].to_numpy()
    np.save(folder[0: folder.rfind('/')] + '/' + 'x.npy', x.reshape(-1, 1))
    np.save(folder[0: folder.rfind('/')] + '/' + 'yhat.npy', yhat.reshape(-1, 1))

if __name__ == '__main__':
    folder = '/home/rui/work/prob-regression-model/data/2022-10-08/model-data'
    # print(folder[0: folder.rfind('/')])
    # gen_csv(folder)
    save_file(folder)