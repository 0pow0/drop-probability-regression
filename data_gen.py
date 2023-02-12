import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import argparse
import os
import re
import math

p_d = re.compile('\d+')
p_nUe_intf = re.compile('-\d+-intf*')
p_nUe_subj = re.compile('-\d+-subj*')
p_index = re.compile('^\d+')

def gen_csv(folder, output_path):
    df = pd.DataFrame()
    df["yhat"] = ""
    files = natsorted(os.listdir(folder))
    df_index = 0
    for i in tqdm(range(0, len(files), 2)):
        if p_nUe_intf.search(files[i]) != None:
            f_intf = files[i]
            f_subj = files[i+1]
        else:
            f_subj = files[i]
            f_intf = files[i+1]
        # print(f_intf, f_subj)
        assert(p_index.search(f_subj).group() == p_index.search(f_intf).group())
        idx = p_index.search(f_subj).group() 
        n_intf = int(p_d.search(p_nUe_intf.search(f_intf).group()).group())
        n_subj = int(p_d.search(p_nUe_subj.search(f_subj).group()).group())
        # print(n_intf, n_subj)
        df.loc[idx, 'Number of Interference Ue'] = n_intf 
        df.loc[idx, 'Number of Subject Ue'] = n_subj 

        df_subj = pd.read_csv(folder + '/' + f_subj)
        seq_set = set(df_subj['Seq number'])
        df_intf = pd.read_csv(folder + '/' + f_intf)
        intf_seq = set(df_intf['Seq number'])
        df.loc[idx, 'yhat'] = (len(intf_seq) / len(seq_set))
    print(df)
    x = df[["Number of Subject Ue", "Number of Interference Ue"]].to_numpy()
    yhat = df["yhat"].to_numpy()
    print(x, yhat)
    print(x.shape, yhat.shape)
    if not output_path.endswith('/'):
        output_path = output_path + '/'
    np.save(output_path + 'x.npy', x)
    np.save(output_path + 'yhat.npy', yhat)

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('output_path', type=str)
    return parser.parse_args()

def main():
    args = parse_argv()
    gen_csv(args.data_folder, args.output_path)

if __name__ == '__main__':
    main()