import os, sys
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-exp', default=1, type=int, help='Which experiment to search in')
parser.add_argument('-loss', default='test', type=str, help='Which loss to plot (train or test)')
parser.add_argument('-sweep', default='sweep3', type=str, help='Which sweep to pull runs from')
args = parser.parse_args()

logdir = './results'
experiments = {
    1:'FashionMNIST/unscrambled',
    2:'FashionMNIST/scrambled',
    3:'Cifar10/unscrambled',
    4:'Cifar10/scrambled'
}

fc_path = '{}/{}/{}/{}'.format(logdir, experiments[args.exp], 'fc_1', args.sweep)
cnn1_path = '{}/{}/{}/{}'.format(logdir, experiments[args.exp], 'cnn_1', args.sweep)
cnn2_path = '{}/{}/{}/{}'.format(logdir, experiments[args.exp], 'cnn_2', args.sweep)

x = np.linspace(0, 100, 100)
fc_df = pd.DataFrame({'Epoch':x})
cnn1_df = pd.DataFrame({'Epoch':x})
cnn2_df = pd.DataFrame({'Epoch':x})

regex = re.compile("run_*")

fc_runs = [run for run in os.listdir(fc_path) if regex.match(run)]
cnn1_runs = [run for run in os.listdir(cnn1_path) if regex.match(run)]
cnn2_runs = [run for run in os.listdir(cnn2_path) if regex.match(run)]

print("FC-1:")
print(fc_runs)
for run in fc_runs:
    run_path = fc_path + '/{}'.format(run)
    tmp = np.load(run_path + '/{}_losses.npy'.format(args.loss), allow_pickle=True)
    for i in range(len(tmp)):
        tmp[i] = tmp[i].item()
    fc_df['{}'.format(run)] = tmp
fc_df = fc_df.astype(float)
fc_df = pd.melt(fc_df, ['Epoch'])
sn.lineplot(x='Epoch', 
            y='value',
            data=fc_df,
            err_style='band', 
            ci='sd')

print("CNN-1:")
print(cnn1_runs)
for run in cnn1_runs:
    run_path = cnn1_path + '/{}'.format(run)
    tmp = np.load(run_path + '/{}_losses.npy'.format(args.loss), allow_pickle=True)
    for i in range(len(tmp)):
        tmp[i] = tmp[i].item()
    cnn1_df['{}'.format(run)] = tmp
cnn1_df = cnn1_df.astype(float)
cnn1_df = pd.melt(cnn1_df, ['Epoch'])
sn.lineplot(x='Epoch', 
            y='value',
            data=cnn1_df,
            err_style='band', 
            ci='sd')

print("CNN-2:")
print(cnn2_runs)
for run in cnn2_runs:
    run_path = cnn2_path + '/{}'.format(run)
    tmp = np.load(run_path + '/{}_losses.npy'.format(args.loss), allow_pickle=True)
    for i in range(len(tmp)):
        tmp[i] = tmp[i].item()
    cnn2_df['{}'.format(run)] = tmp
cnn2_df = cnn2_df.astype(float)
cnn2_df = pd.melt(cnn2_df, ['Epoch'])

sn.lineplot(x='Epoch', 
            y='value',
            data=cnn2_df,
            err_style='band', 
            ci='sd')

plt.ylim(0.0, 0.0001)
plt.ylabel('Test loss')

plt.show()





