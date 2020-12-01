import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-exp_id', default=1, type=int, help='Which experiment to search in')
args = parser.parse_args()

experiments = {
    1:'20epoch_lr_exps',
    2:'fashion_mnist_usc',
    3:'fashion_mnisc_sc'
}

logdir = './results'

if args.exp_id == 1:
    fc_path = '{}/{}/{}'.format(logdir, 'fc_1', experiments[args.exp_id])
    cnn1_path = '{}/{}/{}'.format(logdir, 'cnn_1', experiments[args.exp_id])
    cnn2_path = '{}/{}/{}'.format(logdir, 'cnn_2', experiments[args.exp_id])
    name = '1e-2'

    x = np.linspace(0, 20, 20)
    y_fc = np.load('{}/{}/test_losses.npy'.format(fc_path, name))
    y_cnn1 = np.load('{}/{}/test_losses.npy'.format(cnn1_path, name))
    y_cnn2 = np.load('{}/{}/test_losses.npy'.format(cnn2_path, name))

    plt.plot(x, y_fc, label='Fully connected')
    plt.plot(x, y_cnn1, label='CNN')
    plt.plot(x, y_cnn2, label='CNN maxpool')
    plt.title('Color Estimation (Fashion MNIST)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.show()

elif args.exp_id == 2:
    fc_path = '{}/{}/{}'.format(logdir, 'fc_1', experiments[args.exp_id])
    cnn1_path = '{}/{}/{}'.format(logdir, 'cnn_1', experiments[args.exp_id])
    cnn2_path = '{}/{}/{}'.format(logdir, 'cnn_2', experiments[args.exp_id])
    name = '1e-2'

    x = np.linspace(0, 100, 100)
    y_fc = np.load('{}/{}/test_losses.npy'.format(fc_path, name))
    y_cnn1 = np.load('{}/{}/test_losses.npy'.format(cnn1_path, name))
    y_cnn2 = np.load('{}/{}/test_losses.npy'.format(cnn2_path, name))

    plt.plot(x, y_fc, label='Fully connected')
    plt.plot(x, y_cnn1, label='CNN')
    plt.plot(x, y_cnn2, label='CNN maxpool')
    plt.title('Color Estimation (Fashion MNIST)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.ylim(0, 0.00008)
    plt.legend()
    plt.show()














