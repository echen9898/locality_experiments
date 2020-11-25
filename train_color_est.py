
import os, sys
import json
import torch
import torchvision
from torchvision import datasets, transforms, models
import shuffle_pipe
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import extarget
import network_family
import pandas as pd
from PIL import Image
import random
import argparse
import full_scramble
from utils import *


################################### NETWORKS ###################################
class FC_1(nn.Module):
    # One fully connected layer network
    def __init__(self, params):
        super(FC_1, self).__init__()
        self.params = params
        fc_dim = self.params['fc_dim']

        self.fc = nn.Sequential(
            # nn.Linear(fc_dim, 1),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU()
            )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_1(nn.Module):
    # One convolution network, no pooling
    def __init__(self, params):
        super(CNN_1, self).__init__()
        self.params = params
        kernel = self.params['kernel']
        stride = self.params['stride']
        padding = self.params['padding']
        fc_dim = self.params['fc_dim']

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=1, 
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 1),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_2(nn.Module):
    # One convolution network, with max pooling
    def __init__(self, params):
        super(CNN_2, self).__init__()
        self.params = params
        kernel = self.params['kernel']
        stride = self.params['stride']
        padding = self.params['padding']
        pool_kernel = self.params['pool_kernel']
        pool_stride = self.params['pool_stride']
        pool_padding = self.params['pool_padding']
        fc_dim = self.params['fc_dim']

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=1, 
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
            )
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 1),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


################################## TRAINING ###################################
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_num_correct = 0
    sum_loss = 0
    num_batches_since_log = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        
        data = torch.rand((64, 1, 5, 5))
        target = data.clone()
        # target = torch.mean(data, axis=(1, 2, 3)).clone().detach()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = F.mse_loss(output.view((64, 1, 5, 5)), target)
        # loss = F.mse_loss(output, target)
        sum_loss += loss.item()
        num_batches_since_log += 1
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            # print(output)
            # print('-'*100)
            # print(target)
            print("Epoch {}, Batch {}: Loss {:.3e}".format(epoch, batch_idx, loss.item()))
            # print('#'*100)

def test(model, device, test_loader, trainset=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data, _) in test_loader:

            data = torch.rand((64, 1, 5, 5))
            target = data.clone()
            # target = torch.mean(data, axis=(1, 2, 3)).clone().detach()
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()

            test_loss += F.mse_loss(output.view((64, 1, 5, 5)), target, reduction='sum')
            # test_loss += F.mse_loss(output, target, reduction='sum')

    num_samples = len(test_loader.dataset)
    test_loss /= num_samples
    if trainset:
        print('\n({} samples) Trainset mean loss: {:.3e}\n'.format(num_samples, test_loss))
    else:
        print('\n({} samples) Testset mean loss: {:.3e}\n'.format(num_samples, test_loss))
    return test_loss


if __name__ == '__main__':

        # [x] 5x5
        # [x] 8x8
        # [x] 12x12
        # [x] 20x20
        # [ ] 28x28
        # [ ] 40x40
        # [ ] 64X64
        # [ ] 100x100

    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', choices=['cnn_1', 'cnn_2', 'fc_1'], type=str, help='Which model to train')
    # parser.add_argument('-dataset', default='Random_8x8', type=str, help='Which dataset to train on (Fashion_MNIST, RandomSxS')
    parser.add_argument('-dataset', default='Random_5x5_identity', type=str, help='Which dataset to train on (Fashion_MNIST, RandomSxS')
    parser.add_argument('-epochs', default=20, type=int, help='Number of learning epochs')
    parser.add_argument('-lr', default=float(1e-3), type=float, help='The learning rate to use')
    parser.add_argument('-momentum', default=0.0, type=float, help='The momentum factor to use')
    parser.add_argument('-seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('-cuda', default=False, type=bool, help='Whether or not to use GPUs')
    parser.add_argument('-train_batch_size', default=64, type=int, help='Number of samples in a batch')
    parser.add_argument('-test_batch_size', default=1000, type=int, help='Number of samples in a test batch')
    parser.add_argument('-log', default=False, type=bool, help='Whether or not to log terminal output')
    args = parser.parse_args()

    # Logging
    if args.log:
        logpath = '/locality_experiments/results/{}'.format(args.model)
        run = len(os.listdir(logpath))-1
        logdir = logpath + '/run_{}'.format(run)
        os.makedirs(logdir + '/weights')
        with open('{}/arguments.txt'.format(logdir), 'w') as json_file:
            json_args = json.dumps(vars(args), indent=4)
            json_file.write(json_args)
        print('Training ...')
        f = open('{}/console.out'.format(logdir), 'w')
        sys.stdout = f

    # GPU Usage
    use_cuda = not args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seed
    torch.manual_seed(args.seed)

    # Dataset initialization
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Model initialization
    if args.model == 'fc_1':
        # params = {
        #     'fc_dim':int(28*28*1)
        # }
        params = {
            'fc_dim':int(5*5*1)
        }
        model = FC_1(params).to(device)
    elif args.model == 'cnn_1':
        # params = {
        #     'kernel':(3,3),
        #     'stride':1,
        #     'padding':0,
        #     'fc_dim':int(26*26*1)
        # }
        params = {
            'kernel':(3,3),
            'stride':1,
            'padding':0,
            'fc_dim':int(3*3*1) #k-2
        }
        model = CNN_1(params).to(device)
    elif args.model == 'cnn_2':
        # params = {
        #     'kernel':(3,3),
        #     'stride':1,
        #     'padding':0,
        #     'pool_kernel':(2,2),
        #     'pool_stride':2,
        #     'pool_padding':0,
        #     'fc_dim':int(13*13*1)
        # }
        params = {
            'kernel':(3,3),
            'stride':1,
            'padding':0,
            'pool_kernel':(2,2),
            'pool_stride':1,
            'pool_padding':0,
            'fc_dim':int(2*2*1) #k-3
        }
        model = CNN_2(params).to(device)

    if args.log:
        with open('{}/model_params.txt'.format(logdir), 'w') as model_file:
            json_model = json.dumps(params, indent=4)
            model_file.write(json_model)

    # Optimization loop
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_losses = []
    test_losses = []
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        if 'cnn' in args.model:
            fc_weights = model.fc[0].weight.detach().numpy()
            conv_weights = model.conv[0].weight.detach().numpy()
            print('\nConv kernel:')
            print(conv_weights)
            print('\nFC weights:')
            print(fc_weights)
            if args.log:
                np.save('{}/weights/conv_{}.npy'.format(logdir, epoch), conv_weights)
                np.save('{}/weights/fc_{}.npy'.format(logdir, epoch), fc_weights)
        elif 'fc' in args.model:
            fc_weights = model.fc[0].weight.detach().numpy()
            print('\nFC weights:')
            print(fc_weights)
            if args.log:
                np.save('{}/weights/fc_{}.npy'.format(logdir, epoch), fc_weights)
        train_loss = test(model, device, train_loader, trainset=True)
        test_loss = test(model, device, test_loader, trainset=False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print('\n')
    print('#'*100)
    print('Losses on train set:')
    print(train_losses)
    print('#'*100)
    print('Losses on test set:')
    print(test_losses)

    if args.log:
        torch.save(model.state_dict(), '{}/model.pt'.format(logdir))
        torch.save(optimizer.state_dict(), '{}/optimizer.pt'.format(logdir))
        np.save('{}/train_losses.npy'.format(logdir), np.array(train_losses))
        np.save('{}/test_losses.npy'.format(logdir), np.array(test_losses))
        f.close()

# s = 0.2249+0.1040+0.2230+0.1435-0.1813+0.1627+0.1949+0.1631+0.1907
# l = [0.1963, 0.1873, 0.1872, 0.1966]
# print(l[0]*s)








