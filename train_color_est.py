
import os, sys
import json
import argparse
import random
import numpy as np
from comet_ml import Experiment, OfflineExperiment, Optimizer

import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.optim as optim

from networks import *
from utils import *


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        target = torch.mean(data, axis=(1, 2, 3)).clone().detach()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = F.mse_loss(output, target)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            experiment.log_metric('per_epoch_loss', loss, step=batch_idx)
            print("Epoch {}, Batch {}: Loss {:.3e}".format(epoch, batch_idx, loss.item()))

def test(model, device, test_loader, trainset=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data, _) in test_loader:

            target = torch.mean(data, axis=(1, 2, 3)).clone().detach()
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += F.mse_loss(output, target, reduction='sum')

    num_samples = len(test_loader.dataset)
    test_loss /= num_samples
    if trainset:
        print('\n({} samples) Trainset mean loss: {:.3e}\n'.format(num_samples, test_loss))
    else:
        print('\n({} samples) Testset mean loss: {:.3e}\n'.format(num_samples, test_loss))

    return test_loss


if __name__ == '__main__':

    # CLI Arguments
    parser = argparse.ArgumentParser()

    # must be specified through CLI
    parser.add_argument('-model', choices=['fc_1', 'cnn_1', 'cnn_2'], type=str, help='Which model to train')
    parser.add_argument('-exp_name', type=str, help='What to name this experiment')
    parser.add_argument('-scramble', action='store_true', help='Whether or not to scramble input images in each batch')
    parser.add_argument('-fixed_scramble', action='store_true', help='Whether or not to use fixed or random scrambling')
    parser.add_argument('-cuda', action='store_true', help='Whether you are using GPU or CPU')

    # fixed across sweeps
    parser.add_argument('-dataset', default='FashionMNIST', type=str, help='Which dataset to train on (FashionMNIST, Cifar10)')
    parser.add_argument('-log', default=False, type=bool, help='Whether or not to log terminal output')
    parser.add_argument('-single_run', default=False, type=bool, help='Whether this is a single run or part of a sweep')
    parser.add_argument('-epochs', default=20, type=int, help='Number of learning epochs')

    # variable across sweeps (overridable by Experiment() object)
    parser.add_argument('-lr', default=float(1e-2), type=float, help='The learning rate to use')
    parser.add_argument('-momentum', default=0.0, type=float, help='The momentum factor to use')
    parser.add_argument('-seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('-train_batch_size', default=64, type=int, help='Number of samples in a batch')
    parser.add_argument('-test_batch_size', default=1000, type=int, help='Number of samples in a test batch')
    args = parser.parse_args()

    # Set logging directory for the sweep
    if args.scramble == True:
        sweep_path = './results/{}/scrambled/{}/{}'.format(args.dataset, args.model, args.exp_name)
    else:
        sweep_path = './results/{}/unscrambled/{}/{}'.format(args.dataset, args.model, args.exp_name)
    
    # Define experiments to run
    if args.single_run:
        experiments = [Experiment(project_name=args.exp_name)]
    else:
        hp_sweeper = Optimizer(sweep_path + '/experiment.config', project_name=args.exp_name)
        experiments = hp_sweeper.get_experiments()

    # Go through each experiment
    for experiment in experiments:

        experiment.auto_metric_logging = False
        experiment.auto_param_logging = False
        exp_params = vars(args)

        if args.single_run == False:
            exp_params['lr'] = experiment.get_parameter('lr')
            exp_params['momentum'] = experiment.get_parameter('momentum')
            exp_params['seed'] = experiment.get_parameter('seed')
            exp_params['train_batch_size'] = experiment.get_parameter('train_batch_size')
            exp_params['test_batch_size'] = experiment.get_parameter('test_batch_size')

        experiment.log_parameters(exp_params)

        # Set logging directory for the experiment
        if args.log:
            run = len(os.listdir(sweep_path))-1
            logdir = sweep_path + '/run_{}'.format(run)
            experiment.set_name('run_{}'.format(run))
            create_path(logdir + '/weights')
            with open('{}/exp_params.txt'.format(logdir), 'w') as json_file:
                json_args = json.dumps(exp_params, indent=4)
                json_file.write(json_args)

        # Set random seed/device
        torch.manual_seed(exp_params['seed'])
        use_cuda = not args.cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # Dataset initialization
        if args.dataset == 'FashionMNIST':
            num_channels = 1
            if args.scramble:
                transforms_to_apply = transforms.Compose([ScrambleImg(fixed_scramble=args.fixed_scramble), transforms.ToTensor()])
            else:
                transforms_to_apply = transforms.Compose([transforms.ToTensor()])
            train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transforms_to_apply)
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transforms_to_apply)
        elif args.dataset == 'Cifar10':
            num_channels = 1
            if args.scramble:
                transforms_to_apply = transforms.Compose([transforms.Resize(28), transforms.Grayscale(), ScrambleImg(fixed_scramble=args.fixed_scramble), transforms.ToTensor()])
            else:
                transforms_to_apply = transforms.Compose([transforms.Resize(28), transforms.Grayscale(), transforms.ToTensor()])
            train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transforms_to_apply)
            test_dataset = datasets.CIFAR10('data', train=False, transform=transforms_to_apply)
        log_example_img(train_dataset, experiment, num_channels=num_channels, train=True)
        log_example_img(test_dataset, experiment, num_channels=num_channels, train=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=exp_params['train_batch_size'], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=exp_params['test_batch_size'], shuffle=True, **kwargs)

        # Model initialization
        if args.model == 'fc_1':
            model_params = {
                'fc_dim':int(28*28*num_channels)
            }
            model = FC_1(model_params).to(device)
        elif args.model == 'cnn_1':
            model_params = {
                'kernel':(3,3),
                'stride':1,
                'padding':0,
                'channels':num_channels,
                'fc_dim':int(26*26*num_channels)
            }
            model = CNN_1(model_params).to(device)
        elif args.model == 'cnn_2':
            model_params = {
                'kernel':(3,3),
                'stride':1,
                'padding':0,
                'pool_kernel':(2,2),
                'pool_stride':2,
                'pool_padding':0,
                'channels':num_channels,
                'fc_dim':int(13*13*num_channels)
            }
            model = CNN_2(model_params).to(device)

        if args.log:
            with open('{}/model_params.txt'.format(logdir), 'w') as model_file:
                json_model = json.dumps(model_params, indent=4)
                model_file.write(json_model)

        optimizer = optim.SGD(model.parameters(), lr=exp_params['lr'], momentum=exp_params['momentum'])
        train_losses = []
        test_losses = []
        for epoch in range(1, args.epochs + 1):

            # Train
            train(model, device, train_loader, optimizer, epoch)

            # Save weights
            if 'cnn' in args.model:
                if use_cuda:
                    fc_weights = model.fc[0].weight.detach().cpu().numpy()
                    conv_weights = model.conv[0].weight.detach().cpu().numpy()
                else:
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
                if use_cuda:
                    fc_weights = model.fc[0].weight.detach().cpu().numpy()
                else:
                    fc_weights = model.fc[0].weight.detach().numpy()
                print('\nFC weights:')
                print(fc_weights)
                if args.log:
                    np.save('{}/weights/fc_{}.npy'.format(logdir, epoch), fc_weights)

            # Test
            train_loss = test(model, device, train_loader, trainset=True)
            experiment.log_metric("train_loss", train_loss, step=epoch)
            train_losses.append(train_loss.item())
            test_loss = test(model, device, test_loader, trainset=False)
            experiment.log_metric("test_loss", test_loss, step=epoch)
            test_losses.append(test_loss.item())

        print('\n')
        print('#'*100)
        print('Losses on train set:')
        print(train_losses)
        print('#'*100)
        print('Losses on test set:')
        print('\n')
        print(test_losses)

        # Save model
        if args.log:
            torch.save(model.state_dict(), '{}/model.pt'.format(logdir))
            torch.save(optimizer.state_dict(), '{}/optimizer.pt'.format(logdir))
            np.save('{}/train_losses.npy'.format(logdir), np.array(train_losses))
            np.save('{}/test_losses.npy'.format(logdir), np.array(test_losses))





