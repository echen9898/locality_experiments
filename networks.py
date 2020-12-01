
import torch.nn as nn

class FC_1(nn.Module):
    # One fully connected layer network
    def __init__(self, params):
        super(FC_1, self).__init__()
        self.params = params
        fc_dim = self.params['fc_dim']

        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 1),
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
        channels = self.params['channels']

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                      out_channels=channels, 
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
        channels = self.params['channels']

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                      out_channels=channels, 
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













