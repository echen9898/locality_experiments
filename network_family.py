import torch.nn as nn
import torch.nn.functional as F

#####################
# Defining the CNN: #
#####################

class Small_Net(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Small_Net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,num_classes)
	#
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		x = self.fc1(x)
		return x

class Net(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(in_channels,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,num_classes)
	#
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Net_Gray(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Net_Gray,self).__init__()
		self.conv1 = nn.Conv2d(in_channels,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,num_classes)
	#
	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


################################
# Defining the Shallow Network #
################################

class Shallow_Net(nn.Module):
	# Has only one hidden layer
	def __init__(self,in_channels,num_classes):
		super(Shallow_Net,self).__init__()
		self.fc1 = nn.Linear(32*32*in_channels,10000)
		self.fc2 = nn.Linear(10000,num_classes)
	#
	def forward(self,x):
		x = x.view(-1,32*32*3)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class Shallow_Net_Gray(nn.Module):
	# Has only one hidden layer
	def __init__(self,in_channels,num_classes):
		super(Shallow_Net_Gray,self).__init__()
		self.fc1 = nn.Linear(32*32*in_channels,10000)
		self.fc2 = nn.Linear(10000,num_classes)
	#
	def forward(self,x):
		x = x.view(-1,32*32*1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


###########################
# Defining the FC Network #
###########################

class Net_FC(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Net_FC,self).__init__()
		self.fc1 = nn.Linear(32*32*3,3600)
		self.fc2 = nn.Linear(3600,100)
		self.fc3 = nn.Linear(100,num_classes)
	#
	def forward(self,x):
		x = x.view(-1,32*32*3)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Net_FC_Gray(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Net_FC_Gray,self).__init__()
		self.fc1 = nn.Linear(32*32*1,3600)
		self.fc2 = nn.Linear(3600,100)
		self.fc3 = nn.Linear(100,num_classes)
	#
	def forward(self,x):
		x = x.view(-1,32*32*1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

###############################
# Defining the Kernel Machine #
###############################

class Kernel_Machine(nn.Module):
	def __init__(self,in_channels,num_classes):
		super(Kernel_Machine,self).__init__()
		self.fc1 = nn.Linear(32*32*in_channels,num_classes)
	def forward(self,x):
		x = self.fc1(x)
		return x