import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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
import sys
import os
import random
import full_scramble
import utils

network_model = sys.argv[1]
recognition_task = sys.argv[2]
shuffle_start = int(sys.argv[3])
shuffle_end = int(sys.argv[4])
init_runs = int(sys.argv[5])
end_runs = int(sys.argv[6])
#num_epochs = int(sys.argv[7])

if network_model == 'FC_Single' or network_model == 'DenseNet121':
	num_epochs = 5
elif network_model == 'DenseNet121_20':
	num_epochs = 20
else:
	print('Error!')

device = 'cuda:0'
batch_size = 64
num_workers = 8
visualize = 0

#recognition_task = 'object' # Switch the DataLoaders to Scene if scene keyword added here!
#recognition_task = 'scene'

#shuffle_start = 1
#shuffle_end = 2

# Define Hyperparameters:
if network_model == 'LeNet' or network_model == 'KM' or network_model == 'FC_Multi' or network_model == 'FC_Single' or network_model == 'Qianli' or network_model == 'ResNet18' or network_model == 'Small_Net' or network_model == 'DenseNet121' or network_model == 'DenseNet121_20':
	learning_rate = 0.001
	momentum = 0.2

# Basic Transforms to normalize inputs
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
transform_Augment = transforms.Compose([torchvision.transforms.RandomResizedCrop((32,32),(0.7,1.0),),torchvision.transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
transform_MNIST = transforms.Compose([transforms.Resize([32,32]),transforms.ToTensor(),transforms.Normalize([0.5],[0.22])])
transform_Visualize = transforms.Compose([transforms.ToTensor()])

if recognition_task == 'object':
	in_channels = 3
	# Folder pre-amble
	folder_preamble = 'Object_Color_Estimation'
	#
	# Training Set [Shuffle Training Dataset]
	trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_Augment)
	trainloader = torch.utils.data.DataLoader(trainset,pin_memory=True,batch_size=batch_size,shuffle=True,num_workers=num_workers)
	#
	# Testing Set
	testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
	testloader = torch.utils.data.DataLoader(testset,pin_memory=True,batch_size=batch_size,shuffle=False,num_workers=num_workers)
	classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
	#
	Visualize_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_Visualize)
	Visualize_loader = torch.utils.data.DataLoader(Visualize_set,pin_memory=True,batch_size=batch_size,shuffle=False,num_workers=1)
elif recognition_task == 'scene_ten':
	in_channels = 3
	# Folder pre-amble
	folder_preamble = 'Scene_Ten_Color_Estimation'
	# Create other scene data loaders
	# Copy from paper Deza & Konkle.
	# Define classes:
	#
	class sceneTrainDataset(torch.utils.data.Dataset):
		def __init__(self, text_file, root_dir, transform=transform_Augment):
			self.name_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1))
			self.label_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1, 2))
			self.root_dir = root_dir
			self.transform = transform
		#
		def __len__(self):
			return len(self.name_frame)
		#
		def __getitem__(self, idx):
			img_name = self.name_frame.iloc[idx, 0]
			image = Image.open(img_name).convert('RGB')
			image = self.transform(image)
			labels = int(self.label_frame.iloc[idx, 0]) - 1
			return image, labels
	#
	class sceneValDataset(torch.utils.data.Dataset):
		def __init__(self, text_file, root_dir, transform=transform):
			self.name_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1))
			self.label_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1, 2))
			self.root_dir = root_dir
			self.transform = transform
		#
		def __len__(self):
			return len(self.name_frame)
		#
		def __getitem__(self, idx):
			img_name = self.name_frame.iloc[idx, 0]
			image = Image.open(img_name).convert('RGB')
			image = self.transform(image)
			labels = int(self.label_frame.iloc[idx, 0]) - 1
			return image, labels
	#
	class sceneTestDataset(torch.utils.data.Dataset):
		def __init__(self, text_file, root_dir, transform=transform):
			self.name_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1))
			self.label_frame = pd.read_csv(text_file, header=None, sep=" ", usecols=range(1, 2))
			self.root_dir = root_dir
			self.transform = transform

		#
		def __len__(self):
			return len(self.name_frame)
		#
		def __getitem__(self, idx):
			img_name = self.name_frame.iloc[idx, 0]
			image = Image.open(img_name).convert('RGB')
			image = self.transform(image)
			labels = int(self.label_frame.iloc[idx, 0]) - 1
			return image, labels
	#
	#
	# Load Training Dataset: [Change these directories]:
	trainset_name = '../Dataset_Files/Training_Scenes_Ten.txt'
	valset_name = '../Dataset_Files/Validation_Scenes_Ten.txt'
	testset_name = '../Dataset_Files/Testing_Scenes_Ten.txt'
	#
	trainset = sceneTrainDataset(text_file=trainset_name, root_dir='.', transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
											  num_workers=num_workers, pin_memory=True)
	#
	valset = sceneValDataset(text_file=valset_name, root_dir='.', transform=transform)
	valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True,
											num_workers=num_workers, pin_memory=True)
	#
	testset = sceneTestDataset(text_file=testset_name, root_dir='.', transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,
											 num_workers=num_workers, pin_memory=True)
	#
	Visualize_set = sceneTestDataset(text_file=testset_name, root_dir='.', transform=transform_Visualize)
	Visualize_loader = torch.utils.data.DataLoader(Visualize_set, pin_memory=True, batch_size=batch_size, shuffle=False,
												   num_workers=1)
	classes = ('ocean','industrial_area','badlands','bedroom','bridge','forest_path','kitchen','office','corridor','mountain')

def imshow(img):
	img = img/2+0.5 # un-normalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0))) # flip BGR to RGB colors
	plt.show()

def imshow_regular(img):
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0))) # flip BGR to RGB colors
	plt.show()

hierarchical_scramble = 0
scramble_flag = 0
#batch_size = 1
if visualize == 1:
	# get some random training images:
	dataiter = iter(Visualize_loader)
	images, labels = dataiter.next()
	if scramble_flag == 1:
		# import shuffle_pipe and demo image shuffling
		if hierarchical_scramble == 1:
			shuffle_list = [1]
			images_transpose = shuffle_pipe.transpose_from_cnn_to_standard_format(images)
			shuffled_images = shuffle_pipe.shuffle_at_scales(shuffle_list, np.array(images_transpose))
			shuffled_images_back = shuffle_pipe.transpose_from_standard_to_cnn_format(shuffled_images)
			images_new = shuffled_images_back
		elif hierarchical_scramble == 0:
			# Or full scramble:
			scramble_indx = np.arange(32*32)
			random.shuffle(scramble_indx)
			images_new = full_scramble.full_scramble(images,batch_size,scramble_indx)
		else:
			print('Error!')
	elif scramble_flag == 0:
		images_new = images
	# show images
	#imshow(torchvision.utils.make_grid(images))
	# show images
	imshow_regular(torchvision.utils.make_grid(torch.Tensor(images_new)))
	#print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#optimizer = optim.SGD(net.parameters(),lr=learning_rate, momentum=momentum)

#shuffle_num = 1 # goes from 1 to 6 for CIFAR-10 that has 32x32x3 dimensionality
#shuffle_list = []
#for i in range(shuffle_num):
#	shuffle_list.append(1)
#shuffle_num_str = str(shuffle_num)

for shuffle_num in range(shuffle_start,shuffle_end+1):
	shuffle_list = []
	for i in range(shuffle_num):
		shuffle_list.append(1)
	shuffle_num_str = str(shuffle_num)
	#
	for run_id in range(init_runs,end_runs + 1):
		if shuffle_num == 7:
			shuffle_indx = np.arange(32*32)
			random.shuffle(shuffle_indx)
		run_id_str = str(run_id)
		# Define Networks:
		if network_model == 'Pure_Conv':
			net = network_family.Pure_Conv(in_channels,3)
		elif network_model == 'LeNet':
			if in_channels == 3:
				net = network_family.Net(in_channels,3)
			elif in_channels == 1:
				net = network_family.Net_Gray(in_channels,3)
		elif network_model == 'FC_Multi':
			net = network_family.Net_FC(in_channels,3)
		elif network_model == 'FC_Single':
			if in_channels == 3:
				net = network_family.Shallow_Net(in_channels,3)
			elif in_channels == 1:
				net = network_family.Shallow_Net_Gray(in_channels,3)
		elif network_model == 'Qianli':
			if in_channels == 3:
				net = extarget.NetSimpleConv(32,3)
			elif in_channels == 1:
				net = extarget.NetSimpleConv_Gray(32,3)
		elif network_model == 'ResNet18':
			resnet18 = models.resnet18(pretrained=False)
			net = resnet18
			net.fc = nn.Linear(512, 3)
		elif network_model == 'Small_Net':
			net = network_family.Small_Net(in_channels,3)
		elif network_model == 'DenseNet121':
			densenet = models.densenet121(pretrained=False)
			net = densenet
			net.classifier = nn.Linear(1024,3)
		elif network_model == 'DenseNet121_20':
			densenet = models.densenet121(pretrained=False)
			net = densenet
			net.classifier = nn.Linear(1024,3)
		#
		net.to(device)
		criterion = nn.MSELoss()
		#
		# Train the Network:
		for epoch in range(num_epochs):
			running_loss = 0.0
			total_loss = 0.0
			epoch_str = str(epoch)
			if epoch <= 20:
				optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
			elif epoch <= 40:
				optimizer = optim.SGD(net.parameters(), lr=learning_rate*0.5, momentum=0.9)
			elif epoch <= 80:
				optimizer = optim.SGD(net.parameters(), lr=learning_rate*0.1, momentum=0.9)
			else:
				print('Error!')
			#
			for i, data in enumerate(trainloader,0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				# shuffle stimuli:
				if shuffle_num == 7:
					new_images = full_scramble.full_scramble(inputs,len(labels),shuffle_indx)
					new_images = torch.tensor(new_images,dtype=torch.float)
					inputs, labels = new_images.to(device), labels.to(device)
				else:
					np.random.seed(3)
					images_transpose = shuffle_pipe.transpose_from_cnn_to_standard_format(np.array(inputs))
					shuffled_images = shuffle_pipe.shuffle_at_scales(shuffle_list, np.array(images_transpose))
					shuffled_images_back = torch.Tensor(shuffle_pipe.transpose_from_standard_to_cnn_format(shuffled_images))
					# Map inputs and labels to respective devices:
					inputs, labels = shuffled_images_back.to(device), labels.to(device)
				#
				# zero the parameter gradients
				optimizer.zero_grad()
				# Steps for Pure
				# forward + backward + optimize
				if network_model == 'Qianli':
					outputs = net(inputs, detached=False)
				else:
					outputs = net(inputs)
				# Estimate Average Color and compute loss:
				images_reshape = torch.reshape(inputs, (len(labels), 3, 32 * 32))
				images_1 = torch.mean(images_reshape[:, 0, :], 1)
				images_2 = torch.mean(images_reshape[:, 1, :], 1)
				images_3 = torch.mean(images_reshape[:, 2, :], 1)
				#
				color_GT = torch.empty((len(labels), 3))
				color_GT[:, 0] = images_1
				color_GT[:, 1] = images_2
				color_GT[:, 2] = images_3
				#
				color_GT = color_GT.to(device)
				loss = criterion(outputs, color_GT)
				loss.backward()
				optimizer.step()
				#
				# print statistics
				#
				running_loss += loss.item()
				total_loss += loss.item()
				#
				if i % 20 == 19: # print every 2000 mini-batches
					print('[%d,%5d] %s Loss: %.3f' % (epoch+1, i+1,network_model,running_loss/20.0))
					running_loss = 0.0

			# Save total loss
			PATH_Loss = './results/' + folder_preamble + '/loss/training/' + network_model + '/' + run_id_str
			create_path(PATH_Loss)
			np.save('{}/Shuffle_{}_{}.npy'.format(PATH_Loss, shuffle_num_str, epoch_str), total_loss)
			# Save Network check points:
			PATH_Network = './results/' + folder_preamble + '/networks/' + network_model + '/' + run_id_str
			create_path(PATH_Network)
			torch.save(net.state_dict(), '{}/Shuffle_{}_{}.pth'.format(PATH_Network, shuffle_num_str, epoch_str))
		print('Finished Training run {}'.format(run_id_str))

		##########################################
		# Now make predictions on whole dataset: #
		##########################################
		MSE_Total = 0
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				#scramble at testing time:
				if shuffle_num == 7:
					new_images = full_scramble.full_scramble(images,len(labels),shuffle_indx)
					new_images = torch.tensor(new_images,dtype=torch.float)
					images, labels = new_images.to(device), labels.to(device)
				else:
					np.random.seed(3)
					images_transpose = shuffle_pipe.transpose_from_cnn_to_standard_format(np.array(images))
					shuffled_images = shuffle_pipe.shuffle_at_scales(shuffle_list, np.array(images_transpose))
					shuffled_images_back = torch.Tensor(shuffle_pipe.transpose_from_standard_to_cnn_format(shuffled_images))
					# Map inputs and labels to respective devices:
					images, labels = shuffled_images_back.to(device), labels.to(device)
				# compute outputs for Pure:
				if network_model == 'Qianli':
					outputs = net(images,detached=False)
				else:
					outputs = net(images)
				#
				images_reshape = torch.reshape(images, (len(labels), 3, 32 * 32))
				images_1 = torch.mean(images_reshape[:, 0, :], 1)
				images_2 = torch.mean(images_reshape[:, 1, :], 1)
				images_3 = torch.mean(images_reshape[:, 2, :], 1)
				#
				color_GT = torch.empty((len(labels), 3))
				color_GT[:, 0] = images_1
				color_GT[:, 1] = images_2
				color_GT[:, 2] = images_3
				color_GT = color_GT.to(device)
				#
				MSE_Total += len(labels)*criterion(color_GT.to(device),outputs.to(device)).cpu().detach().numpy()
		#
		MSE_Total = MSE_Total * 255.0 / 10000.0
		#
		print('MSE of Color Estimation-run_' + run_id_str + '_Shuffle_ ' + shuffle_num_str + ' on the 10000 test images: %f %%' % (MSE_Total))
		# Save the Accuracies:
		PATH_Accuracy = './results/' + folder_preamble + '/accuracies/' + network_model +'/' + run_id_str
		create_path(PATH_Accuracy)
		np.save(PATH_Accuracy '{}/Shuffle_{}.npy'.format(PATH_Accuracy, shuffle_num_str), MSE_Total)














