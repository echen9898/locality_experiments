import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
#from .optimizer import Optimizer, required


def sign_all_grads(parameters):  
    for pram in parameters:
        #import pdb; pdb.set_trace() 
        if pram.grad is not None:
            pram.grad = 0.1 * pram.grad.sign().float()   

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        featIn = 3
        featOut = 16
        self.conv12 = nn.Conv2d(featIn, featOut, 3, stride=2, padding=1)     
        featIn = featOut; featOut = featOut*2
        self.bn23   = nn.BatchNorm2d(featIn, affine=False)
        self.conv23 = nn.Conv2d(featIn, featOut, 3, stride=2, padding=1)
        featIn = featOut; featOut = featOut*2
        self.bn34   = nn.BatchNorm2d(featIn, affine=False)
        self.conv34 = nn.Conv2d(featIn, featOut, 3, stride=2, padding=1)
        featIn = featOut; featOut = featOut*2
        self.bn45   = nn.BatchNorm2d(featIn, affine=False)
        self.conv45 = nn.Conv2d(featIn, featOut, 3, stride=2, padding=1)
        #self.fc1 = nn.Linear(featOut , featOut)
        featIn = featOut; featOut = 10
        self.fc1 = nn.Linear(featIn , featOut)
        #self.fc2 = nn.Linear(featIn , featOut)

        self.fc1.target_layer = True

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv23(F.relu(self.bn23(x)))
        x = self.conv34(F.relu(self.bn34(x)))
        x = self.conv45(F.relu(self.bn45(x)))
        # x = self.conv23(F.relu(x))
        # x = self.conv34(F.relu(x))
        # x = self.conv45(F.relu(x))
        self.h1 = x
        #x = self.pool(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        self.h2 = x 
        x = self.fc1(x)
        #x = self.fc2(F.relu(x))
        return x



def subsample_2x_identity(x,featFactor=2):
    x = F.avg_pool2d(x, 2, stride=2)
    x = x.repeat([1,featFactor,1,1])
    return x

def match_target_to_x_shape(x,target):
    featFactor = float( x.size(1) / target.size(1) )  
    #import pdb; pdb.set_trace()
    if featFactor > 1:
        target = target.repeat([1,int(featFactor)])
    elif featFactor < 1:
        tsize_ = target.size
        target = target.view(tsize_(0), -1, int(tsize_(1)*featFactor))
        target = torch.mean(target,1) # dim will be discarded
        #target = target.squeeze(1)
    x_size = x.size()
    if len(x_size) == 4:
        target = target.unsqueeze(2).unsqueeze(3)        
    # if x_size(2) > 1 or x_size(3) > 1:
    #     target = target.repeat([1,1,x_size(2),x_size(3)])
    return target
    

import random

class Net1(nn.Module):
    #def __init__(self):        
    def __init__(self,  hidden_size, numClass):
        super(Net1, self).__init__() 
        self.hidden_size = hidden_size
        featOut = hidden_size
        self.conv1 = SubBRC(3,featOut, kerSize=32, stride=1, padding=0, detached=False, has_nonlinear=0, has_bn=0)    
        #self.conv1 = SubBRC(3,featOut, kerSize=3, stride=2, padding=0, detached=False, has_nonlinear=0, has_bn=0)        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)

        #self._random_proj_final_in = nn.Parameter(0.1*torch.randn(featIn,featIn), requires_grad=False)      
        
        self.criterion = nn.CrossEntropyLoss()
        #self._random_proj_ = nn.Parameter(0.05*torch.randn(numClass,hidden_size), requires_grad=False)
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)         
        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim

        self._random_proj_2nd = nn.Parameter(0.1*torch.randn(hidden_size,hidden_size), requires_grad=True)
        
        self.target_bn  = nn.BatchNorm2d(hidden_size, affine=False) 
        
    def forward(self, x, detached):
        self.conv_out = []
        x = self.conv1(x)
        self.conv_out.append(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        x = self.fc1(F.relu(x))   
        #x = self.fc1(x)       
        self.final_out = x
        return x

    def forward_h1_target(self, x, labels):
        detached = True 
        numClass =  self.numClass
        target, y_onehot = self.compute_target( labels)  
        self.conv_out = []
        x = self.conv1(x)
        # self.conv_out.append(x)
        # import pdb; pdb.set_trace()      
        # x = x*0 + self.nnsoftmax_layer( target.unsqueeze(2).unsqueeze(3) )
        x = x*0 +  target.unsqueeze(2).unsqueeze(3)   
        #x = x*0 +  target.unsqueeze(2).unsqueeze(3)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        x = self.fc1(x)
        self.final_out = x
        return x
    
    def forward_h1_target_2(self, x, labels):
        detached = True 
        numClass =  self.numClass
        target, y_onehot = self.compute_target( labels)  
        self.conv_out = []
        x = self.conv1(x)
        #self.conv_out.append(x)
        #x = x*0 + target
        #import pdb; pdb.set_trace()    
        x = x*0 + y_onehot.unsqueeze(2).unsqueeze(3)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        x = self.fc1(x)
        self.final_out = x
        return x

    def accGrad_bp(self, labels):
        loss = self.criterion(self.final_out, labels)
        loss.backward()        

    # def accGrad_bp_from_last_conv_out(self, labels):
    #     loss = self.criterion(self.final_out, labels)
    #     loss.backward()        
    # def accGrad_perceptron(self, labels):   
    #     for i in range(0,len(self.conv_out)):
    #         self.conv_out[i].backward(self.conv_out[i].grad)
    #     loss.backward()        

    def compute_target(self, labels):
        numClass = self.numClass
        device__ = next(self.parameters()).device
        y_onehot = torch.FloatTensor(len(labels), numClass).to(device__)        
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(-1,1), 1.0)
        target = torch.mm(y_onehot, self._random_proj_).data
        y_onehot = y_onehot
        return (target, y_onehot)
        #nnsoftmax_layer = nn.Softmax(1) # 2nd dim        

    def compute_target_with_fc1(self, labels):
        numClass = self.numClass
        device__ = next(self.parameters()).device
        y_onehot = torch.FloatTensor(len(labels), numClass).to(device__)        
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(-1,1), 1.0)
        target = torch.mm(y_onehot, self.fc1.weight).data
        y_onehot = y_onehot
        return (target, y_onehot)
        #nnsoftmax_layer = nn.Softmax(1) # 2nd dim        

    def compute_target_bn(self, labels):
        numClass = self.numClass
        device__ = next(self.parameters()).device
        y_onehot = torch.FloatTensor(len(labels), numClass).to(device__)        
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(-1,1), 1.0)
        target = torch.mm(y_onehot, self._random_proj_).data
        target = self.target_bn(target.unsqueeze(2).unsqueeze(3))
        target = target.squeeze(3).squeeze(2)       
        y_onehot = y_onehot
        return (target, y_onehot)
        #nnsoftmax_layer = nn.Softmax(1) # 2nd dim        

    def eval_layer_loss(self,labels):   
        target, y_onehot = self.compute_target(labels)
        conv_out_loss = []
        conv_out_loss_sign = []
        for x in self.conv_out:
            target_matched = match_target_to_x_shape(x,target)
            layer_loss = torch.mean((x - target_matched).pow(2)) 
            layer_loss_sign = torch.mean( (x.sign() == target_matched.sign()).float() ) 
            #x = conv_out_new
            conv_out_loss.append(layer_loss)
            conv_out_loss_sign.append(layer_loss_sign)
        conv_out_loss = torch.Tensor(conv_out_loss)
        conv_out_loss_sign = torch.Tensor(conv_out_loss_sign)
        return (conv_out_loss,conv_out_loss_sign)
    

def pool_and_rep(x,rep_factor=2,pool_factor=2):
    #x = F.avg_pool2d(x, [pool_factor, pool_factor])
    if pool_factor > 1:
        x = F.avg_pool2d(x,  pool_factor, stride=pool_factor)
    if rep_factor > 1:
        x = x.repeat([1,rep_factor,1,1])
    return x


def freeze_nnmodule(model):
    #model.isFrozen = True
    for param in model.parameters():
        param.requires_grad = False
        

class Net3fc(Net1):
    def __init__(self,  hidden_size, numClass):
        # import pdb; pdb.set_trace()      
        super(Net3fc, self).__init__( hidden_size, numClass )      
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubRCB(featIn,featOut, kerSize=32, stride=1, padding=0, detached=False, has_nonlinear=0, has_bn=1)        
        featIn = featOut
        featOut = featOut
        self.conv2_sub = SubRCB(featIn,featOut, kerSize=1, stride=1, padding=0, detached=False, has_nonlinear=1, has_bn=1)
        # freeze_nnmodule(self.conv1_sub)
        
        # featIn = featOut
        # featOut = featOut*2        
        # self.conv3_sub = SubRCB(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)
        # featIn = featOut
        # featOut = featOut*2        
        # self.conv4_sub = SubRCB(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)
        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached
        # import pdb; pdb.set_trace()  
        self.conv_out = [] 
        x = self.conv1_sub(x)
        self.conv_out.append(x)                
         
        #import pdb; pdb.set_trace()      
        for i in range(0,1):  
            if detached is True:
                x = torch.autograd.Variable(x.data,requires_grad=False)
            conv_out_new =  self.conv2_sub(x)
            #x = conv_out_new * 0.1 +  pool_and_rep(x,rep_factor=2)
            #x = conv_out_new*0.3 + x
            x = conv_out_new*0.1 + x
            #x =  conv_out_new + x  
            # for jj in self.conv_out:
            #     x = x + jj.data
            self.conv_out.append(x) 
        
        # if detached is True:
        #     x = torch.autograd.Variable(x.data,requires_grad=False)                            
        # x = self.conv3_sub(x)            
        # self.conv_out.append(x)
        
        # if detached is True:
        #     x = torch.autograd.Variable(x.data,requires_grad=False)        
        # x = self.conv4_sub(x)
        # self.conv_out.append(x)        
        
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)

        self.final_in_before_detach = x
        
        if detached is True or detach_last is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
            x = torch.autograd.Variable(x.data,requires_grad=True)
        self.final_in = x
        x = self.fc1(F.relu(x))    
        #x = self.fc1(x)   
        self.final_out = x
        
        return x


class Net3fcBRC(Net1):
    def __init__(self,  hidden_size, numClass):
        # import pdb; pdb.set_trace()      
        super(Net3fcBRC, self).__init__( hidden_size, numClass )          
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubRCB(featIn,featOut, kerSize=32, stride=1, padding=0, detached=False, has_nonlinear=0, has_bn=0)        
        featIn = featOut
        featOut = featOut
        self.conv2_sub = SubRCB(featIn,featOut, kerSize=1, stride=1, padding=0, detached=False, has_nonlinear=1, has_bn=1)
        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached
        # import pdb; pdb.set_trace()         
        self.conv_out = [] 
        x = self.conv1_sub(x)
        self.conv_out.append(x)                     
        #import pdb; pdb.set_trace()      
        for i in range(0,5):
            if detached is True:
                x = torch.autograd.Variable(x.data,requires_grad=False)
            conv_out_new =  self.conv2_sub(x)
            #x = conv_out_new + x
            x = conv_out_new*0.1 + x 
            #x =  conv_out_new    
            self.conv_out.append(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        if detached is True or detach_last is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
            x = torch.autograd.Variable(x.data,requires_grad=True) # requires grad for final_in
        
        self.final_in = x
        x = self.fc1(F.relu(x))      
        #x = self.fc1(x)        
        self.final_out = x        
        return x 
 

class Net3Simple(Net1): 
    def __init__(self,  hidden_size, numClass):
        # import pdb; pdb.set_trace()      
        super(Net3Simple, self).__init__( hidden_size, numClass )      
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0, has_bn=0)
        # freeze_nnmodule(self.conv1_sub)   
        
        featIn = featOut
        featOut = featOut*2        
        self.conv2_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)  
        
        featIn = featOut
        featOut = featOut*2        
        self.conv3_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)

        featIn = featOut
        featOut = featOut*2        
        self.conv4_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)
        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached
        # import pdb; pdb.set_trace()         
        self.conv_out = []  
        x = self.conv1_sub(x)
        self.conv_out.append(x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        conv_out_new =  self.conv2_sub(x)
        x = conv_out_new + pool_and_rep(x,rep_factor=2)   
        #x = conv_out_new + pool_and_rep(x,rep_factor=2,pool_factor=1)           
        #x = conv_out_new + x         
        self.conv_out.append(x) 
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)                            
        x = self.conv3_sub(x) + pool_and_rep(x,rep_factor=2)
        #x = self.conv3_sub(x)
        self.conv_out.append(x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)        
        x = self.conv4_sub(x) + pool_and_rep(x,rep_factor=2)     
        #x = self.conv4_sub(x)
        self.conv_out.append(x)        
        
        # import pdb; pdb.set_trace()            
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        
        if detached is True or detach_last is True:    
            x = torch.autograd.Variable(x.data,requires_grad=False)
            x = torch.autograd.Variable(x.data,requires_grad=True) # requires grad for final_in
        self.final_in = x
        
        x = self.fc1(F.relu(x))
        self.final_out = x
        return x



class NetSimpleConv(Net1): 
    def __init__(self,  hidden_size, numClass):
        # import pdb; pdb.set_trace()      
        super(NetSimpleConv, self).__init__( hidden_size, numClass )      
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0, has_bn=0)
        # freeze_nnmodule(self.conv1_sub)   
        
        featIn = featOut
        featOut = featOut*2        
        self.conv2_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)  
        
        featIn = featOut
        featOut = featOut*2        
        self.conv3_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)

        featIn = featOut
        featOut = featOut*2        
        self.conv4_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)

        featIn = featOut
        featOut = numClass        
        self.conv5_sub = SubBRC(featIn,featOut, kerSize=2, stride=1, padding=0, detached=False, has_nonlinear=0, has_bn=0)   
        
        #featIn = featOut; featOut = numClass
        #self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached

        x = self.conv1_sub(x)        
        
        x =  self.conv2_sub(x)
        
        x = self.conv3_sub(x)

        x = self.conv4_sub(x)
        
        x = self.conv5_sub(x)

        #import pdb; pdb.set_trace() 
         
        x = x.squeeze(3).squeeze(2)
                
        self.final_out = x # save the output for computing the loss 
        
        return x


class NetSimpleConv_Gray(Net1):
    def __init__(self, hidden_size, numClass):
        # import pdb; pdb.set_trace()
        super(NetSimpleConv_Gray, self).__init__(hidden_size, numClass)
        self.hidden_size = hidden_size
        featIn = 1;
        featOut = hidden_size
        self.conv1_sub = SubBRC(featIn, featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0,
                                has_bn=0)
        # freeze_nnmodule(self.conv1_sub)

        featIn = featOut
        featOut = featOut * 2
        self.conv2_sub = SubBRC(featIn, featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1,
                                has_bn=1)

        featIn = featOut
        featOut = featOut * 2
        self.conv3_sub = SubBRC(featIn, featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1,
                                has_bn=1)

        featIn = featOut
        featOut = featOut * 2
        self.conv4_sub = SubBRC(featIn, featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1,
                                has_bn=1)

        featIn = featOut
        featOut = numClass
        self.conv5_sub = SubBRC(featIn, featOut, kerSize=2, stride=1, padding=0, detached=False, has_nonlinear=0,
                                has_bn=0)

        # featIn = featOut; featOut = numClass
        # self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5 * torch.randn(numClass, hidden_size), requires_grad=False)
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1)  # 2nd dim

    def forward(self, x, detached, detach_last=None):
        if detach_last is None:
            detach_last = detached

        x = self.conv1_sub(x)

        x = self.conv2_sub(x)

        x = self.conv3_sub(x)

        x = self.conv4_sub(x)

        x = self.conv5_sub(x)

        # import pdb; pdb.set_trace()

        x = x.squeeze(3).squeeze(2)

        self.final_out = x  # save the output for computing the loss

        return x


class Net3Simple_dropout(Net1):   
    def __init__(self,  hidden_size, numClass, dropout_ratio=0.5):
        # import pdb; pdb.set_trace()      
        super(Net3Simple_dropout, self).__init__( hidden_size, numClass )      
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0, has_bn=0)
        # freeze_nnmodule(self.conv1_sub)   
        
        featIn = featOut
        featOut = featOut*2        
        self.conv2_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)  
        
        featIn = featOut
        featOut = featOut*2        
        self.conv3_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)

        featIn = featOut
        featOut = featOut*2        
        self.conv4_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=1, has_bn=1)
        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim

        self.dropout = []
        for i in range(0,4):
            self.dropout.append( torch.nn.Dropout(p=dropout_ratio) )                          
        
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached
        # import pdb; pdb.set_trace()         
        self.conv_out = []  
        x = self.conv1_sub(x)
        self.conv_out.append(x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        conv_out_new =  self.conv2_sub(x)
        x = conv_out_new + pool_and_rep(x,rep_factor=2)   
        #x = conv_out_new + pool_and_rep(x,rep_factor=2,pool_factor=1)           
        #x = conv_out_new + x         
        self.conv_out.append(x) 

        x = self.dropout[0](x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)                            
        x = self.conv3_sub(x) + pool_and_rep(x,rep_factor=2)
        #x = self.conv3_sub(x)
        self.conv_out.append(x)

        x = self.dropout[1](x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)        
        x = self.conv4_sub(x) + pool_and_rep(x,rep_factor=2)     
        #x = self.conv4_sub(x)
        self.conv_out.append(x)        

        x = self.dropout[2](x)
         
        # import pdb; pdb.set_trace()            
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        
        if detached is True or detach_last is True:    
            x = torch.autograd.Variable(x.data,requires_grad=False)
            x = torch.autograd.Variable(x.data,requires_grad=True) # requires grad for final_in
        self.final_in = x
        
        x = self.fc1(F.relu(x))
        self.final_out = x
        return x


class Net3Simple_nonlinear(Net1): 
    def __init__(self,  hidden_size, numClass, nonlinear_func=torch.nn.functional.relu, last_nonlin=None):
        # import pdb; pdb.set_trace()      
        super(Net3Simple_nonlinear, self).__init__( hidden_size, numClass )           
        self.nonlinear_func =  nonlinear_func
        if last_nonlin is None:
            self.last_nonlin = nonlinear_func
        else:
            self.last_nonlin = last_nonlin
        self.hidden_size = hidden_size
        featIn = 3; featOut = hidden_size
        self.conv1_sub = SubBRC_nonlinear(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False,  nonlinear_func=None, has_bn=0)
        # freeze_nnmodule(self.conv1_sub)   
        
        featIn = featOut
        featOut = featOut*2        
        self.conv2_sub = SubBRC_nonlinear(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False,  nonlinear_func=nonlinear_func, has_bn=1)  
        
        featIn = featOut
        featOut = featOut*2                  
        self.conv3_sub = SubBRC_nonlinear(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False,  nonlinear_func=nonlinear_func, has_bn=1)
         
        featIn = featOut
        featOut = featOut*2        
        self.conv4_sub = SubBRC_nonlinear(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False,  nonlinear_func=nonlinear_func, has_bn=1)            
        
        featIn = featOut; featOut = numClass
        self.fc1 = nn.Linear(featIn , featOut)
        self.criterion = nn.CrossEntropyLoss()
        self._random_proj_ = nn.Parameter(0.5*torch.randn(numClass,hidden_size), requires_grad=False)        
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim
        
    def forward(self, x, detached, detach_last=None):
        if  detach_last is None:
            detach_last = detached
        # import pdb; pdb.set_trace()         
        self.conv_out = []  
        x = self.conv1_sub(x)
        self.conv_out.append(x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        conv_out_new =  self.conv2_sub(x)
        x = conv_out_new + pool_and_rep(x,rep_factor=2)   
        #x = conv_out_new + pool_and_rep(x,rep_factor=2,pool_factor=1)           
        #x = conv_out_new + x         
        self.conv_out.append(x) 
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)                            
        x = self.conv3_sub(x) + pool_and_rep(x,rep_factor=2)
        #x = self.conv3_sub(x)
        self.conv_out.append(x)
        
        if detached is True:
            x = torch.autograd.Variable(x.data,requires_grad=False)        
        x = self.conv4_sub(x) + pool_and_rep(x,rep_factor=2)     
        #x = self.conv4_sub(x)
        self.conv_out.append(x)        
        
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.squeeze(3).squeeze(2)
        
        if detached is True or detach_last is True:    
            x = torch.autograd.Variable(x.data,requires_grad=False)
            x = torch.autograd.Variable(x.data,requires_grad=True) # requires grad for final_in
        self.final_in = x
        
        x = self.fc1(self.last_nonlin(x))
        self.final_out = x
        return x
 

class SubBRC(nn.Module):
    def __init__(self,featIn,featOut, kerSize=3, stride=1, padding=0, detached=False, has_nonlinear=True, has_bn=True):
        super(SubBRC, self).__init__()        
        self.bn   = nn.BatchNorm2d(featIn, affine=False)
        self.conv = nn.Conv2d(featIn, featOut, kerSize, stride=stride, padding=padding)        
        self.detached = detached
        self.has_bn =  has_bn
        self.has_nonlinear = has_nonlinear
        
    def forward(self, x):
        if self.detached == True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        if self.has_bn:
            x = self.bn(x)
        if self.has_nonlinear:
            x = F.relu(x)        
        x = self.conv(x)
        return x


class SubBRC_nonlinear(nn.Module):
    def __init__(self,featIn,featOut, kerSize=3, stride=1, padding=0, detached=False,  nonlinear_func=torch.nn.functional.relu, has_bn=True):
        super(SubBRC_nonlinear, self).__init__()           
        self.bn   = nn.BatchNorm2d(featIn, affine=False)
        self.conv = nn.Conv2d(featIn, featOut, kerSize, stride=stride, padding=padding)        
        self.detached = detached
        self.has_bn =  has_bn        
        self.nonlinear_func = nonlinear_func
        
    def forward(self, x):
        if self.detached == True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        if self.has_bn:
            x = self.bn(x)
        #import pdb; pdb.set_trace();      
        if self.nonlinear_func is not None:
            x = self.nonlinear_func(x) 
        x = self.conv(x)
        return x



class SubRCB(nn.Module):
    def __init__(self,featIn,featOut, kerSize=3, stride=1, padding=0, detached=False, has_nonlinear=True, has_bn=True):
        super(SubRCB, self).__init__()        
        self.bn   = nn.BatchNorm2d(featOut, affine=False)     
        self.conv = nn.Conv2d(featIn, featOut, kerSize, stride=stride, padding=padding)        
        self.detached = detached
        self.has_bn =  has_bn
        self.has_nonlinear = has_nonlinear
        
    def forward(self, x):
        if self.detached == True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        if self.has_nonlinear:
            x = F.relu(x)        
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)        
        return x





    
