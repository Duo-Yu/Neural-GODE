import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):  # filter size 3
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):  # filter size 1
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim): # apply a group normalization over a mini-batch of input; 
    #The input channels are separated into num_groups groups, each containing num_channels/num_groups channels.
    return nn.GroupNorm(min(32, dim), dim)  # number of groups: min(32, dim); number of channels: dim
    
def basisF(u, k, i, t):     # the basis function based on recursive formula
    if k == 0:
        return 1.0 if t[i] <= u < t[i+1] else 0.0
    if t[i+k] == t[i]:
        s1 = 0.0
    else:
        s1 = (u - t[i])/(t[i+k] - t[i]) * basisF(u, k-1, i, t)
    if t[i+k+1] == t[i+1]:
        s2 = 0.0
    else:
        s2 = (t[i+k+1] - u)/(t[i+k+1] - t[i+1]) * basisF(u, k-1, i+1, t)
    return s1 + s2
    
class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)           
        self.relu = nn.ReLU(inplace=True)     
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out) 
        out = self.norm2(out) #trick of batch normalization: adopt the BN right after convolution before activation
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )          # add another dimension for time

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t #fill the tensor with all 1, dimension as similar as x[:,:1,:,:]
        ttx = torch.cat([tt, x], 1)              #concatenate time with x, at dimension 1. 
        return self._layer(ttx)

 # define a layer of linear combination with spline function as coefficients and bias    
class SplineLinear(nn.Module):
    
    def __init__(self, dim, spline_n=3,step_size=0.1):
        super(SplineLinear,self).__init__()                 # inherit the basic module
        
        # parameters
        self.dim = dim
        self.n = spline_n
        self.step_size = step_size
           
        # trainable coefficients of the spline function
        c1 = torch.Tensor(self.dim,self.dim, 3, 3, self.n)
        self.c1 = torch.nn.Parameter(c1)
        nn.init.kaiming_uniform_(self.c1, a= math.sqrt(3))

    
    def BS(self, t, c):     # calculate basis function
        t_index = int(torch.round(t/0.001).item())
        #print(t)
        Tele_spline = torch.Tensor(ele_spline[t_index]).to(device = device)
        BS_fvalue = torch.mul(c,Tele_spline)
        BS_fvalue = torch.sum(BS_fvalue,4)
        return BS_fvalue
    
    def spline_conv(self,t,x):    
        weights = self.BS(t, self.c1) # b-spline function as the cnn weights
        #conv_fun = nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1, bias=False)
        #with torch.no_grad():
        #    conv_fun.weight = nn.Parameter(weights)   
        return F.conv2d(x,weights,stride=1,padding=1,bias=None)
    
    def forward(self,t,x):
        out = self.spline_conv(t,x)
        return out
            

class ODEfunc(nn.Module):
    def __init__(self,dim,spline_n,step_size):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.spline_linear1 = SplineLinear(dim,spline_n,step_size)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.spline_linear2 = SplineLinear(dim,spline_n,step_size)
        
        self.nfe = 0                          # number of function evaluation
        
    def forward(self,t,x):
        #print(x.shape)
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.spline_linear1(t,out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.spline_linear2(t,out)
  
        return out
        

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, args.int_T]).float()  # integrate from 0 to int_T

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        #print(self.integration_time)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='dopri5') # the ODE system
        #out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method = 'euler', options=dict(step_size=args.ode_step_size)) # the ODE system
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        #print(x.shape)
        #print(x.view(-1, shape).shape)
        return x.view(-1, shape)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        #print('value:'+str(self.val))
        #print(self.avg)
        


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([       # chain the transformation together using compose
            transforms.RandomCrop(28, padding=4),    # crop the image at random location with padding 
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def inf_generator(iterable):  # a object is iterable if we can get an iterator from it
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()  # define an interator
    while True:
        try:
            yield iterator.__next__() # get the element of next iterator
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom
    #initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()  # 1. setup a logger
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)     # 2. define the cut-point of level, logger levels have default priority 
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())
 
    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./NeuralODE_experiments/experiment4')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--spline_n', type=int, default=3)  # number of coefficients 
parser.add_argument('--spline_k', type=int, default=2)  # degree of B-spline
parser.add_argument('--int_T', type=int, default = 1)
parser.add_argument('--ode_step_size', type=float, default=0.01)
parser.add_argument('--num_equations', type=int, default = 10)
args = parser.parse_args()

tt = np.append(np.arange(0,args.int_T, args.int_T/(args.spline_k + args.spline_n)),[args.int_T]) # knots vector
ele_spline = [[basisF(t, args.spline_k, i, tt) for i in range(args.spline_n)] for t in np.arange(0,4,0.001)]

if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
else:
        from torchdiffeq import odeint
if __name__ == '__main__':
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),   # in_channels = 1, out_channels = 64, filter_size = 3, stride = 1, outdim = 126
            # number_parameters = 3*3*64+64 = 640
            norm(64),                # group normalization
            # number_parameters = 64*2 = 128
            nn.ReLU(inplace=True),
            # number_parameters = 0
            nn.Conv2d(64, 64, 4, 2, 1), #in_channels = 64, out_channels = 64, filter_size =4, stride = 2, padding = 1
            # number_parameters = 4*4*64 (input_channels)*64+64 = 65600, output_dim = 63
            norm(64),
            # number_parameters = 64*2
            nn.ReLU(inplace=True),
            # number_parameters = 0
            nn.Conv2d(64, 64, 4, 2, 1), # output: 64 filter map with dimension 
            # number_parameters = 65600
            #norm(64),
            #nn.ReLU(inplace=True),
        ]   # total number of parameters = 132096
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    fc_layers_1 = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),Flatten()]
    fc_layers_2 = [nn.Linear(args.num_equations, 10)]
    feature_layers = [ODEBlock(ODEfunc(args.num_equations,args.spline_n,args.ode_step_size))]
    model = nn.Sequential(*downsampling_layers, *feature_layers,*fc_layers_1,*fc_layers_2).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)  # combine softmax and loss 

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(   # loading function
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader) # infinite loop
    batches_per_epoch = len(train_loader)  # number of batches

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()
    
    for itr in range(args.nepochs * batches_per_epoch):       

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        
        #print('value:'+str(batch_time_meter.val))
        #print(batch_time_meter.avg)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()


        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, 
                        batch_time_meter.avg,f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc)
                    )
                #log_time=[]
