from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import json

import datetime
import os
from utils import *

import vae_conv_model_mnist
import matplotlib.pyplot as plt  
import torchvision as thv
import numpy as np
def imshow(img):
    img = img.repeat(3,1,1)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dataroot', help='path to dataset')


parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')


args = parser.parse_args(args=[])
#print('where')
args.cuda = not args.no_cuda and torch.cuda.is_available()
#args.cuda = cuda(2)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("cuda", args.cuda, args.no_cuda, torch.cuda.is_available())

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

args.model = vae_conv_model_mnist.VAE(20)
args.model.have_cuda = args.cuda
if args.cuda:
    args.model.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss( size_average=False, reduction='sum')
optimizer = optim.Adam(args.model.parameters(), lr=1e-3)

#loss function
def loss_function(recon_x, x, pred, y, mu, logvar, num_z):
    xx = torch.cat([x.view(-1, 3, 32, 32)]* num_z)
    MSE = F.mse_loss(recon_x, xx, reduction='sum')/num_z    
    #MSE = F.mse_loss(recon_x, x, reduction='sum')
    #MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    yy = torch.cat([y]*num_z)
    #c = criterion_loss( pred, y ) 
    c = criterion(pred, yy)
    return KLD, MSE, c/num_z

# Reconstruction + KL divergence losses summed over all elements and batch
##def loss_function(recon_x, x, mu, logvar):
    # print(recon_x.size(), x.size())
#    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 #   return BCE, KLD
    
    
########
def data_split(domain, index):
    domain.targets = torch.tensor( domain.targets )
    for k in range( len(index) ):
        if k == 0:
            idx = domain.targets == index[k]
        else:
            idx += domain.targets == index[k]
    domain.targets= domain.targets[idx]
    domain.data = domain.data[idx.numpy().astype(np.bool)]
    return domain

###source dataset and loader
train_domain = thv.datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=transforms.ToTensor() )  
test_domain = thv.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transforms.ToTensor() )

source_train_domain = data_split(train_domain, [0, 1, 2, 3, 4])
source_test_domain = data_split(test_domain, [0, 1, 2, 3, 4])

source_train_loader = torch.utils.data.DataLoader(source_train_domain, batch_size=args.batch_size, shuffle=True, **kwargs, drop_last = True)
source_test_loader = torch.utils.data.DataLoader(source_test_domain, batch_size=args.batch_size, shuffle=False, **kwargs)

### target dataset and loader
train_domain = thv.datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=transforms.ToTensor() )  
test_domain = thv.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transforms.ToTensor() )

target_train_domain = data_split(train_domain, [5, 6, 7, 8, 9])
target_test_domain = data_split(test_domain, [5, 6, 7, 8, 9])

target_train_loader = torch.utils.data.DataLoader(target_train_domain, batch_size=args.batch_size, shuffle=True, **kwargs, drop_last = True)
target_test_loader = torch.utils.data.DataLoader(target_test_domain, batch_size=args.batch_size, shuffle=False, **kwargs)


def train(epoch):
    args.model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(source_train_loader):
        args.count += 1
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar, pred = args.model(data)
        KLD, MSE, c = loss_function(recon_x, x, pred, y, mu, logvar, num_z)       
        loss = KLD + args.lumbda_max * MSE + args.gamma * c
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

def test(loader, num_z):
    args.model.eval()
    #test_loss = 0
    test_BCE, test_KLD, test_C, total, correct= 0., 0., 0., 0, 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            if args.cuda:
                data = data.cuda()
                labels = labels.cuda()
            #data = Variable(data, volatile=True)
            recon_batch, mu, logvar, pred = args.model(data)
            
            #y = th.cat([y]*nz)
            #predicted = pred.argmax(dim=1, keepdim=True)
            #ac = (predicted.eq(y.view_as(predicted)).sum().item())/nz

            _, predicted = torch.max(pred.data, 1)
            ll = torch.cat([labels]*num_z)
            total += ll.size(0)
            correct += (predicted == ll).sum().item()

            #BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            #Cross = criterion(pred, labels)
            KLD, MSE, c = loss_function(recon_batch, x, pred, y, mu, logvar, num_z)   

            test_MSE += MSE.item()
            test_KLD += KLD.item()
            test_C += c.item()
           
        test_BCE /= (total /num_z)
        test_KLD /= (total /num_z)
        #test_C /= len(test_loader.dataset)
        test_C /= (total /num_z)

    return test_KLD, test_BCE, test_C, correct/total
    
def transfer(epoch):   
    num_z = args.num_z
    args.model.train()
    for ( d1, d2 ) in zip ( source_train_loader,  target_train_loader):
        #beta_learning_rate(optimizer, args.itr_in_epoch, args.num_epoch, epoch, args.itr)
        x1, y1 = d1
        x2, y2 = d2
        if args.cuda:
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        optimizer.zero_grad()     
        
        recon1, mu1, logvar1, pred1 = args.model(x1)
        KLD1, MSE1, c1 = loss_function(recon1, x1, pred1, y1, mu1, logvar1, num_z)   
        #BCE1, KLD1 = loss_function(recon1, x1, mu1, logvar1)
        #Cross1 = criterion(pred1, y1)
        loss1 = KLD1 + args.lumbda * MSE1 + args.gamma * c1
        
        recon2, mu2, logvar2, pred2 = args.model(x2)
        KLD2, BCE2, c2 = loss_function(recon2, x2, pred2, y2, mu2, logvar2, num_z)   
        #BCE2, KLD2 = loss_function(recon2, x2, mu2, logvar2)
        #Cross2 = criterion(pred2, y2)
        loss2 = KLD2 + args.lumbda * MSE2 + args.gamma * c2
        
        loss = ( 1 - args.r ) * loss1 + args.r * loss2
        loss.backward()      
        optimizer.step()
        args.itr += 1
        
        
def train_beta(epoch):
    args.model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(source_train_loader):
        #args.count += 1
        beta_learning_rate(optimizer, args.count, args.num_epoch, epoch, args.n)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar, pred = args.model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        Cross = criterion(pred, labels)
        loss = KLD + args.lumbda_max * BCE + args.gamma * Cross
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        args.n += 1
        
#counting number of updates in each epoch across transfer learning
count = 0 
for ( d1, d2 ) in zip ( source_train_loader,  target_train_loader):
    count+=1 
args.itr_in_epoch = count
print( args.itr_in_epoch, 'iterations in one epoch')

####beta learning rate schedule
def beta_learning_rate(optimizer, itr_in_epoch, num_epoch, epoch, itr):
    ita = 8e-3
    alpha, beta = 2, 5
    s = alpha + beta
    iters = itr + epoch * itr_in_epoch
    num_iters = num_epoch * itr_in_epoch
    x = iters / num_iters
    peak = ( alpha ** alpha ) * (  beta ** beta ) / ( s ** s )
    lr_beta_rate =  ita / peak * ( x ** alpha ) * ( ( 1 - x ) ** beta ) + 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_beta_rate 
    if itr == int(itr_in_epoch-1):
        print( lr_beta_rate, 'learning rate at epoch', epoch )
        
        
args.lumbda_max, args.lumbda_min, args.gamma= 2., 0.5, 5.
args.l_list, args.g_list, args.C_list, args.r_list = [args.lumbda_max], [args.gamma], [], [0.]

#pretrain model on source domain
for epoch in range(1, 10):
    args.count = 0 
    train(epoch)
    _, _, C = test(source_test_loader)
    print('====> Source Domain Test Classification Loss: {:.4f}'.format(C))

_, _, args.C0 = test(source_test_loader)
args.C_list.append( args.C0 )
args.step, args.lumbda = 0, args.lumbda_max
args.num_step, args.num_epoch = 25, 8

#####An equilibrium process for transfer learning: keepliing the classicifation loss
for k in range( args.num_step ): 
    args.r = ( k +1 )/args.num_step 
    _, _, C1 = test(source_test_loader)
    _, _, C2 = test(target_test_loader)
    C_check = ( 1 - args.r ) * C1 + args.r * C2
    
    substep = 0 
    while C_check >= args.C0: 
        #updating lumbda and gamma to make sure classification loss bounded across transfer learning
        if substep >0:
            args.lumbda -= ( args.lumbda - args.lumbda_min ) / ( args.num_step - k )
            args.gamma += 1e-1
            print('====> updating lumbda: {:.4f}'.format(args.lumbda))
            print('====> updating gamma: {:.4f}'.format(args.gamma))
            
        for epoch in range(args.num_epoch):
            optimizer = optim.Adam(args.model.parameters(), lr=1e-3)
            args.itr = 0 
            transfer(epoch)
            
        _, _, C1 = test(source_test_loader)
        _, _, C2 = test(target_test_loader)
        C_check = ( 1 - args.r ) * C1 + args.r * C2
        substep += 1 

    args.l_list.append( args.lumbda )
    args.g_list.append( args.gamma )
    args.C_list.append(C_check)
    print('====> Test Classification Loss on interpolating domain: {:.4f}'.format(C_check))

