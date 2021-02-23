#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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

import vae_conv_model_mnist

import matplotlib.pyplot as plt  


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), '%s/vae_epoch_%d.pth' % (outf, epoch))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 16)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), '%s/%03d.png' % (outf_samples, epoch), nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def cifar_imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (0,1)))
    plt.show()
    
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
