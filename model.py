from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import torch.nn.functional as F

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


ngf = 64
ndf = 64
nc = 1

class VAE(nn.Module):
    def __init__(self ):
        super(VAE, self).__init__()
    
        self.have_cuda = False
        
        nz = 16
        self.num_z = 8
        #self.dev = dev

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)
        
        ###linear classifier
        self.fc5 = nn.Linear(nz, 10)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x);
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,1024,1,1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)
    
     # muli-sampling z with size = self.num_z. 
    def reparameterize(self, mu, logvar):
        std =  torch.exp(0.5*logvar)
        z = [mu + std* ( torch.randn_like(logvar).cuda() ) for i in range(self.num_z)] #sampling for self.num_z times and catche them.
        #z = [mu + std* ( torch.randn_like(logvar).to(self.dev) ) for i in range(self.num_z)] #sampling for self.num_z times and catche them.
        z = torch.cat(z)  
        return z

    #def reparametrize(self, mu, logvar):
    #    std = logvar.mul(0.5).exp_()
    #    if self.have_cuda:
    #        eps = torch.cuda.FloatTensor(std.size()).normal_()
    #    else:
   #         eps = torch.FloatTensor(std.size()).normal_()
   #     eps = Variable(eps)
   #     return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparameterize(mu, logvar)
        
        ###prediction
        pred = self.fc5(z)
        #pred = self.fc5(mu)
        # print("z", z.size())
        decoded = self.decode(z)
        # print("decoded", decoded.size())
       
        return decoded, mu, logvar, pred

class VAE_FC(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.num_z = 8
        self.x_dim=784, 
        self.h_dim1= 512, 
        self.h_dim2=256, 
        self.z_dim=16
        
        # encoder part
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, 16)
        self.fc32 = nn.Linear(256, 16)
        # decoder part
        self.fc4 = nn.Linear(16, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        ###linear classifier
        self.fc7 = nn.Linear(16, 10)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def reparameterize(self, mu, logvar):
        std =  torch.exp(0.5*logvar)
        z = [mu + std* ( torch.randn_like(logvar).cuda() ) for i in range(self.num_z)] #sampling for self.num_z times and catche them.
        #z = [mu + std* ( torch.randn_like(logvar).to(self.dev) ) for i in range(self.num_z)] #sampling for self.num_z times and catche them.
        z = torch.cat(z)  
        return z
    
    #def sampling(self, mu, log_var):
    #    std = torch.exp(0.5*log_var)
     #   eps = torch.randn_like(std)
     #   return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        
        ###prediction
        pred = self.fc7(z)
        return self.decoder(z).view(-1, 28, 28), mu, log_var, pred

