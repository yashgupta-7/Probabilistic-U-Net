# -*- coding: utf-8 -*-
"""
Model for Probabilistic U-Net

@author: Administrator
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from unet import U_Net, ConvEntity, MaxPoolEntity


class GenEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, ini_num_features=16, depth=5):
        super(GenEncoder, self).__init__()
        
        nf = ini_num_features
        self.depth = depth
        layers = []
        layers.append(ConvEntity(in_channels,nf))
        
        for i in range(1,depth):
            layers.append(MaxPoolEntity())
            layers.append(ConvEntity(nf,2*nf))
            nf = 2*nf
        
        self.layers = nn.Sequential(*layers)
        layer_mean=[]
        layer_mean.append(nn.Conv2d(nf, latent_dim, kernel_size=1))
        layer_mean.append(nn.ReLU())
        
        layer_std = []
        layer_std.append(nn.Conv2d(nf, latent_dim, kernel_size=1))
        layer_std.append(nn.ReLU())
        
        self.mean_layer = nn.Sequential(*layer_mean)
        self.std_layer = nn.Sequential(*layer_std)
        
    def forward(self, x):
        x = self.layers(x)
        x = torch.mean(x, 2, keepdim=True)
        x = torch.mean(x, 3, keepdim=True)
        mean = self.mean_layer(x)
        logstd = self.std_layer(x)
        
        dist = Independent(Normal(loc=mean, scale=torch.exp(logstd)),1)
        return dist
        
        


class Comb(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels, depth):
        super(Comb, self).__init__()
        ## channel_axis=1, spatial_axis = 2,3
        layers = []
        
        layers.append(nn.Conv2d(in_channels+latent_dim, in_channels, kernel_size=1, stride=1))
        layers.append(nn.ReLU())
        
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1))
            layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1))
        self.layers = nn.Sequential(*layers)
        
        
    def forward(self, features, z):
        bs, nch, h, w = features.shape
        bs, ld = z.shape
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, h)
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, w)        
        features = torch.cat((features, z), dim=1)
        return self.layers(features)
    
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)
        
        
        
    
class ProbU_Net(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, ini_num_features=16, unet_depth=5, enc_depth=5):
        super(ProbU_Net, self).__init__()
        self.latent_dim = latent_dim
        
        self.unet = U_Net(in_channels, out_channels, ini_num_features, unet_depth)       
        self.prior = GenEncoder(in_channels, latent_dim, ini_num_features, enc_depth)       
        self.posterior = GenEncoder(in_channels+1, latent_dim, ini_num_features, enc_depth)
        self.fcomb = Comb(ini_num_features, latent_dim, out_channels, depth=4)
        
        
    def forward(self, x, segx, train=True):      
        if train:
            self.post_latent_space = self.posterior.forward(torch.cat((x, segx), dim=1))
        self.prior_latent_space = self.prior.forward(x)
        self.unet_features = self.unet.forward(x)
    
    def loss(self, segx, beta):
        self.kld_loss = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        
        posterior_sample = self.posterior_latent_space.rsample()
        self.reconstruction = self.fcomb.forward(self.unet_features, posterior_sample)
        self.recon_loss = nn.BCEWithLogitsLoss(self.reconstruction, segx)
        
        return self.recon_loss + beta*self.recon_loss
