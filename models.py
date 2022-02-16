from re import M
from typing_extensions import _SpecialForm
import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self, args):
        super(G, self).__init__()
        self.n_gpu = args.n_gpu
        self.z_dim = args.z_dim
        self.ngf = args.g_feature_dim 
        self.nc = args.channel_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.z_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class D(nn.Module):
    def __init__(self, args):
        super(D, self).__init__()
        self.n_gpu = args.n_gpu
        self.z_dim = args.z_dim
        self.ndf = args.d_feature_dim 
        self.nc = args.channel_dim
        
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(self.ndf), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, biase = False),
            nn.BatchNorm2d(self.ndf * 4), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias = False),
            nn.sigmoid
        )

        
    def forward(self, input): 
        return self.main(input)