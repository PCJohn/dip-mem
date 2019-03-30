import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


class encdec(nn.Module):
    def __init__(self,net_depth):
        super(encdec, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(1024, 2048, 3, stride=2, padding=1),
            #nn.BatchNorm2d(2048),
            #nn.ReLU(),

            #nn.Conv2d(2048, 4096, 3, stride=2, padding=1),
            #nn.BatchNorm2d(4096),
            #nn.ReLU(),
        )
        self.dec = nn.Sequential(
            #nn.UpsamplingBilinear2d(scale_factor=2),
            #nn.Conv2d(4096, 2048, 3, stride=1, padding=1),
            #nn.BatchNorm2d(2048),
            #nn.ReLU(),

            #nn.UpsamplingBilinear2d(scale_factor=2),
            #nn.Conv2d(2048, 1024, 3, stride=1, padding=1),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        for m in self.enc.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.dec.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.dec(self.enc(x))

