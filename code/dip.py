import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utils

class EncDec(nn.Module):
    def __init__(self):
        super(EncDec, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.dec = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
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


def denoise(noisy_img,output_file,lr=0.0001,niter=50000,traj_iter=1000):
    net = EncDec()
       
    eta = torch.randn(*noisy_img.size())
    eta = Variable(eta)
    eta.detach()
    
    fixed_target = noisy_img
    fixed_target = fixed_target.detach()
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    T = []
    for itr in range(niter):
        rec = net(eta)
        loss = mse(rec,fixed_target)
        loss.backward()
        optim.step()
        if itr%traj_iter == 0:
            T.append(rec[0, 0, :, :].transpose(0,1).detach().data.numpy())
            print('Iteration '+str(itr)+': '+str(loss.data),T[-1].shape)
                                         
    final_out = net(eta)
    T.append(final_out[0, 0, :, :].transpose(0,1).detach().data.numpy())
    
    # save trajectory
    np.save(output_file,np.array(T))

"""def traj_intersection(traj1,traj2):
    min_err = 1e10
    intersect = (0,0)
    for t1,im1 in enumerate(traj1):
        for t2,im2 in enumerate(traj2):
            e = ((im1-im2)**2).sum()
            if e < min_err:
                intersect = (t1,t2)
    return intersect
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--img_file', required=True, help='Path to image file'
    )
    parser.add_argument(
        '--output_file', required=True, help='Folder to save trajectories'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='Learning rate'
    )
    parser.add_argument(
        '--niter', type=int, default=100000, help='Num iters'
    )
    parser.add_argument(
        '--traj_iter', type=float, default=1000, help='Traj. logging iter'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    img_file = args.img_file
    output_file = args.output_file

    #load img
    noisy_img = utils.imread(img_file)
    noisy_img = Variable(utils.preproc(noisy_img))

    denoise(noisy_img,args.output_file,lr=args.lr,niter=args.niter,traj_iter=args.traj_iter)



