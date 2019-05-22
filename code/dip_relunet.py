import os
import sys
import cv2
import time
import json
import yaml
import argparse
import itertools
import numpy as np
import matplotlib
import collections
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
from skimage.measure import compare_psnr
from scipy.fftpack import fft2

import utils


def gen_dataset(img):
    bw = (len(img.shape) == 2)
    if bw:
        w,h = img.shape
        c = 1
    else:
        w,h,c = img.shape
    xx,yy = np.meshgrid(range(w),range(h),indexing='ij')
    xx,yy = xx.flatten(),yy.flatten()
    if bw:
        y = img[xx,yy][:,np.newaxis]
    else:
        y = img[xx,yy,:]

    xx = xx / float(w-1)
    yy = yy / float(h-1)
    x = np.hstack([xx[:,np.newaxis],yy[:,np.newaxis]])
    
    return np.float32(x),np.float32(y),w,h,c


def parse_args():
    parser = argparse.ArgumentParser(description='1-D  Deep Image Prior experiment')
    parser.add_argument(
        '--output_dir', default='Outputs/dip_relunet', help='Folder with all outputs'
    )
    parser.add_argument(
        '--cfg', required=True, help='Config file'
    )
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    return parser.parse_args()


class ReLUNet(nn.Module):
    def __init__(self,H=200,d=2,bw=True):
        super(ReLUNet, self).__init__()
        out_dim = (1 if bw else 3)
        self.w = [nn.Linear(2,H,bias=True)] + \
                 [nn.Linear(H,H,bias=True) for _ in range(d-2)] + \
                 [nn.Linear(H,out_dim,bias=True)]
        
        for i,wi in enumerate(self.w):
            self.add_module('w'+str(i),wi)

    def forward(self,x):
        out = x
        d = len(self.w)
        for i in range(d-1):
            out = self._modules['w'+str(i)](out)
            out = F.relu(out,inplace=True)
        out = self._modules['w'+str(d-1)](out)
        out = F.sigmoid(out)
        return out

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def train_net(x,y,w,h,c,H=200,d=2,niter=200000,bz=256,traj_iter=1000):
    bw = (y.shape[-1] == 1)
    net = ReLUNet(H=H,d=d,bw=bw).cuda()
    net.apply(weight_init)

    x = Variable(torch.from_numpy(x)).cuda()
    y = Variable(torch.from_numpy(y)).cuda()

    optim = torch.optim.Adam(net.parameters(), lr=1e-3) #lr=1e-4)
    mse = nn.MSELoss().cuda()
    T = []
    for itr in range(niter):
        optim.zero_grad()
        b = np.random.randint(0,x.shape[0],bz)
        y_ = net(x[b])
        loss = mse(y_,y[b])
        #loss = torch.sum((y_-y)**2)
        loss.backward(retain_graph=True)
        optim.step()
        if (itr%traj_iter == 0):
            #import pdb; pdb.set_trace();
            out_np = net(x).detach().cpu().data.numpy().reshape((w,h,c))
            if bw:
                out_np = out_np[:,:,0]
            utils.imwrite(os.path.join(output_dir,'itr'+str(itr)+'.png'),out_np)
            T.append(out_np)
            print('Iteration '+str(itr)+': '+str(loss.data))
            del out_np
    
    return T


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)

    cfg_file = args.cfg
    cfg_name = os.path.split(cfg_file)[-1].split('.')[0]
    # make folder for this config
    output_dir = os.path.join(output_dir,cfg_name)
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)
    
    exp_name = os.path.split(args.noisy_img)[-1].split('.')[0].split('_s')[0]
    output_dir = os.path.join(output_dir,exp_name)
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)


    # load config
    with open(cfg_file,'r') as f:
        cfg = yaml.load(f)

    img = utils.imread(args.noisy_img)
    x,y,w,h,c = gen_dataset(img)
    
    # load default config file
    with open('configs/default.yaml','r') as f:
        default_cfg = yaml.load(f)
    niter = default_cfg['NUM_ITER']
    traj_iter = default_cfg['TRAJ_ITER']
    lr_val = default_cfg['LR']
    lang_sigma = default_cfg['LANG']
    reg_noise_std = default_cfg['REG_NOISE_STD']
    init = default_cfg['INIT']
    init_scale = default_cfg['INIT_SCALE']
    bn = default_cfg['BN']
    depth = default_cfg['DEPTH']
    stride = default_cfg['STRIDE']
    #####
    
    # load given config file
    with open(args.cfg,'r') as f:
        cfg = yaml.load(f)
                       
    niter = cfg['NUM_ITER']
    traj_iter = cfg['TRAJ_ITER']
    lr_val = cfg['LR']
    if 'LANG' in cfg.keys():
        lang_sigma = cfg['LANG']
    if 'REG_NOISE_STD' in cfg.keys():
        reg_noise_std = cfg['REG_NOISE_STD']
    if 'INIT' in cfg.keys():
        init = cfg['INIT']
    if 'INIT_SCALE' in cfg.keys():
        init_scale = cfg['INIT_SCALE']
    if 'BN' in cfg.keys():
        bn = cfg['BN']
    if 'DEPTH' in cfg.keys():
        depth = cfg['DEPTH']
    if 'STRIDE' in cfg.keys():
        stride = cfg['STRIDE']
    hparams = [lr_val,lang_sigma,init,init_scale,bn,depth,stride]
    ###### 


    channel = [128]
    bz = 2048
    
    added_noise_var = cfg['ADDED_NOISE']
    net_structs = ['net_denoise']
    hparams.extend([net_structs,added_noise_var])
    #noisy = utils.imread(noisy_path)
    for lr,lang,init,init_scale,bn,depth,stride,struct,added_var in itertools.product(*hparams):
        hyp_str = utils.gen_hyp_str(lr,depth,added_var,lang,init,init_scale,bn,stride)
        output_file = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
        added_output_file = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
        T = train_net(x,y,w,h,c,
                      H=channel[0],
                      d=depth,
                      niter=niter,
                      bz=bz,
                      traj_iter=traj_iter)
        utils.save_traj(output_file,T)
        utils.save_traj(added_output_file,T)



