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
from scipy.fftpack import fft
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
from skimage.measure import compare_psnr


def gen_dataset(k,A):
    N = 200.
    N_range = [0,1]
    z = np.arange(N_range[0],N_range[1],1/N)
    lmbda = np.zeros_like(z)
    for i,zi in enumerate(z):
        lmbda[i] = np.sum([Ai*np.sin(2*np.pi*ki*zi) for Ai,ki in zip(A,k)])
    return np.float32(lmbda)

def disp(signal,name=None):
    plt.ylim(-5,5)
    plt.ylabel('s(t)')
    plt.xlabel('Time')
    if name is None:
        plt.plot(signal)
    else:
        plt.plot(signal,label=name)


def parse_args():
    parser = argparse.ArgumentParser(description='1-D  Deep Image Prior experiment')
    parser.add_argument(
        '--output_dir', default='Outputs/one_d', help='Folder with all outputs'
    )
    parser.add_argument(
        '--cfg', required=True, help='Config file'
    )
    return parser.parse_args()


class DipModel(nn.Module):
    def __init__(self,lmbda,H=200,d=2,W=3,fc=False):
        super(DipModel, self).__init__()
        self.fc = fc
        if fc:
            self.w = [nn.Linear(len(lmbda),H,bias=False)] + \
                     [nn.Linear(H,H,bias=False) for _ in range(d-2)] + \
                     [nn.Linear(H,len(lmbda),bias=False)]
        else:
            fs = W
            #pad = (0 if fs%2==0 else fs//2)
            pad = fs//2
            self.w = [nn.Conv1d(1,H,fs,padding=max(0,pad))] + \
                     [nn.Conv1d(H,H,fs,padding=max(0,pad)) for _ in range(d-2)] + \
                     [nn.Conv1d(H,1,fs,padding=max(0,pad))]
        
        for i,wi in enumerate(self.w):
            self.add_module('w'+str(i),wi)

    def forward(self,x):
        out = x
        if not self.fc:
            out = out.unsqueeze(0).unsqueeze(0)
        d = len(self.w)
        for i in range(d-1):
            out = self._modules['w'+str(i)](out)
            out = F.relu(out,inplace=True)
        out = self._modules['w'+str(d-1)](out)
        if not self.fc:
            out = out[0,0,:]

        return out

def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()

### Different init. methods ###
"""def rescale(filt):
    f_ratio = (2 / (np.prod(filt.shape[1:]) + filt.shape[0]))**0.5
    #filt = 2 * f_ratio * (filt-filt.min())/(filt.max()-filt.min()) - f_ratio
    filt = f_ratio * filt # assuming filt  is already in {-1,1}
    return filt
"""

def xavier_init(m,scale):
    if isinstance(m, nn.Conv1d):
        #torch.nn.init.xavier_uniform_(m.weight,gain=gain)
        f_ratio = (scale / (np.prod(m.weight.shape[1:]) + m.weight.shape[0]))**0.5
        #f_ratio = (0.6 / (np.prod(m.weight.shape[1:]) + m.weight.shape[0]))**0.5
        filt = f_ratio * (2 * torch.rand(m.weight.shape) - 1) # uniform
        m.weight = torch.nn.Parameter(filt.cuda())

def uniform_init(m,scale):
    if isinstance(m, nn.Conv1d):
        filt = SCALE * torch.rand(m.weight.shape)
        m.weight = torch.nn.Parameter(filt.cuda())

def normal_init(m,scale):
    if isinstance(m, nn.Conv1d):
        #f_ratio = (scale / (np.prod(m.weight.shape[1:]) + m.weight.shape[0]))**0.5
        #filt = f_ratio * torch.randn(m.weight.shape) # normal
        filt = scale * torch.randn(m.weight.shape) # normal
        m.weight = torch.nn.Parameter(filt.cuda())

"""
def zeros_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.fill_(0.0)
    
def avg_init(m): 
    if isinstance(m, nn.Conv1d):
        filt = np.float32(np.ones(m.weight.shape))
        filt = rescale(filt)
        filt = torch.from_numpy(filt).cuda()
        m.weight = torch.nn.Parameter(filt)

# sharpening
def laplace_init(m):
    if isinstance(m, nn.Conv1d):
        if m.weight.shape[1] != 1:
            return
        filt = 2 * torch.rand(m.weight.shape) - 1
        filt[:,:,0] = -torch.abs(filt[:,:,0])
        filt[:,:,1] = torch.abs(filt[:,:,1])
        filt[:,:,2] = -torch.abs(filt[:,:,2])
        filt = rescale(filt)
        #import pdb; pdb.set_trace();
        m.weight = torch.nn.Parameter(filt.cuda())
"""
### End of init. methods ###


def dip(lmbda,H=200,d=2,W=3,fc=False,z_file=None,conv_init='xavier',scale=1.0):
    net = DipModel(lmbda,H=H,d=d,W=W,fc=fc).cuda()
    if conv_init == 'xavier':
        net.apply(lambda t: xavier_init(t,scale))
    elif conv_init == 'uniform':
        net.apply(lambda t: uniform_init(t,scale))
    elif conv_init == 'normal':
        net.apply(lambda t: normal_init(t,scale))
    elif conv_init == 'zeros':
        net.apply(lambda t: uniform_init(t,1e-8))
    #elif conv_init == 'laplace':
    #    net.apply(laplace_init)

    fixed_target = Variable(torch.from_numpy(lmbda)).cuda()
    if not z_file is None:
        eta = torch.from_numpy(np.load(z_file))
    else:
        eta = torch.randn(*lmbda.shape)
    eta = Variable(eta).cuda()
    optim = torch.optim.Adam(net.parameters(), lr=1e-5)
    mse = nn.MSELoss().cuda()
    niter = 15000
    T = []
    for itr in range(niter):
        optim.zero_grad()
        rec = net(eta)
        loss = mse(rec,fixed_target)
        loss.backward(retain_graph=True)
        optim.step()
        if (itr%1 == 0):
            out_np = rec.detach().cpu().data.numpy()
            T.append(out_np)
            print('Iteration '+str(itr)+': '+str(loss.data))
    
    # find fft trajectory
    T_fft = [fft(t) for t in T]
    
    return T, T_fft


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

    # read config
    with open(cfg_file,'r') as f:
        cfg = yaml.load(f)

    K = cfg['K']
    A = cfg['A']

    lmbda = gen_dataset(K,A)

    disp(lmbda,name='signal')
    plt.savefig(os.path.join(output_dir,'sig.png'))
    plt.close()

    N = lmbda.size
    lmbda_fft = fft(lmbda)
    lmbda_freq = 2.0/N * np.abs(lmbda_fft[0:int(N/2)])
    freq_k = np.linspace(0, int(N/2), int(N/2))
    
    channels = cfg['CHANNELS']
    depths = cfg['DEPTHS']
    use_fc = cfg['USE_FC']

    if 'FILT_SIZE' in cfg.keys():
        filt_size = cfg['FILT_SIZE']
    else:
        filt_size = [3]
    conv_init = cfg['CONV_INIT']
    
    if 'SCALE' in cfg.keys():
        scale = cfg['SCALE']
    else:
        scale = 1.0

    arch_conv = {}
    for H in channels:
        for d in depths:
            for W in filt_size:
                for fc in use_fc:
                    exp_name = 'H-'+str(H)+'_d-'+str(d)+'_W-'+str(W)+'_'+('fc' if fc else 'conv')
                    exp_dir = os.path.join(output_dir,exp_name)
                    if not os.path.exists(exp_dir):
                        os.system('mkdir '+exp_dir)

                    n_rounds = 10
                    T_rounds = [dip(lmbda,H=H,d=d,W=W,fc=fc,conv_init=conv_init,scale=scale) for _ in range(n_rounds)]
                
                    freq_loss = {}
                    for r,(T,T_fft) in enumerate(T_rounds):
                        freq_loss[r] = {k : [] for k in K}
                        for i,(t,ft) in enumerate(zip(T,T_fft)):
                            freq = 2.0/N * np.abs(ft[0:int(N/2)])
                            for k in K:
                                predf = freq[k]
                                truef = lmbda_freq[k]
                                freq_loss[r][k].append((predf-truef)**2)
                    

                    iters = range(len(T))
                
                    # plot variation of loss for each freq. component
                    for k in freq_loss[0].keys():
                        plt.plot(freq_loss[0][k],label='k='+str(k))
                    plt.legend()
                    plt.ylabel('SE(k)')
                    plt.xlabel('Iterations')
                    plt.savefig(os.path.join(exp_dir,'k_vs_T.png'))
                    plt.close() #-- dont close: overlay traj iter on this

                    # plot convergence time for each freq: avg. across all rounds
                    conv_thresh =  0.05
                    conv_iters = []
                    for r in range(n_rounds):
                        freq = [int(k) for k in sorted(freq_loss[r].keys())]
                        conv_iters.append(np.array(
                                [np.where(np.array(freq_loss[r][k]) < conv_thresh)[0][0] for k in sorted(freq_loss[r].keys())]) )
                    conv_iters = np.array(conv_iters)
                    conv_mean = np.mean(conv_iters,axis=0)
                    conv_std = np.std(conv_iters,axis=0)
                    plt.errorbar(freq,conv_mean,conv_std,marker='s',fmt='o',capsize=5)
                    plt.ylim(0,len(T))
                    plt.ylabel('Iterations')
                    plt.xlim(0,max(freq)+10)
                    plt.xlabel('Frequency')
                    plt.savefig(os.path.join(exp_dir,'iter_vs_freq.png'))
                    plt.close()

                    arch_conv[(d,H,W,fc)] = (conv_mean,conv_std)

                    # save traj at first, last and convergence iters
                    # for the first round
                    T, T_fft = T_rounds[0]
                    for i in [0,len(T)-1]+list(conv_iters[0]):
                        # traj iter
                        disp(T[i],name='rec')
                        plt.savefig(os.path.join(exp_dir,'rec_'+str(i)+'.png'))
                        plt.close()
                    
                        # traj iter in frequency domain
                        freq = 2.0/N * np.abs(T_fft[i][0:int(N/2)])
                        plt.ylim(0,1)
                        plt.bar(freq_k, freq, width=0.5)
                        plt.ylabel('|F(k)|')
                        plt.xlabel('Frequency k')
                        plt.savefig(os.path.join(exp_dir,'fft_'+str(i)+'.png'))
                        plt.close()
            
    
    # depth vs mean convergence time/freq. -- keep channels constant
    freq = list(K)
    for k,f in enumerate(freq):
        for fc in use_fc:
            for d in depths:
                arch = (d,channels[0],filt_size[0],fc)
                conv_mean,conv_std = arch_conv[arch]
                plt.errorbar([d],[conv_mean[k]],[conv_std[k]],marker='s',fmt='-',capsize=5,label='depth '+str(d))
            plt.ylim(0,len(T))
            plt.xlim(min(depths)-1,max(depths)+1)
            plt.ylabel('Iteration')
            plt.xlabel('Depth')
            plt.legend(prop={'size':8})
            plt.savefig(os.path.join(output_dir,'depth_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.png'))
            plt.close()

            # save actual mean,std as a table
            with open(os.path.join(output_dir,'depth_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.json'),'w') as fp:
                fp.write(json.dumps({d : [arch_conv[(d,channels[0],filt_size[0],fc)][0][k],
                                        arch_conv[(d,channels[0],filt_size[0],fc)][1][k],
                                        scale] for d in depths},indent=2))



    # num channels vs mean convergence time/freq. -- keep depth constant
    freq = list(K)
    for k,f in enumerate(freq):
        for fc in use_fc:
            for H in channels:
                arch = (depths[0],H,filt_size[0],fc)
                conv_mean,conv_std = arch_conv[arch]
                plt.errorbar([H],[conv_mean[k]],[conv_std[k]],marker='s',fmt='-',capsize=5,label=str(H)+' channels')
            plt.ylim(0,len(T))
            #plt.xlim(min(channels)-1,max(channels)+200)
            plt.xlim(0,max(channels)+200)
            plt.ylabel('Iteration')
            plt.xlabel('Channels')
            plt.legend(prop={'size':8})
            plt.savefig(os.path.join(output_dir,'channels_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.png'))
            plt.close()
        
            # save actual mean,std as a table
            with open(os.path.join(output_dir,'channels_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.json'),'w') as fp:
                fp.write(json.dumps({H : [arch_conv[(depths[0],H,filt_size[0],fc)][0][k],
                                        arch_conv[(depths[0],H,filt_size[0],fc)][1][k],
                                        scale] for H in channels},indent=2))


    # filter size vs mean convergence time/freq. -- keep depth and channels constant
    freq = list(K)
    for k,f in enumerate(freq):
        for fc in use_fc:
            for W in filt_size:
                arch = (depths[0],channels[0],W,fc)
                conv_mean,conv_std = arch_conv[arch]
                plt.errorbar([W],[conv_mean[k]],[conv_std[k]],marker='s',fmt='-',capsize=5,label='width '+str(W))
            plt.ylim(0,len(T))
            #plt.xlim(min(channels)-1,max(channels)+200)
            plt.xlim(0,max(filt_size)+20)
            plt.ylabel('Iteration')
            plt.xlabel('Filter Size')
            plt.legend(prop={'size':8})
            plt.savefig(os.path.join(output_dir,'filter_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.png'))
            plt.close()
    
            # save actual mean,std as a table
            with open(os.path.join(output_dir,'filter_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.json'),'w') as fp:
                fp.write(json.dumps({W : [arch_conv[(depths[0],channels[0],W,fc)][0][k],
                                        arch_conv[(depths[0],channels[0],W,fc)][1][k],
                                        scale] for W in filt_size},indent=2))


