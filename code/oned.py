import os
import sys
import cv2
import time
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


#K = [5,50]
#A = [1,1]


def gen_dataset(k,A):
    #k = K
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
    return parser.parse_args()


class DipModel(nn.Module):
    def __init__(self,lmbda,H=200,d=2,fc=False):
        super(DipModel, self).__init__()
        self.fc = fc
        if fc:
            self.w = [nn.Linear(len(lmbda),H,bias=False)] + \
                     [nn.Linear(H,H,bias=False) for _ in range(d-2)] + \
                     [nn.Linear(H,len(lmbda),bias=False)]
        else:
            self.w = [nn.Conv1d(1,H,3,padding=1,bias=False)] + \
                     [nn.Conv1d(H,H,3,padding=1,bias=False) for _ in range(d-2)] + \
                     [nn.Conv1d(H,1,3,padding=1,bias=False)]
        
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
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def dip(lmbda,H=200,d=2,fc=False,z_file=None):
    net = DipModel(lmbda,H=H,d=d,fc=fc).cuda()
    net.apply(weight_init)
    fixed_target = Variable(torch.from_numpy(lmbda)).cuda()
    if not z_file is None:
        eta = torch.from_numpy(np.load(z_file))
    else:
        eta = torch.randn(*lmbda.shape)
    eta = Variable(eta).cuda()
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    mse = nn.MSELoss().cuda()
    niter = 300
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

    K = [5,50]
    #K = range(5,60,10)
    #K = [10,20,60,70]
    
    # equal amplitudes 
    A = [1 for _ in K]
    #A = [1, 0.1][::-1]

    lmbda = gen_dataset(K,A)

    disp(lmbda,name='signal')
    plt.savefig(os.path.join(output_dir,'sig.png'))
    plt.close()

    N = lmbda.size
    lmbda_fft = fft(lmbda)
    lmbda_freq = 2.0/N * np.abs(lmbda_fft[0:int(N/2)])
    freq_k = np.linspace(0, int(N/2), int(N/2))
    for H in [256]:
        depths = [4,6,8,10,12]
        use_fc = [False,True]
        arch_conv = {}
        for d in depths:
            for fc in use_fc:
                exp_name = 'H-'+str(H)+'_d-'+str(d)+'_'+('fc' if fc else 'conv')
                exp_dir = os.path.join(output_dir,exp_name)
                if not os.path.exists(exp_dir):
                    os.system('mkdir '+exp_dir)

                n_rounds = 10
                T_rounds = [dip(lmbda,H=H,d=d,fc=fc) for _ in range(n_rounds)]
                
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
                conv_thresh = 0.05
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

                arch_conv[(d,fc)] = (conv_mean,conv_std)

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
                    plt.savefig(os.path.join(exp_dir,'fft_'+str(i)+'.png'))
                    plt.close()
            
        # depth vs mean convergence time/freq.
        freq = list(K)
        for k,f in enumerate(freq):
            for fc in use_fc:
                for d in depths:
                    arch = (d,fc)
                    conv_mean,conv_std = arch_conv[arch]
                    plt.errorbar([d],[conv_mean[k]],[conv_std[k]],marker='s',fmt='-',capsize=5,label='depth '+str(d))
                plt.ylim(0,len(T))
                plt.xlim(min(depths)-1,max(depths)+1)
                plt.ylabel('Iteration')
                plt.xlabel('Depth')
                plt.legend()
                plt.savefig(os.path.join(output_dir,'depth_vs_iter_k-'+str(f)+'_'+('fc' if fc else 'conv')+'.png'))
                plt.close()






