import os
import sys
import cv2
import time
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from skimage.measure import compare_psnr

import dip
import utils
import skip

import config as cfg

def fft(ch):
    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift))
    return mag

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior: Analyze clean image')
    parser.add_argument(
        '--img', required=True, help='Path to image file'
    )
    parser.add_argument(
        '--task', required=True, help='Task: traj or plot'
    )
    parser.add_argument(
        '--output_root', default='Outputs', help='Folder with all outputs'
    )
    parser.add_argument(
        '--fixed_start', default=False, action='store_true', help='Start models from the same point'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    im_path = args.img

    output_root = os.path.join(args.output_root,'clean')
    if not os.path.exists(output_root):
        os.system('mkdir '+output_root)
    
    exp_name = os.path.split(im_path)[-1].split('.')[0]
    output_dir = os.path.join(output_root,exp_name)
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)

    im = utils.imread(im_path)

    # dip config params
    niter = cfg.NUM_ITER
    traj_iter = cfg.TRAJ_ITER
    lr = cfg.LR[0]
    lang = cfg.LANG[0]
    reg_noise_std = cfg.REG_NOISE_STD
    struct = 'net_denoise'

    # arch search space
    num_channels = [16,32,64,128,256]
    depths = [5,10,15,25]
    skip_conn = 4
    ntraj = 2

    basic_ch = 128
    basic_depth = 5

    if args.task == 'traj':
        # vary number of channels with constant depth 5
        for ch in num_channels:
            n_ch_up = ch
            n_ch_down = ch
            depth = basic_depth
            hyp_str = '_ch-'+str(ch)+'_d-'+str(depth)
            for t in range(ntraj):
                traj_file = os.path.join(output_dir,'traj'+str(t+1)+hyp_str+'.npz')
                cmd = 'sbatch gypsum/scripts/clean_analyze.sh '+' '.join([im_path,traj_file,
                    str(lr),str(niter),str(traj_iter),str(struct),str(int(args.fixed_start)),str(lang),str(reg_noise_std),
                    str(n_ch_down), str(n_ch_up), str(skip_conn), str(depth)])
                print('Submitting:',cmd)
                os.system(cmd)
    
        # vary depth with constant 128 channels
        for depth in depths:
            n_ch_up = basic_ch
            n_ch_down = basic_ch
            hyp_str = '_ch-'+str(basic_ch)+'_d-'+str(depth)
            for t in range(ntraj):
                traj_file = os.path.join(output_dir,'traj'+str(t+1)+hyp_str+'.npz')
                cmd = 'sbatch gypsum/scripts/clean_analyze.sh '+' '.join([im_path,traj_file,
                    str(lr),str(niter),str(traj_iter),str(struct),str(int(args.fixed_start)),str(lang),str(reg_noise_std),
                    str(n_ch_down), str(n_ch_up), str(skip_conn), str(depth)])
                print('Submitting:',cmd)
                os.system(cmd)

    elif args.task == 'plot':
        traj_ext = '.npz'
        hparam_res = {}
        for f in os.listdir(output_dir):
            if not f.endswith(traj_ext):
                continue
            hyp_str = os.path.split(f)[-1].split(traj_ext)[0]
            hyp_str = hyp_str[hyp_str.index('_ch-'):]
            ch,d = [int(h.split('-')[1]) for h in hyp_str.split('_')[1:]]
            hparam_res[(ch,d)] = {}
            
            t1_file = os.path.join(output_dir,'traj1'+hyp_str+'.npz')
            t2_file = os.path.join(output_dir,'traj2'+hyp_str+'.npz')
            traj1 = utils.load_traj(t1_file)
            traj2 = utils.load_traj(t2_file)
            assert (len(traj1) == len(traj2))
            
            err_true_t1 = []
            err_true_t2 = []
            err_traj = []
            for t1,t2 in zip(traj1,traj2):
                err_true_t1.append(((im - t1)**2).sum())
                err_true_t2.append(((im - t2)**2).sum())
                err_traj.append(((t1 - t2)**2).sum())
      
            hparam_res[(ch,d)]['err_traj'] = err_traj
            hparam_res[(ch,d)]['err_true_t1'] = err_true_t1
            hparam_res[(ch,d)]['err_true_t2'] = err_true_t2
       
            # save distance between trajectories
            plt.xlabel('Iterations')
            plt.ylabel('SSE')
            plt.plot(err_traj)
            err_traj = os.path.join(output_dir,'err_traj'+hyp_str+'.png')
            plt.savefig(err_traj,bbox_inches='tight')
            plt.cla(); plt.clf(); plt.close()

            # save true error of the trajectories
            plt.xlabel('Iterations')
            plt.ylabel('SSE')
            plt.plot(err_true_t1)
            err_true_t1 = os.path.join(output_dir,'err_true_t1'+hyp_str+'.png')
            plt.savefig(err_true_t1,bbox_inches='tight')
            plt.close()
            plt.xlabel('Iterations')
            plt.ylabel('SSE')
            plt.plot(err_true_t2)
            err_true_t2 = os.path.join(output_dir,'err_true_t2'+hyp_str+'.png')
            plt.savefig(err_true_t2,bbox_inches='tight')
            plt.cla(); plt.clf(); plt.close()

            # save img and fft trajectories
            fft_dir = os.path.join(output_dir,'fft'+hyp_str)
            if not os.path.exists(fft_dir):
                os.system('mkdir '+fft_dir)

            traj_dir = os.path.join(output_dir,'traj'+hyp_str)
            if not os.path.exists(traj_dir):
                os.system('mkdir '+traj_dir)
        
            fft_var = [fft(t[:,:]) for t in traj1]
            for i in range(0,min(50,len(fft_var)*traj_iter),2):
                cv2.imwrite(os.path.join(fft_dir,str(i)+'_fft.png'),20*fft_var[i])
                utils.imwrite(os.path.join(traj_dir,str(i))+'_img.png',traj1[i,:,:])
                plt.cla(); plt.clf(); plt.close()

            plt.plot([np.sum(f) for f in fft_var])
            plt.savefig(os.path.join(output_dir,'fft_var.png'),bbox_inches='tight')
            plt.cla(); plt.clf(); plt.close()
       
        traj_iter = cfg.TRAJ_ITER
        for err_type in ['err_traj','err_true_t1','err_true_t2']:
            # compare across varying num channels
            for ch in num_channels:
                depth = basic_depth
                err_var = hparam_res[(ch,depth)][err_type]
                x = [(t+1)*traj_iter for t in range(len(err_var))]
                plt.plot(err_var, label=str(ch)+' channels')
            plt.xlabel('Iterations')
            plt.ylabel('SSE')
            plt.legend()
            plt.savefig(os.path.join(output_dir,err_type+'_channels.png'),bbox_inches='tight')
            plt.cla(); plt.clf(); plt.close()

            # compare across varying depth
            for depth in depths:
                ch = basic_ch
                err_var = hparam_res[(ch,depth)][err_type]
                x = [(t+1)*traj_iter for t in range(len(err_var))]
                plt.plot(err_var,label='depth '+str(depth))
            plt.xlabel('Iterations')
            plt.ylabel('SSE')
            plt.legend()
            plt.savefig(os.path.join(output_dir,err_type+'_depths.png'),bbox_inches='tight')
            plt.cla(); plt.clf(); plt.close()


