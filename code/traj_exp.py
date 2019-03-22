import os
import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import dip
import utils

def traj_intersection(traj1,traj2):
    min_err = 1e10
    intersect = (0,0)
    for t1,im1 in enumerate(traj1):
        for t2,im2 in enumerate(traj2):
            e = ((im1-im2)**2).sum()
            if e < min_err:
                intersect = (t1,t2)
    return intersect


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--clean_img', required=True, help='Path to image file'
    )
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    parser.add_argument(
        '--task', required=True, help='Task: denoise or hyperopt'
    )
    parser.add_argument(
        '--output_root', default='Outputs', help='Folder with all outputs'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    #clear_path = args.clean_img
    #noisy_path = args.noisy_img

    clean_path = 'data/saturn.png' #'../data/denoising/lena.png'
    noisy_path = 'data/saturn-noisy.png' # '../data/denoising/lena-noisy.png'

    exp_name = os.path.split(clean_path)[-1].split('.')[0]
    output_dir = os.path.join(args.output_root,exp_name)
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)

    im = utils.imread(clean_path)
    added_noisy_path = os.path.join(output_dir,'added_noise.png')
    noisy_output = os.path.join(output_dir,'T_noisy.npy')
    added_noisy_output = os.path.join(output_dir,'T_added.npy')

    lr = 0.001
    niter = 50000
    traj_iter = 200
    
    if args.task == 'denoise':
        noisy = utils.imread(noisy_path)
        added_noisy = utils.add_noise('gauss',noisy)
        utils.imwrite(added_noisy_path,added_noisy)
        for out_path in [noisy_output,added_noisy_output]:
            cmd = 'sbatch gypsum/scripts/denoise.sh '+noisy_path+' '+out_path+' '+str(lr)+' '+str(niter)+' '+str(traj_iter)
            print('Running:',cmd)
            os.system(cmd)
    
    elif args.task == 'hypopt':
        noisy_T = np.load(noisy_output)
        added_noisy_T = np.load(added_noisy_output)
        
        assert (len(noisy_T) == len(added_noisy_T))

        err_true = []
        err_pred = []
        for t1,t2 in zip(noisy_T,added_noisy_T):
            err_true.append(((im - t1)**2).sum())
            err_pred.append(((t1 - t2)**2).sum())

        # save variation of true error
        plt.title('True Error')
        x = [i*traj_iter for i in range(len(err_true))]
        plt.plot(x,err_true)
        plt.savefig(os.path.join(output_dir,'err_true.png'),bbox_inches='tight')
        plt.close()

        # find true best iteration and save image
        best_iter = np.argmin(err_true)
        best_err_true = err_true[best_iter]
        plt.title('True Best. '+'iter: '+str(best_iter*traj_iter)+' err: {:.2f}'.format(best_err_true),fontsize=12)
        plt.imshow(noisy_T[best_iter])
        plt.savefig(os.path.join(output_dir,'best_true.png'),bbox_inches='tight')
        plt.close()

        # variation of added error
        plt.title('Pred Error')
        x = [i*traj_iter for i in range(len(err_pred))]
        plt.plot(x,err_pred)
        plt.savefig(os.path.join(output_dir,'err_pred.png'),bbox_inches='tight')
        plt.close()

        # find true best iteration and save image
        best_iter_pred = np.argmin(err_pred)
        best_err_pred = err_true[best_iter_pred]
        plt.title('Pred Best. '+'iter: '+str(best_iter_pred*traj_iter)+' err: {:.2f}'.format(best_err_pred),fontsize=12)
        plt.imshow(noisy_T[best_iter_pred])
        plt.savefig(os.path.join(output_dir,'best_pred.png'),bbox_inches='tight')
        plt.close()



