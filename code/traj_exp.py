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


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--clean_img', required=True, help='Path to image file'
    )
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    parser.add_argument(
        '--mask', default='', help='Path to the mask for inpainting'
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
    clean_path = args.clean_img
    noisy_path = args.noisy_img

    output_root = os.path.join(args.output_root,args.task)
    if not os.path.exists(output_root):
        os.system('mkdir '+output_root)
    
    exp_name = os.path.split(clean_path)[-1].split('.')[0]
    output_dir = os.path.join(output_root,exp_name)
    if not os.path.exists(output_dir):
        os.system('mkdir '+output_dir)

    im = utils.imread(clean_path)

    niter = 10000 # fixed, high value
    traj_iter = 100
    
    lr_val = [1e-1,1e-2,1e-3]
    net_depth = [5]
    hparams = [lr_val,net_depth]

    if args.task == 'denoise':
        added_noise_var = [1e-1,1e-2,1e-3,1e-4]
        hparams.append(added_noise_var)
        noisy = utils.imread(noisy_path)
        for lr,depth,added_var in itertools.product(*hparams):
            hyp_str = utils.gen_hyp_str(lr,depth,added_var)

            added_noisy_path = os.path.join(output_dir,'added_noise'+hyp_str+'.png')
            added_noisy = utils.add_noise('gauss',noisy,var=added_var)
            utils.imwrite(added_noisy_path,added_noisy)
            
            noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
            added_noisy_output = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
            for im_path,out_path in [(noisy_path,noisy_output),(added_noisy_path,added_noisy_output)]:
                cmd = 'sbatch gypsum/scripts/denoise.sh '+im_path+' '+out_path+' '+str(lr)+' '+str(niter)+' '+str(traj_iter)+' '+str(depth)
                print('Submitting:',cmd)
                os.system(cmd)
    
    elif args.task == 'inpaint':
        mask_path = args.mask
        noisy_mask_path = os.path.join(output_dir,'added_noise_mask.png')
        assert (len(mask_path) > 0)
        masked_img = utils.imread(noisy_path)
        mask = utils.imread(mask_path)
        mask_noisy = utils.add_mask_noise(mask)
        utils.imwrite(noisy_mask_path,mask_noisy)
        added_noisy = utils.mask_img(masked_img,mask_noisy)
        utils.imwrite(added_noisy_path,added_noisy)
        
        drop_frac = [0.5]
        hparams.append(drop_frac)
        for lr,depth,drop_frac in itertools.product(*hparams):
            #hyp_str = '_lr-'+str(lr)+'_d-'+str(depth)
            hyp_str = utils.gen_hyp_str(lr,depth,added_var)
            noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
            added_noisy_output = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
            for im_path,out_path,mask_path in [(noisy_path,noisy_output,mask_path),(added_noisy_path,added_noisy_output,noisy_mask_path)]:
                cmd = 'sbatch gypsum/scripts/inpaint.sh '+im_path+' '+out_path+' '+str(lr)+' '+str(niter)+' '+str(traj_iter)+' '+str(depth)+' '+str(mask_path)
                print('Submitting:',cmd)
                os.system(cmd)





