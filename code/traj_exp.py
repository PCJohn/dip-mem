import os
import sys
import cv2
import time
import yaml
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

#import config as cfg

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
    parser.add_argument(
        '--fixed_start', default=False, action='store_true', help='Start models from the same point'
    )
    parser.add_argument(
        '--cfg', required=True, type=str, help='Path to config file'
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
    act_fun = default_cfg['ACT']
    upsample = default_cfg['UPSAMPLE']
    net_structs = default_cfg['NET_STRUCT']
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
    if 'ACT' in cfg.keys():
        act_fun = cfg['ACT']
    if 'UPSAMPLE' in cfg.keys():
        upsample = cfg['UPSAMPLE']
    if 'NET_STRUCT' in cfg.keys():
        net_structs = cfg['NET_STRUCT']
    hparams = [lr_val,lang_sigma,init,init_scale,bn,depth,stride]
    ######    

    if args.task == 'denoise':
        added_noise_var = cfg['ADDED_NOISE']
        #net_structs = ['net_denoise']
        hparams.extend([net_structs,added_noise_var])
        
        noisy = utils.imread(noisy_path)
        for lr,lang,init,init_scale,bn,depth,stride,struct,added_var in itertools.product(*hparams):
            hyp_str = utils.gen_hyp_str(lr,depth,added_var,lang,init,init_scale,bn,stride)

            added_noisy_path = os.path.join(output_dir,'added_noise'+hyp_str+'.png')
            added_noisy = utils.add_noise('gauss',noisy,var=added_var)
            utils.imwrite(added_noisy_path,added_noisy)
            
            noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
            added_noisy_output = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
            for im_path,out_path in [(noisy_path,noisy_output),
                                     (added_noisy_path,added_noisy_output)]:
                cmd = 'sbatch gypsum/scripts/denoise.sh '+' '.join([im_path,out_path,
                        str(lr),str(niter),str(traj_iter),str(struct),str(int(args.fixed_start)),str(lang),str(reg_noise_std),
                        str(init),str(init_scale),str(bn),str(depth),str(stride),str(act_fun),str(upsample)])
                print('Submitting:',cmd)
                os.system(cmd)
    
    elif args.task == 'inpaint':
        drop_frac = cfg.DROP_FRAC
        net_structs = ['net_inpaint']
        hparams.extend([net_structs,drop_frac])
        
        mask_path = args.mask
        assert (len(mask_path) > 0)
        masked_img = utils.imread(noisy_path)
        mask = utils.imread(mask_path)

        for lr,lang,struct,drop_frac in itertools.product(*hparams):
            hyp_str = utils.gen_hyp_str(lr,struct,drop_frac,lang)
            
            noisy_mask_path = os.path.join(output_dir,'added_noise'+hyp_str+'_mask.png')
            mask_noisy = utils.add_mask_noise(mask,drop_frac=drop_frac)
            utils.imwrite(noisy_mask_path,mask_noisy)
            added_noisy_path = os.path.join(output_dir,'added_noise'+hyp_str+'.png')
            added_noisy = utils.mask_img(masked_img,mask_noisy)
            utils.imwrite(added_noisy_path,added_noisy)
            
            noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
            #noisy_rerun_output = utils.fname_with_hparams(output_dir,'T_noisy_rerun.npz',hyp_str)
            added_noisy_output = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
            for im_path,out_path,mask_path in [(noisy_path,noisy_output,mask_path),
                                               (added_noisy_path,added_noisy_output,noisy_mask_path)]:
                cmd = 'sbatch gypsum/scripts/inpaint.sh '+' '.join([im_path,out_path,
                        str(lr),str(niter),str(traj_iter),str(struct),str(mask_path),str(int(args.fixed_start)),str(lang),str(reg_noise_std)])
                print('Submitting:',cmd)
                os.system(cmd)





