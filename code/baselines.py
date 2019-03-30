import os
import sys
import cv2
import time
import argparse
import itertools
import json
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

import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--output_dir', required=True, help='Folder with traj results'
    )
    parser.add_argument(
        '--clean_img', required=True, help='Path to clean image file'
    )
    return parser.parse_args()


# Pavel Mrazek -- min. corr between signal and noise
def corr_baseline(traj):
    u0 = traj[0]
    corr = [corrcoef((ut-u0), ut) for ut in traj]
    return argmin(corr)



if __name__ == '__main__':
    
    baselines = {
                    'corr' : corr_baseline
                }
    
    args = parse_args()
    print(args)

    output_dir = args.output_dir.strip()
    traj_iter = 100

    # load clean image
    im = utils.imread(args.clean_img)
   
    res  =  {
                'pred_best_psnr':0,
                'pred_best_hyp':'',
                'pred_best_iter':0
            }
    final_best_pred = None
    
    traj_ext = '.npz'
    for f in os.listdir(output_dir):
        if not f.endswith(traj_ext):
            continue
        
        hyp_str = os.path.split(f)[-1].split(traj_ext)[0]
        hyp_str = hyp_str[hyp_str.index('_lr'):]
        
        noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
        noisy_T = utils.load_traj(noisy_output)

        # find pred. best iteration and save image
        best_iter_pred = 0
        best_rec_pred = noisy_T[best_iter_pred]
        best_psnr_pred = compare_psnr(best_rec_pred.astype(im.dtype),im)
        
        if best_err_pred < res['pred_best_err']:
            res['pred_best_psnr'] = float(best_psnr_pred)
            res['pred_best_hyp'] = hyp_str
            res['pred_best_iter'] = int((best_iter_pred+1)*traj_iter)
            final_best_pred = best_rec_pred

    # save final results
    with open(os.path.join(output_dir,'res.json'),'w') as f:
        f.write(json.dumps(res,indent=2))
    f.close()

    # save predicted best denoised image
    title = ' '.join([
                'pred best iter: '+str(res['pred_best_iter']),
                'psnr: {:.2f}'.format(res['pred_best_psnr'])
                ])
    plt.title(title,fontsize=12)
    utils.imshow(final_best_pred)
    plt.savefig(utils.fname_with_hparams(output_dir,'best_pred.png',hyp_str),bbox_inches='tight')
    plt.close()







