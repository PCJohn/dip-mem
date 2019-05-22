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
from scipy import signal
from scipy.ndimage import correlate

import utils
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Baselines for automatic stopping')
    parser.add_argument(
        '--output_dir', required=True, help='Folder with traj results'
    )
    parser.add_argument(
        '--clean_img', required=True, help='Path to clean image file'
    )
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    return parser.parse_args()

# Simple early stopping -- used fixed stopping time
def stop500(noisy_img, traj):
    return 500

def stop1500(noisy_img, traj):
    return 1500

# Pavel Mrazek -- min. corr between signal and noise
def corr_baseline(noisy_img, traj):
    def corrcoef(im1,im2):
        if len(im1.shape) > 2:
            corr = []
            for ch in range(im1.shape[-1]):
                v1,v2 = np.std(im1)**2, np.std(im2)**2
                cov = np.mean(np.multiply(im1-im1.mean(),im2-im2.mean()))
                corr.append(cov / np.sqrt(v1 * v2))
            return np.mean(corr)
        else:
            v1,v2 = np.std(im1)**2, np.std(im2)**2
            cov = np.mean(np.multiply(im1-im1.mean(),im2-im2.mean()))
            return cov / np.sqrt(v1 * v2)
    u0 = noisy_img
    corr = [corrcoef(u0-ut, ut) for ut in traj[1:]]
    return np.argmin(corr)


# Minimize Total Variation
def tv_baseline(noisy_img, traj):
    # TV norm from: https://gist.github.com/crowsonkb/ddf8167359be4ba2aa34835aa207e241  
    def total_var(x):
        x_diff = x - np.roll(x, -1, axis=1)
        y_diff = x - np.roll(x, -1, axis=0)
        grad_norm2 = x_diff**2 + y_diff**2 + np.finfo(np.float32).eps
        norm = np.sum(np.sqrt(grad_norm2))
        return norm
    tv = [total_var(ut) for ut in traj[1:]]
    return np.argmin(tv)


if __name__ == '__main__':
    args = parse_args()
    print(args)

    output_dir = args.output_dir.strip()
    traj_iter = cfg.TRAJ_ITER

    # load clean image and noisy image
    im = utils.imread(args.clean_img)
    noisy_im = utils.imread(args.noisy_img) 
   
    res = {'pred_best_err':1e10,'pred_best_psnr':0,'pred_best_hyp':'','pred_best_iter':0}
    baselines = {
                    'corr' : {'func':corr_baseline, 'res':res.copy()},
                    'tv' : {'func':tv_baseline,'res':res.copy()}
                }
 
    # make folders to save baseline preds
    baseline_output_dir = os.path.join(output_dir,'baselines')
    if not os.path.exists(baseline_output_dir):
        os.system('mkdir '+baseline_output_dir)
    for b in baselines.keys():
        save_dir = os.path.join(baseline_output_dir,b)
        baselines[b]['save_dir'] = save_dir
        if not os.path.exists(save_dir):
            os.system('mkdir '+save_dir)

    # dict to track final preds. of baselines
    final_preds = {b : None for b in baselines.keys()}

    traj_files = utils.traj_file_list(output_dir)
    for f in traj_files:
        hyp_str = utils.extract_hyp_str(f)

        noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
        noisy_T = utils.load_traj(noisy_output)

        # find pred. best iteration with different baselines
        for b in baselines.keys():
            best_iter_pred = baselines[b]['func'](noisy_im,noisy_T)
            best_rec_pred = noisy_T[best_iter_pred]
            best_err_pred = ((im - best_rec_pred)**2).sum()
            best_psnr_pred = compare_psnr(best_rec_pred.astype(im.dtype),im)
        
            if best_err_pred < baselines[b]['res']['pred_best_err']:
                baselines[b]['res']['pred_best_psnr'] = float(best_psnr_pred)
                baselines[b]['res']['pred_best_hyp'] = hyp_str
                baselines[b]['res']['pred_best_iter'] = int((best_iter_pred+1)*traj_iter)
                final_preds[b] = best_rec_pred

    # save final results and best predicted denoised image
    for b in baselines.keys():
        save_dir = baselines[b]['save_dir']
        with open(os.path.join(save_dir,'res.json'),'w') as f:
            f.write(json.dumps(baselines[b]['res'],indent=2))
        f.close()

        title = ' '.join([
                    'pred best iter: '+str(baselines[b]['res']['pred_best_iter']),
                    'psnr: {:.2f}'.format(baselines[b]['res']['pred_best_psnr'])
                    ])
        plt.title(title,fontsize=12)
        utils.imshow(final_preds[b])
        plt.savefig(utils.fname_with_hparams(save_dir,'best_pred.png',hyp_str),bbox_inches='tight')
        plt.close()







