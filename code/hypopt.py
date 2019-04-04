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

import dip
import utils
import skip

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--output_dir', required=True, help='Folder with traj results'
    )
    parser.add_argument(
        '--clean_img', required=True, help='Path to clean image file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    
    traj_iter = 100
    
    args = parse_args()
    print(args)

    output_dir = args.output_dir.strip()

    # load clean image
    im = utils.imread(args.clean_img)
   
    res  =  {
                'true_best_err':1e10,
                'true_best_psnr':0,
                'true_best_hyp':'',
                'true_best_iter':0,
                'pred_best_err':1e10,
                'pred_best_psnr':0,
                'pred_best_hyp':'',
                'pred_best_iter':0
            }
    final_best_true = None
    final_best_pred = None
    final_err_true_pred = None

    traj_ext = '.npz'
    for f in os.listdir(output_dir):
        if not f.endswith(traj_ext):
            continue
        
        hyp_str = os.path.split(f)[-1].split(traj_ext)[0]
        hyp_str = hyp_str[hyp_str.index('_lr'):]
        
        noisy_output = utils.fname_with_hparams(output_dir,'T_noisy.npz',hyp_str)
        added_noisy_output = utils.fname_with_hparams(output_dir,'T_added.npz',hyp_str)
        
        noisy_T = utils.load_traj(noisy_output)
        added_noisy_T = utils.load_traj(added_noisy_output)
        
        assert (len(noisy_T) == len(added_noisy_T))

        err_true = []
        err_traj = []
        for t1,t2 in zip(noisy_T,added_noisy_T):
            err_true.append(((im - t1)**2).sum())
            err_traj.append(((t1 - t2)**2).sum())

        # find true best iteration and save image
        best_iter = np.argmin(err_true)
        best_err_true = err_true[best_iter]
        best_rec = noisy_T[best_iter]
        best_psnr_true = compare_psnr(best_rec.astype(im.dtype),im)
        if best_err_true < res['true_best_err']:
            res['true_best_err'] = float(best_err_true)
            res['true_best_psnr'] = float(best_psnr_true)
            res['true_best_hyp'] = hyp_str
            res['true_best_iter'] = int((best_iter+1)*traj_iter)
            final_best_true = best_rec.copy()

        
        # find pred. best iteration and save image
        best_err_pred = 1e10
        best_iter_pred = 0
        for it1,t1 in enumerate(noisy_T):
            for it2,t2 in enumerate(added_noisy_T):
                err = ((t1 - t2)**2).sum()
                if (err < best_err_pred):
                    best_err_pred = err
                    best_iter_pred = it1
        best_rec_pred = noisy_T[best_iter_pred]
        best_psnr_pred = compare_psnr(best_rec_pred.astype(im.dtype),im)
       
        print(best_err_pred,best_psnr_pred,hyp_str)
        
        if best_err_pred < res['pred_best_err']:
            res['pred_best_err'] = float(best_err_pred)
            res['pred_best_psnr'] = float(best_psnr_pred)
            res['pred_best_hyp'] = hyp_str
            res['pred_best_iter'] = int((best_iter_pred+1)*traj_iter)
            final_best_pred = best_rec_pred.copy()
            final_err_true_pred = err_true.copy()
            final_err_traj_pred = err_traj.copy()

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

    # save variation of true error for the best predicted params
    plt.title('True error with predicted params')
    x = [(i+1)*traj_iter for i in range(len(final_err_true_pred))]
    plt.subplot(121)
    plt.plot(x,final_err_true_pred)
    plt.subplot(122)
    plt.plot(x[10:],final_err_true_pred[10:])
    plt.axvline(x=res['pred_best_iter'],label='pred. iter',c='black')
    plt.legend()
    plt.savefig(utils.fname_with_hparams(output_dir,'err_true.png',hyp_str),bbox_inches='tight')
    plt.close()

    # save variation of error between trajectories
    plt.title('True error with predicted params')
    x = [(i+1)*traj_iter for i in range(len(final_err_traj_pred))]
    plt.subplot(121)
    plt.plot(x,final_err_traj_pred)
    plt.subplot(122)
    plt.plot(x[10:],final_err_traj_pred[10:])
    plt.axvline(x=res['pred_best_iter'],label='pred. iter',c='black')
    plt.legend()
    plt.savefig(utils.fname_with_hparams(output_dir,'err_traj.png',hyp_str),bbox_inches='tight')
    plt.close()

    # save best denoised image
    title = ' '.join([
                'true best iter: '+str(res['true_best_iter']),
                'psnr: {:.2f}'.format(res['true_best_psnr'])
                ])
    plt.title(title,fontsize=12)
    utils.imshow(final_best_true)
    plt.savefig(utils.fname_with_hparams(output_dir,'best_true.png',hyp_str),bbox_inches='tight')
    plt.close()





