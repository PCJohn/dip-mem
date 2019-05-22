import os
import sys
import cv2
import time
import yaml
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
from scipy.ndimage import correlate

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
    parser.add_argument(
        '--noisy_img', required=True, help='Path to noisy image file'
    )
    parser.add_argument(
        '--cfg', default='configs/default.yaml', help='Path to config file'
    )
    return parser.parse_args()

# scale to [0,1]
def rescale(im):
    #return np.clip(im,0,1)
    return (im-im.min())/(im.max()-im.min())


if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    
    # load given config file
    with open(args.cfg,'r') as f:
        cfg = yaml.load(f)
    traj_iter = cfg['TRAJ_ITER']

    output_dir = args.output_dir.strip()

    # load clean image
    im = utils.imread(args.clean_img)
    noisy_im = utils.imread(args.noisy_img)
   
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

    bandpass_filt_set = utils.bandpass_set(im.shape,s=5)
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
        err_true_added = []
        err_true_inv = []
        
        #true_noise = rescale(noisy_im-im)
        for t1,t2 in zip(noisy_T,added_noisy_T):
            err_true.append(((im - t1)**2).sum())
            err_traj.append(((t1 - t2)**2).sum())
            err_true_added.append(((im - t2)**2).sum())
            
            pred_inv = rescale(noisy_im - t1)
            err_true_inv.append(((im - pred_inv)**2).sum())

        # find best pred iters
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

        # find best pred for the inverted version: fitting the noise first
        best_iter_inv = np.argmin(err_true_inv)
        #print(err_true_inv)
        #print('>>>',best_iter_inv)
        best_err_true_inv = err_true_inv[best_iter_inv]
        best_rec_inv = rescale(noisy_im - noisy_T[best_iter_inv])
        best_psnr_true_inv = compare_psnr(best_rec_inv.astype(im.dtype),im)

        
        ######## use 2 trajectories to find the pred. best iteration #######
        best_err_pred = 1e20
        best_iter_pred = 0
        
        # select iteration just before degradation
        pos = np.argmax(err_traj[2:])-1
        best_err_pred = err_traj[pos]
        best_iter_pred = pos
        
        ###################################


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
            final_err_true_added = err_true_added.copy()


        # Save in separate folder for each hyp setting
        hyp_dir = os.path.join(output_dir,hyp_str)
        if not os.path.exists(hyp_dir):
            os.system('mkdir '+hyp_dir)
        # save variation of |FFT|
        x = [(i+1)*traj_iter for i in range(len(final_err_traj_pred))]
        fft_traj = [np.log(utils.fft(t)) for t in noisy_T]
        #fft_clean = utils.fft(im)
        #fft_err = [(ft-fft_clean)**2 for ft in fft_traj]  # squared error in fft components across channels
        fft_noisy = np.log(utils.fft(noisy_im))
        fft_err = [(ft-fft_noisy)**2 for ft in fft_traj]  # squared error in fft components across channels
        fft_iters = np.hstack([utils.power_variation(ft,bandpass_filt_set)[::2,np.newaxis] for ft in fft_err[:50]])
        plt.imshow(fft_iters,cmap='hot')
        #plt.locator_params(axis='y', nbins=3)
        #plt.locator_params(axis='x', nbins=3)
        #plt.imshow(fft_iters,cmap='hot',extent=[x[0],x[50],0,50])
        plt.gca().set_yticklabels(['',0]+['' for _ in range(6)]+[1])
        plt.gca().set_xticklabels(['',0]+['' for _ in range(3)]+[500])
        #plt.gca().set_xticklabels(x)
        plt.xlabel('Iterations')
        plt.ylabel('Normalized Frequency')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(hyp_dir,'fft_iters.png'),bbox_inches='tight')
        plt.close()

        # save selected iterations
        #for itr in [0,5,50,-1]:
        for itr in [0,50,100,500,1000,-1]:
            itr_img = noisy_T[itr//traj_iter]
            if len(itr_img.shape) == 2:
                plt.imshow(itr_img,cmap='gray')
            else:
                plt.imshow(itr_img)
            plt.savefig(os.path.join(hyp_dir,'iter-'+str(itr)+'.png'),bbox_inches='tight')
            plt.close()

            itr_fft = np.log(utils.fft(itr_img))
            plt.imshow(itr_fft,cmap='gray')
            plt.savefig(os.path.join(hyp_dir,'fft-'+str(itr)+'.png'),bbox_inches='tight')
            plt.close()

        # save true best denoised image for this hyp setting
        title = ' '.join(['true best iter: '+str(best_iter),'psnr: '+str(best_psnr_true)])
        plt.title(title,fontsize=12)
        utils.imshow(noisy_T[best_iter])
        plt.savefig(os.path.join(hyp_dir,'hyp_best_true.png'),bbox_inches='tight')
        plt.close()

        # save inv. best denoised image: best when fitting the noise first
        title = ' '.join(['true best inv. iter: '+str(best_iter_inv),'psnr: '+str(best_psnr_true_inv)])
        plt.title(title,fontsize=12)
        utils.imshow(best_rec_inv)
        plt.savefig(os.path.join(hyp_dir,'hyp_best_inv.png'),bbox_inches='tight')
        plt.close()


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
    plt.plot(x[:],final_err_true_pred[:])
    #plt.axvline(x=res['pred_best_iter'],label='pred. iter',c='black')
    #plt.legend()
    plt.savefig(utils.fname_with_hparams(output_dir,'err_true.png',hyp_str),bbox_inches='tight')
    plt.close()

    # save variation of true error for trajectory with added noise
    plt.title('True error of the added noise trajectory')
    x = [(i+1)*traj_iter for i in range(len(final_err_true_added))]
    plt.plot(x[:],final_err_true_added[:])
    #plt.axvline(x=res['pred_best_iter'],label='pred. iter',c='black')
    #plt.legend()
    plt.savefig(utils.fname_with_hparams(output_dir,'err_true_added.png',hyp_str),bbox_inches='tight')
    plt.close()

    # save variation of error between trajectories
    plt.title('Error between trajectories')
    x = [(i+1)*traj_iter for i in range(len(final_err_traj_pred))]
    plt.plot(x[:],final_err_traj_pred[:])
    #plt.axvline(x=res['pred_best_iter'],label='pred. iter',c='black')
    #plt.legend()
    plt.savefig(utils.fname_with_hparams(output_dir,'err_traj.png',hyp_str),bbox_inches='tight')
    plt.close()
    
    """
    # save variation of |FFT|
    x = [(i+1)*traj_iter for i in range(len(final_err_traj_pred))]
    fft_traj = [utils.fft(t) for t in noisy_T]
    fft_iters = np.hstack([utils.power_variation(ft)[::-1,np.newaxis] for ft in fft_traj])
    #fft_iters /= fft_iters.sum()
    #plt.plot(fft_iters)
    plt.imshow(fft_iters,cmap='hot')
    plt.xlabel('Iterations')
    plt.ylabel('|log F(k)|')
    plt.savefig(utils.fname_with_hparams(output_dir,'fft_iters.png',hyp_str),bbox_inches='tight')
    plt.close()
    """

    # save best denoised image
    title = ' '.join([
                'true best iter: '+str(res['true_best_iter']),
                'psnr: {:.2f}'.format(res['true_best_psnr'])
                ])
    plt.title(title,fontsize=12)
    utils.imshow(final_best_true)
    plt.savefig(utils.fname_with_hparams(output_dir,'best_true.png',hyp_str),bbox_inches='tight')
    plt.close()





