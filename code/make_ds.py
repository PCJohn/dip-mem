"""
Usage: 
        python code/make_ds.py denoise
        
        or,
        
        python code/make_ds.py inpaint
"""

import os
import cv2
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utils


def make_denoise_ds(data_dir):
    for img_dir in os.listdir(data_dir):
        img_name = os.path.split(img_dir)[-1]
        if len(img_name) == 0:
            img_name = os.path.split(img_dir)[-2]
        ext = '.png'
        clean_img = os.path.join(data_dir,img_dir,img_name+ext)
        img = utils.imread(clean_img)
        for sigma in [5,10,20,25,30,35,40,50,60,70,75,80,90,100]:
            output_file = os.path.join(data_dir,img_dir,img_name+'_s'+str(sigma)+ext)
            sigma = sigma/255.
            noisy_img = utils.get_noisy_image(img,sigma)
            utils.imwrite(output_file,noisy_img)


def make_inpaint_ds(data_dir):
    for img_dir in os.listdir(data_dir):
        img_name = os.path.split(img_dir)[-1]
        print(img_name)
        if len(img_name) == 0:
            img_name = os.path.split(img_dir)[-2]
        ext = '.png'
        img = utils.imread(os.path.join(data_dir,img_dir,img_name+ext))
        
        # read mask, if available. otherwise, generate maske dropping 50% pixels
        mask_path = os.path.join(data_dir,img_dir,img_name+'_mask'+ext)
        if os.path.exists(mask_path):
            mask = utils.imread(mask_path)
        else:
            mask = np.ones((img.shape[0],img.shape[1]))
            mask = utils.add_mask_noise(mask)
            utils.imwrite(mask_path,mask)

        masked_img = utils.mask_img(img,mask)

        output_file = os.path.join(data_dir,img_dir,img_name+'_noisy'+ext)
        utils.imwrite(output_file,masked_img)



if __name__ == '__main__':
    task = sys.argv[1]
    if task == 'denoise':
        data_dir = 'data/denoise'
        make_denoise_ds(data_dir)
    elif task == 'inpaint':
        data_dir = 'data/inpaint'
        make_inpaint_ds(data_dir)



