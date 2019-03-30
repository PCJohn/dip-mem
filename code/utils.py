import cv2
import os
import cv2
import torch
import errno
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def gen_hyp_str(lr,depth,added_var):
    return '_lr-'+str(lr)+'_d-'+str(depth)+'_var-'+str(added_var)

def fname_with_hparams(output_dir,fname,hyp_str):
    fl,ext = fname.split('.')[:2]
    return os.path.join(output_dir,fl+hyp_str+'.'+ext)

def imshow(image):
    if len(image.shape) == 2:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)

def imread(path):
    img = plt.imread(path).astype(float)
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    if img.max() > 1.0:
        img /= 255.0
    return np.array(img)

def imwrite(path,img):
    img = 255.0 * img
    if len(img.shape) == 2:
        #plt.imsave(path,np.uint8(img),cmap='gray')
        # use cv2 to writr 1-channel images: mplotlib adds channels
        cv2.imwrite(path,np.uint8(img))
    else:
        plt.imsave(path,np.uint8(img))

def preproc(img_n):
    if (len(img_n.shape) == 2):
        img_n = torch.FloatTensor(img_n).unsqueeze(0).unsqueeze(0).transpose(2,3)
    elif (len(img_n.shape) == 3):
        img_n = torch.FloatTensor(img_n).transpose(0,2).unsqueeze(0)
    return img_n

def shape(img):
    r,c = img.shape[:2]
    if len(img.shape) > 2:
        return (r,c,img.shape[2])
    else:
        return (r,c,1)

# Noise addition from the original DIP repo -- use to generate dataset
def get_noisy_image(img_np, sigma):
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    return img_noisy_np

# adding noise: 
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def add_noise(noise_typ,image,var=1e-4):
    row,col,ch = shape(image)
    if noise_typ == "gauss":
        mean = 0
        #var = 0.0001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(*image.shape)
        noisy = image + gauss
        # normalize back to [0,1]
        noisy = (noisy-noisy.min())/noisy.max()
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
        out[coords] = 0
        return out

def mask_img(img,mask):
    if len(img.shape) == 2:
        return np.multiply(img.copy(),mask)
    
    masked_img = img.copy()
    for ch in range(img.shape[-1]):
        masked_img[:,:,ch] = np.multiply(masked_img[:,:,ch],mask)
    return masked_img

def add_mask_noise(mask,drop_frac=0.5):
    w,h = mask.shape
    
    m = np.random.random((w,h))
    m[m < drop_frac] = 0
    m[m >= drop_frac] = 1
    
    m = np.multiply(mask,m)
    return m


def save_traj(path,traj_arr):
    # uint8 and compressed to save disk space
    traj_arr = np.array(traj_arr, dtype=np.float32)
    traj_arr = np.uint8(traj_arr * 255)
    np.savez_compressed(path, traj_arr)

def load_traj(path):
    traj_arr = np.load(path)['arr_0']
    traj_arr = np.float32(traj_arr) / 255.
    return traj_arr


