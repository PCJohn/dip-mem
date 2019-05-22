import cv2
import os
import cv2
import torch
import errno
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def gen_hyp_str(lr,depth,added_var,lang_sigma,init,init_scale,bn,stride):
    return '_lr-'+str(lr)+'_d-'+str(depth)+'_var-'+str(added_var)+'_L-'+str(lang_sigma)+'_init-'+str(init)+'_scale-'+str(init_scale)+'_bn-'+str(bn)+'_st-'+str(stride)

def extract_hyp_str(fname):
    hyp_str,_ = os.path.splitext(os.path.split(fname)[-1])
    return hyp_str[hyp_str.index('_lr'):]

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
    if traj_arr.shape[-1] == 1:
        traj_arr = traj_arr[:,:,:,0]
    return traj_arr

def traj_file_list(folder):
    traj_ext = '.npz'
    return [f for f in os.listdir(folder) if f.endswith(traj_ext)]



### FFT Utils ###
def channel_fft(ch):
    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    return mag

def fft(im):
    if len(im.shape) == 2:
        return channel_fft(im)
    else:
        return np.stack([channel_fft(im[:,:,ch]) for ch in range(im.shape[2])],axis=-1)

def band_pass_filter(im_shape,r,s,n_ch):
    w,h = im_shape[0],im_shape[1]
    f1 = np.zeros((w,h))
    f2 = np.zeros((w,h))
    cv2.circle(f1,(h//2,w//2),r+s,1,-1)
    cv2.circle(f2,(h//2,w//2),r,1,-1)
    filt = (f1-f2)
    if n_ch > 1:
        filt = np.stack([filt]*n_ch,axis=-1)
    #plt.imshow(filt,cmap='gray')
    #plt.savefig('filt.png')
    #import pdb; pdb.set_trace();
    return filt
   
def bandpass_set(im_shape,s=1):
    n_ch = 1
    if len(im_shape) == 3:
        n_ch = im_shape[-1]
    R = min(im_shape[0],im_shape[1])
    filt_bank = [band_pass_filter(im_shape,r,s,n_ch) for r in np.arange(s,R,s)]
    filt_bank = [f for f in filt_bank if f.sum() > 0]
    return filt_bank

def power_variation(img_fft,filt_bank):
    pow_var = np.array([(img_fft * filt).sum() for filt in filt_bank])
    return pow_var #/ pow_var.max()

### *** ###



