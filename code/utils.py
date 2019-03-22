import cv2
import os
import cv2
import torch
import errno
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def imread(path):
    img = plt.imread(path).astype(float)
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    if img.max() > 1.0:
        img /= 255.0
    return img

def imwrite(path,img):
    img = 255.0 * img
    cv2.imwrite(path,img)

def preproc(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).transpose(2,3)

def shape(img):
    r,c = img.shape[:2]
    if len(img.shape) > 2:
        return (r,c,img.shape[2])
    else:
        return (r,c,1)

# adding noise: 
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def add_noise(noise_typ,image):
    row,col,ch = shape(image)
    if noise_typ == "gauss":
        mean = 0
        var = 0.005
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(*image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
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
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        gauss = np.random.randn(*image.shape)
        gauss = gauss.reshape(*image.shape)        
        noisy = image + image * gauss
        return noisy


