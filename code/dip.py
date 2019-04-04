import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from encdec import encdec
from skip import skip
import utils
import model_utils


def load_fixed_start(net_struct):
    if net_struct == 'net_denoise':
        pass
    elif net_struct == 'net_inpaint':
        pass
    return net


def dip(noisy_img,output_file,
            mask=None,
            lr=0.0001,
            niter=50000,
            traj_iter=1000,
            net_struct='',
            num_ch=3,
            fixed_start=False):
    
    #net = encdec.encdec(net_depth)

    pad = 'reflection'
    input_depth = num_ch
    output_depth = num_ch
    
    
    # net for denoising
    if net_struct == 'net_denoise':
        reg_noise_std = 0
        net = skip(input_depth, output_depth, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    
    # net for the inpainting
    elif net_struct == 'net_inpaint':
        reg_noise_std = 0.03
        net = skip(input_depth, output_depth,
            num_channels_down = [128]*5,
            num_channels_up   = [128]*5,
            num_channels_skip = [4]*5, 
            upsample_mode='bilinear')
    

    net.cuda()

    eta = torch.randn(*noisy_img.size())
    eta = Variable(eta).cuda()

    fixed_target = noisy_img
    fixed_target = fixed_target.cuda()
    
    if not (mask is None):
        mask = torch.cat([mask]*eta.shape[1], 1)
        mask = mask.cuda()
   
    optim = torch.optim.Adam(net.parameters(), lr=lr) # prev optim line
    mse = nn.MSELoss().cuda()
    T = []

    net_input = eta.clone().cuda()
    reg_noise = net_input.clone().cuda()
    
    for itr in range(niter):
        optim.zero_grad()
        
        if reg_noise_std > 0:
            net_input = eta + reg_noise.normal_() * reg_noise_std
        
        #rec = net(eta)
        rec = net(net_input)

        if not (mask is None):
            loss = mse(rec*mask,fixed_target*mask)
        else:
            loss = mse(rec,fixed_target)
        loss.backward()
        optim.step()
        if (itr%traj_iter == 0):
            T.append(rec[0, :, :, :].transpose(0,2).detach().cpu().data.numpy())
            print('Iteration '+str(itr)+': '+str(loss.data),T[-1].shape)
                                         
    final_out = net(eta)
    T.append(final_out[0, :, :, :].transpose(0,2).detach().cpu().data.numpy())
    
    # save trajectory
    utils.save_traj(output_file,T)



def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior')
    parser.add_argument(
        '--img_file', required=True, help='Path to image file'
    )
    parser.add_argument(
        '--mask_file', default='', help='Path to image file'
    )
    parser.add_argument(
        '--output_file', required=True, help='Folder to save trajectories'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='Learning rate'
    )
    parser.add_argument(
        '--niter', type=int, default=100000, help='Num iters'
    )
    parser.add_argument(
        '--traj_iter', type=float, default=1000, help='Traj. logging iter'
    )
    parser.add_argument(
        '--net_struct', required=True, help='Depth of enc or dec'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    img_file = args.img_file
    output_file = args.output_file
    mask_file = args.mask_file

    #load img and mask
    noisy_img = utils.imread(img_file)
    if len(noisy_img.shape) == 2:
        num_ch = 1
    else:
        num_ch = noisy_img.shape[-1]

    noisy_img = Variable(utils.preproc(noisy_img))
    if (len(mask_file) > 0):
        mask = utils.imread(mask_file)
        mask = Variable(utils.preproc(mask))
    else:
        mask = None

    dip(noisy_img, args.output_file, mask=mask,
            lr=args.lr,
            niter=args.niter,
            traj_iter=args.traj_iter,
            net_struct=args.net_struct,
            num_ch=num_ch)



