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


def dip(noisy_img,output_file,mask=None,
            lr=0.0001,
            niter=50000,
            traj_iter=1000,
            net_depth=5):
    
    #net = encdec.encdec(net_depth)

    pad = 'reflection'
    input_depth = 3
    output_depth = 3
    net = skip(input_depth, output_depth, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    net.cuda()

    eta = torch.randn(*noisy_img.size())
    eta = Variable(eta)
    eta = eta.cuda()

    fixed_target = noisy_img
    fixed_target = fixed_target.cuda()
    
    if not (mask is None):
        mask = torch.cat([mask]*eta.shape[1], 1)
        mask = mask.cuda()
    
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()
    T = []
    for itr in range(niter):
        optim.zero_grad()
        rec = net(eta)
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
    #np.savez_compressed(output_file,np.array(T,dtype=np.float32))
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
        '--net_depth', type=int, default=5, help='Depth of enc or dec'
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
    noisy_img = Variable(utils.preproc(noisy_img))
    if (len(mask_file) > 0):
        mask = utils.imread(mask_file)
        mask = Variable(utils.preproc(mask))
    else:
        mask = None
    #else:
    #    mask = np.ones((noisy_img.shape[0],noisy_img.shape[1]))

    dip(noisy_img, args.output_file, mask=mask,
            lr=args.lr,
            niter=args.niter,
            traj_iter=args.traj_iter,
            net_depth=args.net_depth)



