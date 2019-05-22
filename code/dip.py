import os
import argparse
import numpy as np
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from encdec import encdec
from skip import skip
import utils
import model_utils


def langevin_grad(grad, sigma=0.01):
    return grad + Variable(grad.data.new(grad.size()).normal_(0,sigma))

def apply_langevin_grad(m, langevin_sigma=0.01):
    if (type(m) in [nn.Linear, nn.Conv2d]):
        m.weight.register_hook(functools.partial(langevin_grad, sigma=langevin_sigma))
        if m.bias is not None:
            m.bias.register_hook(functools.partial(langevin_grad, sigma=langevin_sigma))


def load_fixed_start(net_struct, input_depth, output_depth):
    model_path = os.path.join('models/',net_struct+'_'+str(input_depth)+','+str(output_depth)+'.pth')
    return torch.load(model_path)

##### Diffferent init. methods #####
def xavier_init(m,scale):
    if isinstance(m, nn.Conv2d):
        f_ratio = (scale / (np.prod(m.weight.shape[1:]) + m.weight.shape[0]))**0.5
        filt = f_ratio * (2 * torch.rand(m.weight.shape) - 1) # uniform
        m.weight = torch.nn.Parameter(filt.cuda())

def uniform_init(m,scale):
    if isinstance(m, nn.Conv2d):
        filt = scale * torch.rand(m.weight.shape)
        m.weight = torch.nn.Parameter(filt.cuda())

def normal_init(m,scale):
    if isinstance(m, nn.Conv2d):
        #f_ratio = (scale / (np.prod(m.weight.shape[1:]) + m.weight.shape[0]))**0.5
        #filt = f_ratio * torch.randn(m.weight.shape) # normal
        filt = scale * torch.randn(m.weight.shape) # normal
        m.weight = torch.nn.Parameter(filt.cuda())
##########


class AbsAct(nn.Module):
    def __init__(self):
        super(AbsAct, self).__init__()
       
    def forward(self, x):
        return torch.abs(x)
        #return torch.pow(F.relu(x),4)


def dip(noisy_img, output_file,
           mask=None,
            lr=0.0001,
            niter=50000,
            traj_iter=1000,
            net_struct='',
            num_ch=3,
            fixed_start=False,
            langevin=0.0,
            tv_reg=0.0,
            reg_noise_std=0.0,
            n_ch_down=128,
            n_ch_up=128,
            skip_conn=4,
            depth=5,
            stride=2,
            init='default',
            init_scale=-1,
            bn=True,
            act_fun='LeakyReLU',
            upsample='bilinear'):
    
    pad = 'reflection'
    input_depth = num_ch
    output_depth = num_ch

    #n_ch_down=4096//2
    #in_ch_up=4096//2
    use_fc = (net_struct=='fc')
    if use_fc:
        skip_conn = 0
    net = skip(input_depth, output_depth,
            num_channels_down = [n_ch_down]*depth,
            num_channels_up   = [n_ch_up]*depth,
            num_channels_skip = [skip_conn]*depth, 
            upsample_mode=upsample,
            act_fun=act_fun,
            stride=stride,
            use_bn=bn,
            use_fc=use_fc,
            input_shape=noisy_img.shape)

    if fixed_start:
        net = load_fixed_start(net_struct,input_depth,output_depth)

    if langevin > 0:
        net.apply(functools.partial(apply_langevin_grad, langevin_sigma=langevin*lr))

    if init == 'xavier':
        net.apply(lambda t: xavier_init(t,init_scale))
    elif init == 'uniform':
        net.apply(lambda t: uniform_init(t,init_scale))
    elif init == 'normal':
        net.apply(lambda t: normal_init(t,init_scale))

    net.cuda()

    net = nn.DataParallel(net)

    weight_decay = 0
    #if langevin > 0:
    #    grad_noise = [torch.zeros_like(param) for param in net.parameters()]
    #    #weight_decay = (5e-8*1024*1024)/(noisy_img.shape[0]*noisy_img.shape[1]) # as in sgld paper

    eta = torch.randn(*noisy_img.size())
    
    ######### load some weird image ##########
    #import cv2;import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as plt;
    #eta = np.array(plt.imread('data/starry_night.jpg'))
    #eta = cv2.resize(eta,(256,256)).astype(float) / 255.0
    #plt.imshow(eta); plt.savefig('eta.png');
    #eta = utils.preproc(eta)
    #############################

    eta = Variable(eta).cuda()
    reg_noise = Variable(torch.zeros_like(eta)).cuda()

    fixed_target = noisy_img
    fixed_target = fixed_target.cuda()
    
    if not (mask is None):
        mask = torch.cat([mask]*eta.shape[1], 1)
        mask = mask.cuda()
    
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss().cuda()
    T = []

    for itr in range(niter):
        optim.zero_grad()
        
        # add reg noise for inpainting
        if reg_noise_std > 0:
            rec = net(eta + reg_noise.normal_() * reg_noise_std)
        else:
            rec = net(eta)
        
        rec = net(eta)

        if not (mask is None):
            loss = mse(rec*mask,fixed_target*mask)
        else:
            loss = mse(rec,fixed_target)
        loss.backward(retain_graph=True)
        
        optim.step()
        if (itr%traj_iter == 0):
            out_np = rec[0, :, :, :].transpose(0,2).detach().cpu().data.numpy()
            T.append(out_np)
            print('Iteration '+str(itr)+': '+str(loss.data),T[-1].shape)
                                         
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
    parser.add_argument(
        '--fixed_start', type=int, default=0, help='Start traj. from the same point'
    )
    parser.add_argument(
        '--langevin', type=float, default=0, help='Variance of langevin noise reg.'
    )
    parser.add_argument(
        '--tv_reg', type=float, default=0, help='Weight on TV norm reg.'
    )
    parser.add_argument(
        '--reg_noise_std', type=float, default=0, help='Var. of noise added to the input as a regularizer'
    )
    parser.add_argument(
        '--n_ch_down', type=int, default=128, help='Num channels for downsampling'
    )
    parser.add_argument(
        '--n_ch_up', type=int, default=128, help='Num channels for upsampling'
    )
    parser.add_argument(
        '--skip_conn', type=int, default=4, help='Skip connection indices'
    )
    parser.add_argument(
        '--depth', type=int, default=5, help='Enc-Dec depth'
    )
    parser.add_argument(
        '--stride', type=int, default=2, help='Stride for down and up sampling'
    )
    parser.add_argument(
        '--fc', default=False, action='store_true', help='Use a model with only FC layers'
    )
    parser.add_argument(
        '--init', default='default', type=str, help='Type of weight initialization'
    )
    parser.add_argument(
        '--init_scale', default=1.0, type=float, help='Scale weight initialization'
    )
    parser.add_argument(
        '--bn', default=1, type=int, help='Use batch norm in model'
    )
    parser.add_argument(
        '--act_fun', default='LeakyReLU', help='Activation function'
    )
    parser.add_argument(
        '--upsample', default='bilinear', help='Upsampling method'
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
            num_ch=num_ch,
            fixed_start=(args.fixed_start!=0),
            langevin=args.langevin,
            tv_reg=args.tv_reg,
            reg_noise_std=args.reg_noise_std,
            n_ch_down=args.n_ch_down,
            n_ch_up=args.n_ch_up,
            skip_conn=args.skip_conn,
            depth=args.depth,
            stride=args.stride,
            init=args.init,
            init_scale=args.init_scale,
            bn=(args.bn==1),
            act_fun=args.act_fun,
            upsample=args.upsample)




