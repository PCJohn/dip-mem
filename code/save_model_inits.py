import os
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


model_root = 'models'


# nets for denoising
for input_depth,output_depth in [(1,1),(3,3)]:
    net = skip(input_depth, output_depth,
            num_channels_down = [8, 16, 32, 64, 128],
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
    model_path = os.path.join(model_root,'net_denoise_'+str(input_depth)+','+str(output_depth)+'.pth')
    torch.save(net,model_path)
                                
# nets for the inpainting
for input_depth,output_depth in [(1,1),(3,3)]:
    net = skip(input_depth, output_depth,
        num_channels_down = [128]*5,
        num_channels_up   = [128]*5,
        num_channels_skip = [4]*5,
        upsample_mode='bilinear')
    model_path = os.path.join(model_root,'net_inpaint_'+str(input_depth)+','+str(output_depth)+'.pth')
    torch.save(net,model_path)
    
    
    
    
    
    
    

 
