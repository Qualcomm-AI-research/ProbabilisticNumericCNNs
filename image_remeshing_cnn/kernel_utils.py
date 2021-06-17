# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
from oil.utils.utils import export


@export
def gaussian_blur(signal,std):
    """ assumes signal is of shape (bs,c,n1,...,nk) where n1,...,nk are dimensions to be blurred over.
        """
    filter_size = math.ceil(2*std)
    ix = torch.arange(-filter_size,filter_size+1,device=signal.device,dtype=signal.dtype)
    blur_weights = (-(ix/std)**2/2).exp()
    blur_weights = blur_weights/blur_weights.sum()
    bs,c,*spatial_dims = signal.shape
    out = signal
    blur_weights = blur_weights[None,None,:]
    for i,dim_size in enumerate(spatial_dims):
        transposed = out.transpose(i,-1)
        padded = F.pad(transposed.reshape(-1,1,dim_size),(filter_size,filter_size),mode='reflect')
        conved = F.conv1d(padded,weight=blur_weights)
        out = conved.reshape(*transposed.shape).transpose(i,-1)
    return out
