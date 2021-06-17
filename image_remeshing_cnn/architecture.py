# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from oil.utils.utils import export,Expression
from image_remeshing_cnn.kernel import Phi,LinearOperatorGP, IntegralGP,RBFwBlur,\
    MaskedLinearOpGP, MultiChannelIntegralGP, HeatRBF, phi
from image_remeshing_cnn.kernel import GPinterpolateLayer,ChannelSeperateLinOpGP
from functools import partial

class Linear(nn.Linear):
    """ Probabilistic Numeric equivalent of a nn.Linear layer. Propagates uncertainty
        through the linear layer as stdout = \sqrt{(W*W) @ std} where * is elementwise product
         (unless nounc=True in which case outputs 0s for the uncertainty.)"""
    def __init__(self,*args,nounc=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.nounc = nounc
    def forward(self,x):
        coords,meanin,stdin = x[:3]

        meanout = super().forward(meanin)
        if self.nounc:
            return (coords,meanout,torch.zeros_like(meanout))+x[3:]
        stdout = ((stdin**2)@(self.weight.T**2)).clamp(min=1e-7).sqrt()
        out = (coords,meanout,stdout)
        #mask_out = (x[3].sum(-1)>0).float().unsqueeze(-1).repeat((1,1,meanout.shape[-1]))
        return out+x[3:]

class sReLU(nn.Module):
    """ Probabilistic Numeric version of the ReLU. Propagates the mean and uncertainty through
        the ReLU assuming the input follows a normal distribution."""
    def __init__(self,nounc=False):
        super().__init__()
        self.nounc=nounc
    def forward(self,x):
        coords,meanin,stdin = x[:3]
        if self.nounc:
            return (coords,meanin.relu(),torch.zeros_like(stdin))+x[3:]
        stdin = (stdin+1e-6).clamp(min=1e-6)
        Phis = Phi(meanin/stdin)
        density = phi(meanin/stdin)
        meanout = meanin*Phis + stdin*density
        stdout = (((meanin**2+stdin**2)*Phis+meanin*stdin*density-meanout**2)+1e-6).clamp(min=1e-6).sqrt()
        return (coords,meanout,stdout) + x[3:]

    
def ReLuLinearConvGP(chin,chout,kernel=RBFwBlur,d=2,nounc=False):
    return nn.Sequential(LinearOperatorGP(chin,chout,kernel=kernel,d=d,nounc=nounc),
                         sReLU(nounc=nounc),Linear(chout,chout,nounc=nounc))

def pthash(xyz):
    """ a hash function for pytorch arrays """
    return hash(tuple(xyz.cpu().data.numpy().reshape(-1)))


class AvgPool(nn.Module):
    def forward(self,x):
        coords,vals,stds,ids = x
        return coords,vals.mean(1),(stds**2).mean(1).sqrt()


@export
class PNCNN(nn.Module):
    """ Main version of the Probabilistic Numeric CNN. Intermediate number of channels k, number of layers num_layers
        X dimension d, and choice of operator via kernel, nounc disables uncertainty, and num_basis is used if the
        operator kernel is HeatRBF."""
    def __init__(self,channels_in=1,k=64,num_layers=3,num_targets=10,d=2,
                 kernel=HeatRBF,nounc=False,num_basis=None):
        super().__init__()
        self._unc = nn.Parameter(torch.tensor(-2.))
        if num_basis: kernel = partial(kernel,num_basis=num_basis)
        self.network = nn.Sequential(ReLuLinearConvGP(channels_in,k,kernel=kernel,d=d,nounc=nounc),
                                     *[ReLuLinearConvGP(k,k,kernel=kernel,d=d,nounc=nounc) for i in range(num_layers-1)],
                                     IntegralGP(d=d,nounc=nounc),
                                     Linear(k,num_targets,nounc=nounc),
                        )
        #self.network = ReLuLinearConvGP(channels_in, num_targets, datasize, kernel=kernel, d=d, nounc=nounc)
        self.std_tracker = 0
        self.logit_sqr = 0
        self.logit_mean =0

    def forward(self,x):
        coords,vals = x
        ids = None#self.get_ids(coords) if self.id_map is not None else None
        stdin = torch.ones_like(vals)*F.softplus(self._unc)#.to(vals.device)
        _,mean,std,_ = self.network((coords,vals,stdin,ids))
        self.std_tracker += .01*(std.mean().cpu().item()-self.std_tracker)
        
        predictions = mean
        self.logit_sqr += .01*((predictions**2).mean().cpu().data.item()-self.logit_sqr)
        self.logit_mean += .01*(predictions.mean().cpu().data.item()-self.logit_mean)
        return predictions
    
    def log_data(self,logger,step,name):
        pass
        #logger.add_scalars('info', {f'out_std':self.std_tracker}, step=step)
        #logger.add_scalars('info',{f'logit_std':np.sqrt(self.logit_sqr-self.logit_mean**2)},step=step)



Swish = Expression(lambda x: x.sigmoid()*x)

class Dropout(nn.Module):
    def __init__(self,p=.2):
        super().__init__()
        self.p = p

    def forward(self,x):
        if not self.training: return x
        coordsin, meanin, stdin, mask, ids = x
        bs, n, d = coordsin.shape
        not_dropped = (torch.rand_like(meanin)>self.p).float()
        new_mask = not_dropped if mask is None else not_dropped*mask
        return (coordsin, meanin*not_dropped, stdin, new_mask, ids)

def ConvBlock(chin, chout,  kernel=RBFwBlur, d=2,p=0,probes=20):
    return nn.Sequential(Dropout(p) if p else nn.Sequential(),
                         MaskedLinearOpGP(chin, chout,  kernel=kernel, d=d,probes=probes),
                         sReLU(),
                         Linear(chout, chout))
@export
class GetMean(nn.Module):
    def forward(self,x):
        return x[1]




@export
class PhysioPNCNN_simple(PNCNN):
    """ Simple Version of PNCNN for PhysIONet. First interpolates the data onto
        a grid with mean and standard deviation with GPinterpolateLayer
        and then follows with GP Linear Operator that are the same over the different channels.
        Demographics information ('Age','Gender','Height','ICUType','Weight') are converted into
        time series with constant values.
        """
    def __init__(self,channels_in=1,k=64,num_layers=3,num_targets=1,d=2,
                 kernel=HeatRBF,num_basis=5,nounc=False):
        super().__init__()
        self._gp_unc = nn.Parameter(torch.tensor(-1.))
        kernel = partial(kernel,num_basis=num_basis)#partial(RBFwBlur,d=d)
        self.network = nn.Sequential(
            GPinterpolateLayer(channels_in,d=d,nounc=nounc),
            ReLuLinearConvGP(channels_in,k,d=d,kernel=kernel,nounc=nounc),
            *[ReLuLinearConvGP(k,k,d=d,kernel=kernel,nounc=nounc) for i in range(num_layers-1)],
            IntegralGP(d=d,nounc=nounc),
            Linear(k,num_targets,nounc=nounc),
            GetMean())

    def forward(self,x):
        coords, vals,demos,mask = x
        if demos.shape[-1]:
            const_time_demos = demos.unsqueeze(1).repeat((1, vals.shape[1], 1))
            any_vitals_mask = (mask.sum(-1) > 0).float().unsqueeze(-1).repeat((1, 1, const_time_demos.shape[-1]))
            demos_mask = (~torch.isnan(const_time_demos)).float()*any_vitals_mask
            mask = torch.cat([mask, demos_mask], dim=2)
            zeroed_demos = torch.where(demos_mask>0,const_time_demos,torch.zeros_like(const_time_demos))
            vals = torch.cat([vals, zeroed_demos], dim=2)
        ids =None# self.get_ids(coords) if self.id_map is not None else None
        stdin = torch.ones_like(vals) * F.softplus(self._unc)

        assert not torch.isnan(mask).any(), mask
        return self.network((coords, vals, stdin, mask,ids))

def CReLuLinearConvGP(chin,chout,kernel=RBFwBlur,d=2,nounc=False):
    return nn.Sequential(ChannelSeperateLinOpGP(chin,chout,kernel=kernel,d=d,nounc=nounc),
                         sReLU(nounc=nounc),Linear(chout,chout,nounc=nounc))
@export
class PhysioPNCNN_interp_sepchannels(PhysioPNCNN_simple):
    """ Like PhysioPNCNN_simple but treats the channels separately with their own
        uncertainties in the network. ChannelSeperateLinOpGP instead of LinearOperatorGP """
    def __init__(self,channels_in=1,k=64,num_layers=3,num_targets=1,d=2,
                 kernel=HeatRBF,num_basis=5,nounc=False,res=100):
        super().__init__()
        self._gp_unc = nn.Parameter(torch.tensor(-1.))
        kernel = partial(kernel,num_basis=num_basis)#partial(RBFwBlur,d=d)
        self.network = nn.Sequential(
            GPinterpolateLayer(channels_in,d=d,nounc=nounc,res=res),
            CReLuLinearConvGP(channels_in,k,d=d,kernel=kernel,nounc=nounc),
            *[CReLuLinearConvGP(k,k,d=d,kernel=kernel,nounc=nounc) for i in range(num_layers-1)],
            IntegralGP(d=d,nounc=nounc),
            Linear(k,num_targets,nounc=nounc),
            GetMean())
