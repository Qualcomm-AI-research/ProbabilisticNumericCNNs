# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from image_remeshing_cnn.kernel import LinearOperatorGP,HeatRBF,RBF,Phi, IntegralGP
from image_remeshing_cnn.architecture import GetMean,AvgPool
from image_remeshing_cnn.equivariant.symmetric_subspaces import multilinear_active_subspaces,\
    repsize,T,Scalar,Vector,Matrix,Quad,get_active_subspaces

import itertools

class EquiHeatRBF(HeatRBF):
    """ Replaces the Diffusion RBF opkernel with one that acts equivariantly
        according to a given representation repmiddle. (Assumes rotation equivariance)"""
    def __init__(self,repmiddle,d=2):
        RBF.__init__(self,d=d)
        self.ksize = repsize(repmiddle,d)
        rep_heat = T(1,0)+T(2,0) # The representation of combined drift and diffusion
        rot_gen = [np.array([[0, -1], [1, 0]])] # The generator for rotation in 2d
        active_dims, self.projection = multilinear_active_subspaces(rot_gen,[repmiddle,rep_heat],[])
        self._heat_params = nn.Parameter(torch.randn(active_dims)*.1)

    def beta_sigma(self):
        """ Computes the drift and diffusion parameters"""
        beta_L = self.projection(self._heat_params).reshape(self.ksize,-1)
        beta = beta_L[:,:self.d]
        L = beta_L[:,self.d:].reshape(self.ksize,self.d,self.d)
        sigma = L@L.permute(0,2,1)
        #print(f"sigma = {sigma}")
        #print(f"beta = {beta}")
        #print(f"sigma0 = {sigma[0]}")
        return beta,sigma

class EquiDerivRBFwBlur(RBF):
    """ Replaces the Diffusion RBF opkernel with one that acts equivariantly
        according to the given representation of [1,\nabla,\nabla\nabla^T]. (Assumes rotation equivariance)"""
    def __init__(self,d=2):
        RBF.__init__(self,d=d)
        self.repmiddle = T(0, 0) + T(0, 1) + T(0, 2)  # The representation of combined drift and diffusion
        self.ksize = repsize(self.repmiddle,d)
        self._blur_lengthscale = nn.Parameter(torch.tensor(-1.))

    def gradient(self,delta,sigma):
        return (-delta/sigma**2)

    def hessian(self,delta,sigma):
        bs, *nm, d = delta.shape
        hess = (delta[...,:,None]*delta[...,None,:]/sigma**4)-torch.eye(d,device=delta.device)/sigma**2
        return hess.reshape(bs,*nm,-1)

    def deriv3(self,delta,sigma):
        bs, *nm, d = delta.shape
        x = delta/sigma
        I = torch.eye(d,device=delta.device)
        xI = x[...,None,None]*I
        xxx = x[...,None,None]*x[...,None,:,None]*x[...,None,None,:]
        p3 = np.arange(3) #spatial indices
        b = len(delta.shape[:-1]) # number of non spatial indices
        rb = np.arange(b) # non spatial indices for permutation
        d3 = (xI+xI.permute(*rb,*np.roll(p3+b,-1)) + xI.permute(*rb,*np.roll(p3+b,-2))-xxx)/sigma**3
        return d3.reshape(*delta.shape[:-1],-1)

    def deriv4(self,delta,sigma):
        bs, *nm, d = delta.shape
        x = delta / sigma
        xxxx = x[...,None,None,None]*x[...,None,:,None,None]*x[...,None,None,:,None]*x[...,None,None,None,:]
        I = torch.eye(d, device=delta.device)
        II = I[:,:,None,None]*I
        I_terms = II + II.permute(0,2,1,3)+II.permute(0,3,1,2)
        p4 = np.arange(4)  # spatial indices
        b = len(delta.shape[:-1])  # number of non spatial indices
        rb = np.arange(b)  # non spatial indices for permutation
        xxI= (x[...,None,None,None]*x[...,None,:,None,None]*I)#.permute(*rb,5,6,3,4)
        sym_perms = itertools.permutations(b+np.arange(4))
        xxI_terms = sum(xxI.permute(*rb,*perm) for perm in sym_perms)
        return (xxxx + I_terms - xxI_terms).reshape(bs,*nm,-1)/sigma**4

    @property
    def blur_lengthscale(self):
        return F.softplus(self._blur_lengthscale)

    def opiK(self,x_m,x_n):
        sigma = (self.length_scale**2 + self.blur_lengthscale**2).sqrt()
        delta = (x_m[:, :, None, :] - x_n[:, None, :, :]) # (bs,m,n,d)
        bs,m,n,d = delta.shape
        deriv_orders = self.Dk(delta,sigma)[:,:3] # (bs,3,m,n,d)
        scalar = self.K(x_m,x_n,new_lengthscale=sigma)[...,None] # (bs,m,n,1)
        gradient = scalar*self.gradient(delta,sigma)#deriv_orders[:,1]*scalar
        # (bs,d,d,
        hessian = scalar*self.hessian(delta,sigma)
        scalar_grad_hessian = torch.cat([scalar,gradient,hessian],dim=-1).permute(0,3,1,2) # (bs,m,n,1+d+d*d)->(bs,k,m,n)
        return scalar_grad_hessian#[:,1+d:]


    def opiKopjfull(self, x_m, x_n):
        sigma = (self.length_scale ** 2 + 2*self.blur_lengthscale ** 2).sqrt()
        delta = (x_m[:, :, None, :] - x_n[:, None, :, :])  # (bs,m,n,d)
        bs, m, n, d = delta.shape
        # construct matrix [1, d, d^2][1, d', d'^2]^T = [[1 , -d, d^2],[d,-d^2,d^3],[d^2,-d^3,d^4]]
        #scalar = self.K(x_m, x_n, new_lengthscale=sigma)  # (bs,m,n)
        scalar =torch.ones_like(delta[...,:1])
        d1 = self.gradient(delta,sigma)
        # (bs,d,d,
        d2 = self.hessian(delta,sigma)
        d3 = self.deriv3(delta,sigma)
        d4 = self.deriv4(delta,sigma)
        op1j = torch.cat([scalar[...,None],d1[...,None],d2[...,None]],dim=-2)
        op2j = -torch.cat([d1.reshape(bs,m,n,1,d), d2.reshape(bs,m,n,d,d), d3.reshape(bs,m,n,d*d,d)], dim=-2)
        op3j = torch.cat([d2.reshape(bs,m,n,1,d*d), d3.reshape(bs,m,n,d,d*d), d4.reshape(bs,m,n,d*d,d*d)],dim=-2)
        opij = torch.cat([op1j,op2j,op3j],dim=-1) #(bs,m,n,1+d+d^2,1+d+d^2)
        return (opij*self.K(x_m, x_n, new_lengthscale=sigma)[...,None,None]).permute(0,3,4,1,2)

    def opiKopj(self, x_m):
        sigma = (self.length_scale ** 2 + 2*self.blur_lengthscale ** 2).sqrt()
        delta = torch.zeros_like(x_m) # (bs,m,d)
        bs, m, d = delta.shape
        D = self.Dk(delta, sigma)[:, :5]  # (bs,5,m,d)
        # construct matrix [1, d, d^2][1, d', d'^2]^T = [[1 , -d, d^2],[d,-d^2,d^3],[d^2,-d^3,d^4]]
        #scalar = self.K(x_m, x_n, new_lengthscale=sigma)  # (bs,m)
        scalar =torch.ones_like(delta[...,:1])
        d1 = self.gradient(delta,sigma)
        # (bs,d,d,
        d2 = self.hessian(delta,sigma)
        d3 = self.deriv3(delta, sigma)
        d4 = self.deriv4(delta, sigma)
        op1j = torch.cat([scalar[...,None],d1[...,None],d2[...,None]],dim=-2)
        op2j = -torch.cat([d1.reshape(bs,m,1,d), d2.reshape(bs,m,d,d), d3.reshape(bs,m,d*d,d)], dim=-2)
        op3j = torch.cat([d2.reshape(bs,m,1,d*d), d3.reshape(bs,m,d,d*d), d4.reshape(bs,m,d*d,d*d)],dim=-2)
        opij = torch.cat([op1j,op2j,op3j],dim=-1) #(bs,n,1+d+d^2,1+d+d^2) -> (bs,k,k,n)
        return (opij*self.K0(delta, new_lengthscale=sigma)[...,None,None]).permute(0,2,3,1)


def append_gate_scalars(rep):
    """ Adds a scalar at the end of the representation for each non scalar
        in the input representation. Meant to be used with the gated nonlinearity
        for the extra scalar gate channels."""
    return rep+sum((1 for tensor in rep if tensor!=Scalar[0]))*Scalar

class EquivariantLinearOperatorGP(LinearOperatorGP):
    """ Variant of LinearOperatorGP layer that is steerably rotation equivariant
        according to a given representations repin and repout. Equivariant nonlinearities
        are by default included in the layer but can be disabled like for uncertainty."""
    def __init__(self, repin, repout, d=2,nonlinearity=True,nounc=False):
        nn.Module.__init__(self) #
        #repmiddle=9*Scalar # Temporary for debugging equivariance
        #T(0,2)#T(0,0)+T(0,1)+T(0,2)
        self.nounc=nounc
        self.d=d
        rot_gen = [np.array([[0,-1],[1,0]])]
        self.opkernel = EquiDerivRBFwBlur(d=d)
        repmiddle = self.opkernel.repmiddle
        repoutplus = append_gate_scalars(repout) if nonlinearity else repout
        self.repout = repout
        self.rep_sizes = repsize(repoutplus,d),repsize(repmiddle,d),repsize(repin,d)
        active_dims, self.projection = multilinear_active_subspaces(rot_gen,[repoutplus],[repmiddle,repin])
        bias_active_dims, self.bias_projection = get_active_subspaces(rot_gen,repoutplus)

        init_std = 1/np.sqrt(np.prod(self.rep_sizes))
        self._weight_params = nn.Parameter(torch.randn(active_dims)*init_std)
        self._bias_params = nn.Parameter(torch.zeros(bias_active_dims))
        self.nonlinearity = nonlinearity
        #assert nonlinearity
        self.coords_embedding =  None
        self.initialized_coord_ids = set()
        self.jitter_coords = False

    @property
    def op_weight(self):
        W = self.projection(self._weight_params) # (repout,repmiddle,repin)
        assert W.shape==self.rep_sizes
        outsize, middlesize, insize = self.rep_sizes
        #print(W)
        return W.permute(0,2,1).reshape(outsize,insize*middlesize)

    @property
    def op_bias(self):
        b = self.bias_projection(self._bias_params)  # (repout,)
        #print(b)
        return b

    def op_linear(self,x):
        """x (*,insize*middlesize)
            out (*,outsize)"""
        return x@self.op_weight.T + self.op_bias

    def forward(self,x):
        assert not torch.isinf(self.op_weight).any()
        coords, outputs_and_gates,outputs_and_gates_std,ids = super().forward(x)
        if not self.nonlinearity: return (coords,outputs_and_gates,outputs_and_gates_std,ids)
        assert not torch.isnan(outputs_and_gates).any()
        assert not torch.isnan(outputs_and_gates_std).any()
        outdims = repsize(self.repout,d=self.d)
        outputs = outputs_and_gates[...,:outdims]
        gates = outputs_and_gates[...,outdims:]
        outputs_std = outputs_and_gates_std[..., :outdims]
        gates_std = outputs_and_gates_std[..., outdims:]
        gated_mean,gated_std = gated_nonlinearity(outputs,outputs_std,gates,gates_std,self.repout,self.d,self.nounc)
        return coords,gated_mean,gated_std,ids

    def log_data(self,*args,**kwargs):
        pass

    def nll(self):
        if self.nounc:
            return torch.tensor([0.]).cuda()
        else:
            return super().nll()

def sigmoid_mean_and_variance(gate_mean,gate_std,K=50):
    """ assumes input is of shape (a,b,c), computes E[Sigmoid(z)] and Var[Sigmoid(z)]
        for z ~ N(gate_mean,gate_std)"""
    icdf = torch.distributions.normal.Normal(gate_mean[None], gate_std[None]).icdf
    unif = ((torch.arange(K).float().to(gate_mean.device)+ 1 / 2) / K)[:, None,None,None]
    grid_samples = icdf(unif)
    sigmoid_mean = grid_samples.sigmoid().mean(0)
    sigmoid_var = (grid_samples.sigmoid() ** 2).mean(0) - sigmoid_mean**2
    return sigmoid_mean,sigmoid_var

def gated_nonlinearity(mean,std,gate_mean,gate_std,repout,d,nounc=False):
    """ Applies ReLU to scalar channels and gated nonlinearity higher tensors"""
    assert not (torch.isnan(mean).any() | torch.isnan(std).any())
    out_mean = torch.zeros_like(mean)
    out_var = torch.zeros_like(std)
    phis = Phi(mean/std)
    density = torch.exp(-(mean/std)**2/2)/np.sqrt(2*np.pi)
    relu_meanout = mean*phis + std*density
    relu_varout = ((mean ** 2 + std ** 2) * phis + mean*std*density - relu_meanout**2)
    #return relu_meanout, relu_varout.clamp(min=1e-6).sqrt()
    sigmoid_gate_mean, sigmoid_gate_var = sigmoid_mean_and_variance(gate_mean,gate_std)
    i=0
    gateid=0
    for p,q in repout:
        num_elem = d**(p+q)
        if p==q==0:
            if nounc:
                out_mean[...,i:i+num_elem] = mean[...,i:i+num_elem].relu()
            else:
                out_mean[...,i:i+num_elem] += relu_meanout[...,i:i+num_elem]
                out_var[..., i:i+num_elem] += relu_varout[..., i:i+num_elem]
        else:
            if nounc:
                out_mean[..., i:i + num_elem] = gate_mean[...,gateid].sigmoid().unsqueeze(-1)*mean[...,i:i+num_elem]
            else:
                aggregate_varin = (std[...,i:i+num_elem]**2).mean(-1).unsqueeze(-1)
                sigmoid_mean = sigmoid_gate_mean[..., gateid].unsqueeze(-1)
                sigmoid_var = sigmoid_gate_var[...,gateid].unsqueeze(-1)

                out_mean[...,i:i+num_elem] += mean[...,i:i+num_elem]*sigmoid_mean
                out_var[...,i:i+num_elem] += sigmoid_var*aggregate_varin + \
                    sigmoid_var*(mean[...,i:i+num_elem]**2).mean(-1,keepdims=True) + (sigmoid_mean**2)*aggregate_varin
                gateid += 1
        i+=num_elem
    assert not torch.isnan(out_var).any()|torch.isnan(out_mean).any()
    return out_mean, out_var.clamp(min=1e-6).sqrt()


class GLinear(nn.Module):
    """ Rotation Equivariant Linear Layer (mixes only channels)"""
    def __init__(self,repin,repout,nounc=False):
        super().__init__()
        rot_gen = [np.array([[0,-1],[1,0]])]
        d = rot_gen[0].shape[-1]
        active_dims, self.projection = multilinear_active_subspaces(rot_gen, [repout], [Scalar+repin])
        init_std = 1 / np.sqrt(repsize(repout,d)*repsize(repin,d))
        self._weight_params = nn.Parameter(torch.randn(active_dims) * init_std)
        self.nounc=nounc

    @property
    def weight(self):
        return self.projection(self._weight_params)[:,1:] #exclude bias
    @property
    def bias(self):
        return self.projection(self._weight_params)[:,0]

    def forward(self, x):
        coords, meanin, stdin = x[:3]
        weight = self.weight
        meanout = meanin@weight.T + self.bias
        stdout = ((stdin ** 2) @ (weight.T ** 2)).clamp(min=1e-6).sqrt()
        if self.nounc: stdout = torch.zeros_like(meanout)
        out = (coords, meanout, stdout)
        return out + x[3:]

class GPNCNN(nn.Module):
    """ Steerably rotation equivariant PNCNN"""
    def __init__(self, channels_in=1, k=64, num_layers=3, num_targets=10, d=2,nounc=False):
        super().__init__()
        self._unc = nn.Parameter(torch.tensor(-2.))
        feature_rep = 52*Scalar+20*Vector+5*Matrix+2*T(1,2)
        self.network = nn.Sequential(EquivariantLinearOperatorGP(channels_in*Scalar,feature_rep,d=d,nounc=nounc,nonlinearity=True),
                                     GLinear(feature_rep, feature_rep,nounc=nounc),
                                     EquivariantLinearOperatorGP(feature_rep, feature_rep, d=d,nounc=nounc,nonlinearity=True),
                                     GLinear(feature_rep, feature_rep,nounc=nounc),
                                     EquivariantLinearOperatorGP(feature_rep, feature_rep, d=d,nounc=nounc,nonlinearity=True),
                                     GLinear(feature_rep, feature_rep, nounc=nounc),
                                     EquivariantLinearOperatorGP(feature_rep, feature_rep, d=d, nounc=nounc,nonlinearity=True),
                                     GLinear(feature_rep, num_targets*Scalar,nounc=nounc),
                                     AvgPool(),
                                     GetMean())

    def forward(self, x):
        coords, vals = x
        ids = None
        stdin = torch.ones_like(vals) * F.softplus(self._unc)
        return self.network((coords, vals, stdin, ids))
