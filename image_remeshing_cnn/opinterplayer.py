# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpytorch#
import gpytorch.kernels as kernels
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from oil.utils.utils import export
from image_remeshing_cnn.kernel import LinearOperatorGP, Phi, phi,phip,HeatRBF,RBF,IdOpKernel
from functools import partial

global debug
debug=False

def anynan(x):
    return (torch.isnan(x)|torch.isinf(x)).any()

##############################################################################################
# Specialized GP, ReLU, LinOPGP & IntegralGP that passes forward full spatial
# covariance matrices. I believe this is a better approach for calibrated uncertainties;
# however, propagating uncertainties through ReLU proved challenging as the closed form soln
# for the covariance function /matrix involves the multivariate normal CDF which isn't
# implemented in pytorch/tf. Right now uses the incorrect version for the ReLU that lacks
# these extra terms.
##############################################################################################

@export
class GP(nn.Module):
    def __init__(self,d=2):
        super().__init__()
        self.kernel = RBF(d=d)
        #k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.kernel = k1#+k2
    def forward(self,inp,xout=None):
        xin,fin,stdin = inp
        bs,n,d = xin.shape
        #Kxxs = self.kernel(xin,xin).add_diag(stdin.mean(-1)**2)
        uncin = torch.diag_embed(stdin.mean(-1)**2)+1e-5*torch.eye(n,device=xin.device)
        Kxx = self.kernel.K(xin,xin)
        #U = torch.cholesky(Kxx + uncin+1e-6*torch.eye(n,device=xin.device))
        Kxxs = gpytorch.lazy.NonLazyTensor(Kxx+uncin)
        Kxxsinv = Kxxs.inv_matmul#lambda x: torch.solve(x,Kxx+uncin)[0]
        #xgrid = torch.stack(torch.meshgrid([torch.linspace(0,1,9) for _ in range(d)]),dim=-1).reshape(-1,d)
        xgrid = xin#xgrid.to(xin.device)[None].repeat((bs,1,1))
        xout = xgrid if xout is None else xout# same as input for now
        Kyx = self.kernel.K(xout,xin)#.evaluate()
        Kyy = self.kernel.K(xout,xout)#.evaluate()
        # Compute output mean and covariance
        Kxxsinvf = Kxxs.inv_matmul(fin)
        meanout = Kyx@Kxxsinvf
        kout = Kyy - Kyx@Kxxs.inv_matmul(Kyx.transpose(-2,-1))+1e-6*torch.eye(xout.shape[1],device=xin.device)
        # Compute NLL
        #zero_mean = gpytorch.means.ZeroMean()(xin)
        #marginal_distribution = gpytorch.distributions.MultivariateNormal(zero_mean, Kxxs)
        logdet = torch.logdet(Kxx + uncin)#2 * torch.diagonal(U, dim1=-2, dim2=-1).log().sum(-1)#
        nll = ((fin*Kxxsinvf).sum(1)+logdet +n*np.log(2*np.pi))/2
        #nll = -1*marginal_distribution.log_prob(fin.permute(2,0,1)).mean()
        self._cached_nll = nll.mean()
        if debug: assert not anynan(kout)
        return (xout,meanout,kout)

    def nll(self):
        return self._cached_nll#.mean()

@export
class LinearOperatorInterp(nn.Module):
    def __init__(self, cin, cout, kernel=HeatRBF, d=2):
        super().__init__()
        self.d = d
        self.opkernel = kernel(d=d)#                         IdOpKernel(torch.tensor([1.]),torch.tensor([.3]),d=d)#kernel(d=d)
        self.opkernel._gp_scale.data = torch.tensor(.5414)
        self.opkernel._gp_scale.requires_grad=False# = torch.tensor(.5414).cuda()
        self.op_linear = nn.Linear(cin * self.opkernel.ksize, cout)
        self._saved_for_log=None

    def forward(self,inp,xout=None):
        """ (bs,n,d), (bs,n,c), (bs,n,n) """
        xin,meanin,Kin = inp
        self._saved_for_log = xin.detach(),meanin.detach(),Kin.detach()
        if debug: assert not anynan(meanin)
        if debug: assert not anynan(Kin)
        bs,n,c = meanin.shape
        I = torch.eye(n,device=xin.device,dtype=xin.dtype)
        Kxx = self.opkernel.K(xin,xin)
        #U = torch.cholesky(Kxx+1e-5*I)
        Kxxs = gpytorch.lazy.NonLazyTensor(Kxx+1e-5*I)#1e-4*I)
        Kxxinv = Kxxs.inv_matmul#lambda x: torch.solve(x,Kxx+1e-4*I)[0]


        xout = xin if xout is None else xout
        opiKmn = self.opkernel.opiK(xout,xin) #(bs,k,m,n)
        k,m = opiKmn.shape[1:3]
        I = torch.eye(m,device=xin.device)
        # Compute rbf interpolated Amu(xout):
        opimean = (opiKmn.reshape(bs,-1,n)@Kxxinv(meanin)).reshape(bs,k,m,c)
        # (bs,k,m,c) -> (bs,m,c,k) ->(bs,m,ck) -> (bs,m,cout)
        Amean = self.op_linear(opimean.permute(0,2,3,1).reshape(bs,m,-1))
        # Compute rbf interpolated (AKinA')(xout,xout')
        # average variance over the channels (tr(Cov)/cout)
        W = self.op_linear.weight.reshape(-1, c, k)
        W2 = (W[:,:,:,None]*W[:,:,None,:]).sum(1).mean(0) #(k,k) # average over channels tr(cov)/cout
        Kxxinvopi = Kxxinv(opiKmn.reshape(bs,k*m,n).permute(0,2,1)).permute(0,2,1) #(bs,km,n)
        interp_weights = Kxxinv(Kin).permute(0,2,1) #(bs,n,n)
                        # (bs,km,n)@(bs,n,n) ->(bs,km,n)->(bs,k,m,n) -> (bs,m,n,k)
        opi_interp = (Kxxinvopi@interp_weights).reshape(bs,k,m,n).permute(0,2,3,1)

        #opi_interp -= Kxxinvopi.reshape(bs,k,m,n).permute(0,2,3,1)
        # (bs, m, n, k) @ (k, k) -> (bs, m, nk)
        Ak_interp = (opi_interp@W2).reshape(bs,m,n*k)
        AkinA = Ak_interp@opiKmn.permute(0,3,1,2).reshape(bs,n*k,m) + 1e-4*I # (bs,m,nk)@(bs,nk,m)->(bs,m,m)
        #AKK = (Kxxinvopi.reshape(bs,k,m,n).permute(0,2,3,1)@W2).reshape(bs,m,n*k) #(bs,m,nk)
        #AKK@opiKmn.permute(0,3,1,2).reshape(bs,n*k,m)
        # *opiKmn).sum(1) #(bs,m,m
        opiKxxxopj = (Kxxinvopi@opiKmn.reshape(bs, k * m, n).permute(0,2,1)).reshape(bs,k,m,k,m).permute(0,1,3,2,4)
        ijkyy = self.opkernel.opiKopjfull(xout, xout)- opiKxxxopj#(bs,k,k,m,m)
        AkAyy = (ijkyy*W2[None,:,:,None,None]).sum(2).sum(1) + 1e-5*I

        e = torch.symeig(AkAyy)[0]
        assert e.min() > 0, e.min()
        full_AKA = AkinA+AkAyy
        e = torch.symeig(AkinA)[0]
        assert e.min()>0, e.min()
        return (xout,Amean,full_AKA)


    def log_data(self,logger,step,name):
        if self._saved_for_log is not None:
            pass

@export
class IntegralInterp(nn.Module):
    def __init__(self, kernel=RBF, d=2):
        super().__init__()
        self.d = d
        self.kernel = kernel(d=d)

    def forward(self,inp):
        """ (bs,n,d), (bs,n,c), (bs,n,n) """
        xin,meanin,Kin = inp
        bs,n,c = meanin.shape
        I = torch.eye(n,device=xin.device,dtype=xin.dtype)
        Kxx = self.kernel.K(xin,xin)
        Kxxs = gpytorch.lazy.NonLazyTensor(Kxx+1e-6*I)
        #U = torch.cholesky(Kxx+1e-5*I)
        Kxxinv = Kxxs.inv_matmul#lambda x: torch.solve(x,Kxx+1e-4*I)[0]
        intKx = self.kernel.integral_k(xin) #(bs,n)
        integral_mean = (intKx[...,None]*Kxxinv(meanin)).sum(1) #(bs,c)
        return integral_mean

def sqrt2x2(M):
    """ computes matrix square root of M (*,2,2)"""
    det = M[...,0,0]*M[...,1,1]-M[...,1,0]*M[...,0,1]
    tr = M[...,0,0]+M[...,1,1]
    s = det.clamp(min=1e-8).sqrt()
    t = (tr+2*s).sqrt()
    sI = s[...,None,None]*torch.eye(2,device=M.device,dtype=M.dtype)
    return (M + sI)/t[...,None,None].clamp(min=1e-8)

def inverse2x2(M):
    """ computes matrix square root of M (*,2,2)"""
    det = M[...,0,0]*M[...,1,1]-M[...,1,0]*M[...,0,1]
    invM = torch.zeros_like(M)
    invM[...,0,0] += M[...,1,1]
    invM[...,1,1] += M[...,0,0]
    invM[...,0,1] -= M[...,0,1]
    invM[...,1,0] -= M[...,1,0]
    invM /= det[...,None,None].clamp(min=1e-8)
    return invM



@export
class ReLUcov(nn.Module):
    """ Propagates spatial uncertainty through the ReLU assuming Gaussian input"""
    def forward(self,inp):
        """ (bs,n,d) (bs,n,c), (bs,n,n)"""

        xin,meanin,kin = inp
        e = torch.symeig(kin)[0]
        assert not (e < 1e-6).any(), (e.min(), kin.shape)
        if debug: assert not anynan(meanin)
        if debug: assert not anynan(kin)
        bs,n,c = meanin.shape
        kin += torch.eye(n,device=xin.device)*1e-6
        kdiag = torch.diagonal(kin,dim1=-2,dim2=-1) # (bs,n)
        stdin = kdiag.unsqueeze(-1).clamp(min=1e-8).sqrt() #(bs,n,1)
        # compute meanout:
        Phis = Phi(meanin/stdin)
        phis = phi(meanin/stdin)
        Eh = meanin * Phis + stdin * phis #(bs,n,c)
        # compute Kout
        K1 = torch.stack([kdiag[...,None].repeat((1,1,n)),kin],dim=-1) #(bs,n,n,2)
        K2 = torch.stack([kin.transpose(-1,-2),kdiag[...,None,:].repeat((1,n,1))],dim=-1)
        K12 = torch.stack([K1,K2],dim=-2) # (bs,n,n,2,2)
        #K12 += 10*torch.eye(n,device=xin.device)[:,:,None,None]*torch.eye(2,device=xin.device)[None,None]
        if debug: assert not anynan(K12)
        sqrtK12 = sqrt2x2(K12)
        #if debug: assert (sqrtK12@sqrtK12-K12).abs().mean()<1e-5
        isqrtK12 = inverse2x2(sqrtK12) #(bs,n,n,2,2)

        if debug: assert not anynan(sqrtK12)
        if debug: assert not anynan(isqrtK12)
        u1 = isqrtK12[...,0,0,None]*meanin[:,:,None,:]+isqrtK12[...,0,1,None]*meanin[:,None,:,:] #(bs,n,n,c)
        u2 = isqrtK12[...,1,0,None]*meanin[:,:,None,:]+isqrtK12[...,1,1,None]*meanin[:,None,:,:]
        Phi1 = Phi(u1)
        Phi2 = Phi(u2)
        Psi = Phi1*Phi2
        psi1 = phi(u1)*Phi2
        psi2 = Phi1*phi(u2)
        EhhT_Psi = (meanin[:,:,None,:]*meanin[:,None,:,:]+kin[...,None])*Psi
        # mu2S1 + mu1S2
        EhhT_psi1 = (meanin[:,None,:,:]*sqrtK12[...,0,0,None]+meanin[:,:,None,:]*sqrtK12[...,1,0,None])*psi1
        EhhT_psi2 = (meanin[:,None,:,:]*sqrtK12[...,0,1,None]+meanin[:,:,None,:]*sqrtK12[...,1,1,None])*psi2
        SHPhiS11 = sqrtK12[...,0,0,None]*sqrtK12[...,0,1,None]*phip(u1)*Phi(u2)
        SHPhiS12 = sqrtK12[..., 0, 1,None] * sqrtK12[..., 0, 1,None] * phi(u1) * phi(u2)
        SHPhiS22 = sqrtK12[..., 0, 1,None] * sqrtK12[..., 1, 1,None] * Phi(u1) * phip(u2)
        SHPhiS = SHPhiS11+2*SHPhiS12+SHPhiS22
        EhhT = EhhT_Psi + (EhhT_psi1+EhhT_psi2) + SHPhiS
        covar = (EhhT - Eh[:,:,None,:]*Eh[:,None,:,:]).mean(-1) #(bs,n,n)
        diag_covar = torch.diag_embed(((meanin**2 + stdin**2)*Phis+meanin*stdin*phis - Eh**2).mean(-1).clamp(min=1e-6)) #(bs,n,n)
        full_covar = torch.where(diag_covar>1e-7,diag_covar,covar)
        e = torch.symeig(full_covar)[0]
        assert not (e<1e-6).any(), (e.min(),full_covar.shape)
        return xin,Eh,full_covar+torch.eye(n,device=xin.device)*1e-6


@export
class PNCNNinterp(nn.Module):
    """ A version of the PNCNN that uses each of these components"""
    def __init__(self, channels_in=1, k=64, num_layers=3, num_targets=10, d=2,
                 kernel=HeatRBF, nounc=False, num_basis=None):
        super().__init__()
        self._unc = nn.Parameter(torch.tensor(-1.))
        if num_basis: kernel = partial(kernel, num_basis=num_basis)
        self.network = nn.Sequential(GP(d=d),
                                     LinearOperatorInterp(channels_in, k, kernel=kernel, d=d),
                                     ReLUcov(),
                                     LinearOperatorInterp(k, k, kernel=kernel, d=d),
                                     ReLUcov(),
                                     # LinearOperatorInterp(k, k, kernel=kernel, d=d),
                                     # ReLUcov(),
                                     LinearOperatorInterp(k, num_targets, kernel=kernel, d=d),
                                     IntegralInterp(d=d))

    def forward(self, x):
        coords, vals = x
        stdin = torch.ones_like(vals) * F.softplus(self._unc).to(vals.device)
        return self.network((coords, vals, stdin))





