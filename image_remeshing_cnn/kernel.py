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
from image_remeshing_cnn.kernel_utils import gaussian_blur

def Phi(x):
    """ gaussian cdf"""
    return (1+torch.erf(x/np.sqrt(2)))/2

def phi(x):
    """ gaussian pdf"""
    return torch.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

def phip(x):
    """ derivative of phi"""
    return -x*torch.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


class OpKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def opiK(self, x_n, x_m):
        raise NotImplementedError

    def opiKopj(self, x_n, x_m):
        raise NotImplementedError
@export
class RBF(OpKernel):
    """ Defines an RBF kernel K(x,x') = se^{-(x-x')^2/2l^2}/\sqrt{2\pi\ell^2}
        which is normalized by 1/\sqrt{2\pi\ell^2} and contains the learnable
        parameters s and l (uncertainty scale and lengthscale). """
    def __init__(self, d=2):
        super().__init__()
        self._gp_length_scale = nn.Parameter(torch.tensor(-1.))
        self._gp_scale = nn.Parameter(torch.tensor(-1.))
        self.d = d
        self.ksize=3**d
        assert d <= 2, "only 1d and 2d supported right now"

    @property
    def length_scale(self):
        """ Returns the lengthscale of the RBF kernel"""
        return F.softplus(self._gp_length_scale)

    @property
    def scale(self):
        """ Returns the Uncertainty scale of the RBF kernel"""
        return F.softplus(self._gp_scale)

    def forward(self, x_n, x_m):
        """ Same as evaluating the kernel."""
        return self.K(x_n,x_m)

    def K(self, x_n, x_m,new_lengthscale = None):
        """Evaluates the kernel at locations x=x_n and x'=x_m.
            Inputs: [x_n (bs,n,d)],[x_n (bs,n,d)] Outputs: [K (bs,n,m,d)]"""
        #sqdist = (x_n**2).sum(-1)[...,None] - 2*x_n@x_m.permute(0,2,1) + (x_m**2).sum(-1)[...,None,:]
        sqdist = ((x_n[...,None,:] - x_m[...,None,:,:])**2).sum(-1) # slower but more numerically stable
        ls = new_lengthscale or self.length_scale
        return self.scale*torch.exp(-sqdist/(2 * ls ** 2))/(2*np.pi*ls**2)**(self.d/2)

    def K0(self, x_n,new_lengthscale=None):
        """ Returns the Kernel evaluated at x=x' with (bs,n) when x_n has shape (bs,n,c)
            Inputs: [x_n (bs,n,d)] Outputs: [K(0) (bs,n)]"""
        ls = new_lengthscale or self.length_scale
        return torch.ones_like(x_n[:, :, 0]) * self.scale /(2*np.pi*ls**2)**(self.d/2)

    def opiK(self, x_n, x_m):
        """ Applies the given operator onto the Kernel and evaluates it
            at the locations x=x_n and x'=x_m: opiK(x_n,x_m). By default is
            the derivatives but can be overrided.
            Inputs: [x_n (bs,n,d)], [x_m (bs,m,d)]
            Outputs: [opiK (bs,k,n,m)]"""
        return self.derivs_one_sided(x_n, x_m)

    def opiKopj(self, x_n):
        """ Applies the given operator on both sides of the kernel and
            evaluates the diagonal at x=x' (with the shape of x_n)
            Inputs: [x_n (bs,n,d)] Outputs: [opiKopj (bs,k,k,n)] """
        return self.derivs_double_diag(x_n)

    def opiKopjfull(self, x_m, x_n):  # (bs,i,j,m,n)
        """ Same as opiKopj except that this function evaluates opiKopj
            at all pairs of x=x_m and x'=x_n and not just the diagonal x=x'
            Inputs [x_m (bs,m,d)],[x_n (bs,n,d)]
            Outputs: [opiKopj(x_m,x_n) (bs,k,k,m,n)]"""
        return self.derivs_double_full(x_m,x_n)

    def Dk(self, delta, sigma):
        """ gives vector of derivatives up to kth order (hermite polynomials)
            (bs,n,d) -> (bs,5,n,d) """
        D = torch.stack([torch.ones_like(delta),
                         -delta / sigma ** 2,
                         -1 / sigma ** 2 + delta ** 2 / sigma ** 4,
                         3 * delta / sigma ** 4 - delta ** 3 / sigma ** 6,
                         3 / sigma ** 4 - 6 * delta ** 2 / sigma ** 6 + delta ** 4 / sigma ** 8], dim=1)
        return D

    def derivs_one_sided(self, x_m, x_n,new_lengthscale=None):  # (\partial^i\partial^j k(x,x_n)|_{x=x_m})
        """ returns [1,dx,d^2x] x [1,dy,d^2y] x K with shape (bs,k,m,n)"""
        delta = (x_m[:, :, None, :] - x_n[:, None, :, :])
        assert delta.shape[-1] in (1,2)
        bs, m, n = delta.shape[:-1]
        sigma = new_lengthscale or self.length_scale
        D = self.Dk(delta, sigma)[:, :3]
        if self.d == 1: return (D[..., 0]*self.K(x_m,x_n)[:,None]).reshape(bs, -1, m, n)
        DxDy = D[:, :, None, ..., 0] * D[:, None, :, ..., 1]
        return (DxDy * self.K(x_m, x_n,sigma)[:, None, None, ...]).reshape(bs, -1, m, n)

    def derivs_double_diag(self, x_n,new_lengthscale=None):
        """ returns [1,dx,d^2x] x [1,dy,d^2y] x K(x,x') x [1,dx',d^2x'] x [1,dy',d^2y'] | x_n,x_n
            since dx' = -dx we have
            Outputs: (bs,k,k,n) """
        sigma = new_lengthscale or self.length_scale
        bs, n = x_n.shape[:2]
        D = self.Dk(0 * x_n, sigma)
        Dxxp = torch.stack([D[:, :3], -D[:, 1:4], D[:, 2:5]], dim=1)  # (symmetric matrix)
        Kvals = self.K0(x_n,new_lengthscale=sigma)
        if self.d == 1: return Dxxp[..., 0] * Kvals  # 1d case
        DXXpK = Dxxp[:, :, None, :, None, :, 0] * Dxxp[:, None, :, None, :, :, 1] * Kvals[:, None, None, None,None,...]  # TODO check the ordering ' vs unprimed
        return DXXpK.reshape(bs, self.ksize, self.ksize, n)

    def derivs_double_full(self,x_m,x_n,new_lengthscale=None):
        assert False
        sigma = new_lengthscale or self.length_scale
        bs, n = x_n.shape[:2]
        D = self.Dk(x_m[:,:,None,:] - x_n[:,None,:,:], sigma) #(bs,3,3,m,n,d)
        Dxxp = torch.stack([D[:, :3], -D[:, 1:4], D[:, 2:5]], dim=1)  # (symmetric matrix)
        if self.d == 1: return Dxxp[..., 0] * self.K(x_m,x_n,new_lengthscale=sigma)[:,None,None]  # 1d case
        assert d!=2
        DXXpK = Dxxp[:, :, None, :, None, :, 0] * Dxxp[:, None, :, None, :, :, 1] * Kvals[:, None, None, None, None,...]
        return DXXpK.reshape(bs, self.ksize, self.ksize, n)
    def integral_k(self, x_n):  # \integral_-inf^inf k(x,X_n)dx
        """ (bs,n,d) -> (bs,n)"""
        return 1*self.scale*torch.ones_like(x_n[...,0])
        # axes_k_integrals = (Phi(x_n / self.length_scale) - Phi((x_n - 1) / self.length_scale))
        # total_integral = torch.prod(axes_k_integrals, dim=-1) * self.scale
        # return total_integral

    def integral_integral_k(self):
        """ -> scalar"""
        return self.scale*(2*3)**self.d# actually infinite
        # d = self.d
        # axis_wise_int = (torch.exp(-1 / (2 * self.length_scale ** 2)) - 1)*self.length_scale * np.sqrt(2/np.pi)+ \
        #                 (2 * Phi(1 / self.length_scale) - 1)
        # return (axis_wise_int ** d) * self.scale

    def log_data(self, logger, step, name):
        logger.add_scalars('info', {f'{name}_lengthscale': self.length_scale.cpu().data}, step=step)
        # logger.add_scalars('info', {f'{name}_uncscale':self.scale.cpu().data}, step=step)
        #logger.add_scalars('info', {f'{name}_blurls': self.blur_lengthscale.cpu().data.item()}, step=step)
@export
class RBFwBlur(RBF):
    """ Derivative operator version of RBF kernel where deriv op is followed
        by an isotropic blur by blur_lengthscale """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._blur_lengthscale = nn.Parameter(torch.tensor(-3.))

    @property
    def blur_lengthscale(self):
        return F.softplus(self._blur_lengthscale)

    def opiK(self, x_m, x_n):
        blurred_ls = (self.length_scale**2 + self.blur_lengthscale**2).sqrt()
        return self.derivs_one_sided(x_m, x_n,new_lengthscale=blurred_ls)

    def opiKopj(self, x_n):
        double_blurred_ls = (self.length_scale**2 + 2*self.blur_lengthscale**2).sqrt()
        return self.derivs_double_diag(x_n, new_lengthscale=double_blurred_ls)
    def opiKopjfull(self,x_m,x_n):
        double_blurred_ls = (self.length_scale ** 2 + 2 * self.blur_lengthscale ** 2).sqrt()
        return self.derivs_double_full(x_m,x_n,new_lengthscale=double_blurred_ls)

class IdOpKernel(RBF):
    """ A version of the opkernel where the operator is the identity. For debugging and plotting. """
    def __init__(self,scale,lengthscale,d=1):
        super().__init__(d=d)
        self._gp_length_scale.data = (lengthscale.exp()-1).log()
        self._gp_scale.data = (scale.exp()-1).log()
        self.ksize=1
    def opiK(self,xn,xm):
        return self.K(xn,xm)[:,None,:,:] # (bs,n,m) -> (bs,1,n,m)
    def opiKopj(self, x_n):
        return self.K0(x_n)[:,None,None,:] # (bs,n) -> (bs,1,1,n)
    def opiKopjfull(self, x_m, x_n,left_both=False):  # (bs,i,j,m,n)
        return self.K(x_m,x_n)[:,None,None,:,:] # (bs,m,n) -> (bs,1,1,m,n)

class HeatRBF(RBF):
    """ A version of the RBF op kernel but where the operator is the time evolution
        of the (anisotropic) diffusion equation. e^D_i where D = \beta^T\nabla + (1/2) \nabla^T\Sigma\nabla
        and i runs from 1,2,... to num_basis. """
    def __init__(self, d=2,num_basis=9):
        super().__init__(d=2)
        self._means = nn.Parameter(.1*torch.randn(num_basis,d))
        self._tril_cov = nn.Parameter(.5*torch.randn(num_basis,d,d) if d==1 else .05*torch.randn(num_basis,d,d))
        #self._tril_cov.data -= 1.5*torch.eye(d,device=self._tril_cov.device,dtype=self._tril_cov.dtype)
        self.ksize = num_basis
    @property
    def tril_cov(self):
        #upper = torch.triu(self._tril_cov, diagonal=1)
        # Constrain diagonal of Cholesky to be positive
        return self._tril_cov#(upper + torch.diag_embed(F.softplus(torch.diagonal(self._tril_cov, dim1=-2, dim2=-1)))).permute(0,2,1)
    def beta_sigma(self):
        """ Retrieve the drift and diffusion terms beta and Sigma. """
        L = self.tril_cov  # (k,d,d)
        sigma = L@L.permute(0,2,1)
        beta = self._means
        return beta,sigma
    def opiK(self, x_m, x_n):
        """
        :param x_m: (bs,m,d)
        :param x_n: (bs,n,d)
        :return: AiK (bs, k, m, n)
        """
        #print(self._gp_length_scale.device,self._means.device,self._tril_cov.device)
        beta,S = self.beta_sigma()
        I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).unsqueeze(0)
        Cov =  I*self.length_scale**2 + S
        precision = Cov.inverse() # (k,d,d)
        delta = x_m[:,:,None]-x_n[:,None] #(bs,m,n,d)
        #print(L.device,delta.device)
        # prec_delta = (delta@precision.reshape(-1,L.shape[-1]).T).reshape(*delta.shape[:-1],*L.shape[:2]) # (bs,n,m,k,d)
        # b_prec_b = (self._means*(self._means[:,None,:]*precision).sum(-1)).sum(-1) # (k)
        #                 # (bs,n,m,k)                              #(bs,n,m,k)                        #
        # quadratic = (prec_delta@delta.unsqueeze(-1)).squeeze(-1) +2*(prec_delta*self._means).sum(-1) + b_prec_b
        #opK = (torch.exp(-quadratic / 2) / (2 * np.pi * Cov).det().sqrt()).permute(0,3,1,2) # (bs,n,m,k) -> (bs,k,n,m)
        arg = (delta[:,None] + beta[None,:,None,None,:]) #(bs,k,m,n,d)
        quadratic = ((precision[None,:,None,None]*arg.unsqueeze(-1)).sum(-2)*arg).sum(-1)
        opK = (torch.exp(-quadratic / 2) / (2 * np.pi * Cov).det().sqrt()[None,:,None,None])
        return opK*self.scale

    def opiKopj(self, x_n): # (bs,k,k,n)
        beta,S = self.beta_sigma()
        delta_beta = beta[:, None] - beta[None, :]  # (k,k,d)
        I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
        Cov =  S[:,None] + S[None,:] + I * self.length_scale ** 2
        precision = Cov.inverse() #(k,k,d,d)
        quadratic = ((precision@delta_beta[...,None]).squeeze(-1)*delta_beta).sum(-1) #(k,k)
        opKop = torch.exp(-quadratic/2)/(2*np.pi*Cov).det().sqrt()
        return opKop[None,:,:,None].repeat(x_n.shape[0],1,1,x_n.shape[1])*self.scale

    def opiKopjfull(self, x_m,x_n,left_both=False): # (bs,i,j,m,n)
        """ x_m (bs,m,d) x_n (bs,n,d)
            Computes A_ikA_j(x_m,x_n) or A_iA_jk(x_m,x_n) if left_both=True"""
        # check that the sign on delta means is consistent
        beta,S = self.beta_sigma()
        delta_beta = beta[:,None]-beta[None,:] # (i,j,d)
        if left_both:
            #assert False, "No longer supported"
            delta_beta = beta[:,None]+beta[None,:]
        I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
        Cov = S[:, None] + S[None, :] + I * self.length_scale ** 2 # (i,j,d,d)
        precision = Cov.inverse() # (i,j,d,d)
        b_prec_b = torch.einsum("ijd,ijdc,ijc->ij",delta_beta,precision,delta_beta)
        b_prec_xm = torch.einsum("ijd,ijdc,bmc->bijm",delta_beta,precision,x_m)
        b_prec_xn = torch.einsum("ijd,ijdc,bnc->bijn", delta_beta,precision, x_n)
        xm_prec_xn = torch.einsum("bmd,ijdc,bnc->bijmn",x_m,precision,x_n)
        xn_prec_xn = torch.einsum("bnd,ijdc,bnc->bijn",x_n,precision,x_n)
        xm_prec_xm = torch.einsum("bmd,ijdc,bmc->bijm", x_m, precision, x_m)
        quadX = xm_prec_xm.unsqueeze(-1) -2*xm_prec_xn+xn_prec_xn.unsqueeze(-2) #(bijmn)
        cross_terms = 2*b_prec_xm.unsqueeze(-1) - 2*b_prec_xn.unsqueeze(-2) #(bijmn)
        quadratic = quadX + cross_terms + b_prec_b[None,:,:,None,None]
        opKop = torch.exp(-quadratic / 2) / (2 * np.pi * Cov).det().sqrt()[None,:,:,None,None]
        return opKop*self.scale

@export
class LinearOperatorGP(nn.Module):
    """ Module that forms a GP from input observations and applies the learnable linear
        operator A to the GP and returns the mean (bs,n,c) and elementwise standard deviation (bs,n,c)
        If initialized with nounc, will no propagate uncertainty to output standard deviation."""
    def __init__(self, cin, cout, kernel=RBFwBlur,d=2,nounc=False):
        super().__init__()
        self.d=d
        self.opkernel = kernel(d=d)
        self.op_linear = nn.Linear(cin * self.opkernel.ksize, cout)
        # conv.bias (cout,) conv.weight (cout,chin,ksize)
        self.saved_for_log = None
        self._cached_x = None
        self.nounc=nounc

    @property
    def op_weight(self):
        return self.op_linear.weight

    def meanStd(self, f, sigin, x_in, x_out):
        """ Given the input observations f, with measurement uncertainty sigin
            at the locations x_in, this function computes the elementwise mean
            and standard deviation of Af at the locations x_out.
            Inputs: [f (bs,n,cin), sigma (bs,n,cin), x_in (n,d), x_out (m,d)]
            Outputs: [mu (bs,m,cout), sigma (bs,m,cout)]"""
        #print("L")
        assert not (torch.isnan(sigin).any()|torch.isnan(f).any())
        ksize = self.opkernel.ksize#np.prod(self.opkernel.shape)
        bs, n, cin = f.shape
        bs, m, d = x_out.shape
        opiKmn = self.opkernel.opiK(x_out, x_in)  # (bs,3x3,m,n)
        if self.nounc: sigin = 1e-2*torch.ones_like(sigin)
        # Compute low rank decomposition of [Kxx+sigma^2] via lanczos
        Kxx = gpytorch.lazy.NonLazyTensor(self.opkernel.K(x_in,x_in)) #(bs,n,n) #
        Kxxs = Kxx.add_diag((sigin.mean(-1) ** 2)) #(bs,n,cin) -> (bs,n)
        probe = opiKmn.mean((1, 2))
        with gpytorch.settings.max_root_decomposition_size(20),gpytorch.settings.max_cg_iterations(20):
            KxxsInv = Kxxs.root_inv_decomposition(initial_vectors=probe, test_vectors=probe) # (bs,n,n)
            opikkinv = (KxxsInv@opiKmn.reshape(bs, -1, n).permute(0, 2, 1)).permute(0, 2, 1)  # (bs,ksize*m,n)

        kf = (opikkinv @ f).reshape(bs, ksize, m, cin).permute(0, 2, 3, 1).reshape(bs, m, -1)  # (bs,m,cin*ksize)
        #assert not torch.isnan(kf).any()
        mu = self.op_linear(kf)  # (bs,m,cout)
        cout = mu.shape[-1]
        if self.nounc:
            return (mu,torch.zeros_like(mu))

        # (bs,m,ksize,n) @ (bs,m,n,ksize) -> (bs,m,ksize,ksize)
        opkKopk = opikkinv.reshape(bs,ksize,m,n).permute(0,2,1,3)@opiKmn.reshape(bs, ksize, m, n).permute(0,2,3,1)
        W = self.op_weight.reshape(-1, cin, ksize)
        AAT = (W.permute(0, 2, 1) @ W).reshape(-1, ksize ** 2).T  # (ksize*ksize,cout) #(c^3 + c^2n + n^2 *10 )
        #assert not torch.isnan(AAT).any()
        kxxij = self.opkernel.opiKopj(x_out).reshape(bs, -1, m).permute(0,2,1)
        gp_var = (kxxij - opkKopk.reshape(bs, m, -1)) @ AAT
        sigout = (gp_var + 1e-6).clamp(min=1e-6).sqrt()
        return (mu, sigout)

    def forward(self, x,xout=None):
        coordsin, meanin, stdin, ids = x
        bs,n,d = coordsin.shape
        self._cached_x = coordsin.detach(), meanin.detach(), stdin.detach()
        self.saved_for_log = coordsin.detach(), meanin.detach(), stdin.detach()
        coordsout = coordsin if xout is None else xout
        meanout, stdout = self.meanStd(meanin, stdin, coordsin, coordsout)
        return (coordsout, meanout, stdout,ids)

    def clear_cached(self):
        self._cached_x = None

    def log_data(self, logger, step, name):
        #return # why is pointclose now giving errors??
        if self.saved_for_log is not None:
            # Also compute statistics for logging, the mean and uncertainty of a few channels with a wider sampling
            # Hack the layer to produce only the first 3 channels
            with torch.no_grad():
                xin, f, s = self.saved_for_log
                old_op_linear = self.op_linear
                old_op = self.opkernel
                self.op_linear = nn.Linear(self.op_linear.weight.shape[-1], 3).to(f.device, f.dtype)
                self.op_linear.weight.data[:3, :] = old_op_linear.weight.data[:3, :]
                self.op_linear.bias.data[:3] = old_op_linear.bias.data[:3]
                # self.op_linear.weight.data[3, :] = 0
                # self.op_linear.weight.data[3, 0] = 1
                # self.op_linear.bias.data[3] = 0
                b,n,cin = f.shape
                bs = 6
                h = w = 40
                x_large = torch.stack(torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w)), dim=-1).reshape(
                    -1, 2).unsqueeze(0)
                x_large = x_large.repeat((bs, 1, 1)).to(f.device, f.dtype)
                op_mean, op_std = self.meanStd(f[:bs], s[:bs], xin[:bs], x_large)
                self.op_linear = nn.Linear(cin, cin).to(f.device, f.dtype)
                self.op_weight.data = torch.eye(cin, device=xin.device)
                self.op_linear.bias.data *= 0
                # PosteriorCovarianceLazy(Kxxs,Kmn,Kmm,W,self.unc_scales)
                self.opkernel = IdOpKernel(self.opkernel.scale, self.opkernel.length_scale)
                in_mean, in_std = self.meanStd(f[:bs], s[:bs], xin[:bs], x_large)
                in_std = in_std[..., :3].expand(bs,h*w,3)  # .mean(-1,keepdims=True).repeat((1,1,3)).log()
                in_mean = (in_mean - in_mean.mean((0, 1))) / in_mean.std((0, 1))
                in_std = (in_std - in_std.median(0)[0].median(0)[0])
                in_std = (in_std / (in_std.abs().median(0)[0].median(0)[0])).clamp(min=-2, max=2)
                inp = in_mean[..., :3].expand(bs,h*w,3)
                # inp  = (inp-inp.mean((0,1)))/inp.std((0,1))
                pointclose = \
                (((xin[:bs, None, :, :] - x_large[:, :, None, :]) ** 2).sum(-1).sqrt() < .015).any(dim=-1).float()[
                    ..., None]
                #print(std.shape,pointclose.shape)
                in_std += pointclose * (1 * torch.eye(3)[1].to(inp.device, inp.dtype) - in_std)*.8
                op_mean,op_std = op_mean[...,:3],op_std[...,:3]
                op_mean = (op_mean - op_mean.mean((0, 1))) / op_mean.std((0, 1))
                op_std = (op_std - op_std.median(0)[0].median(0)[0])
                op_std = (op_std / (op_std.abs().median(0)[0].median(0)[0])).clamp(min=-2, max=2)
                imgs = torch.cat([inp,in_std, op_mean[..., :3],op_std[...,:3] ], dim=0)
                imgs = imgs.reshape(4 * bs, h, w, -1).permute(0, 3, 2, 1).detach().cpu().data
                img_grid = vutils.make_grid(imgs, nrow=bs, normalize=True, scale_each=True)
                logger.add_image(f'{name}_meanstd', img_grid, step)
                self.op_linear = old_op_linear
                self.opkernel = old_op
            # # Plot learned coordinates for several images
            # if self.coords_embedding is not None:
            #     fig = plt.figure()
            #     for i in range(5):
            #         plt.scatter(*self.coords_embedding(torch.tensor(i).to(xin.device)).reshape(*xin.shape[1:]).T.cpu().data)
            #         plt.ylim(0,1)
            #         plt.xlim(0,1)
            #     logger.add_figure(f'{name}_coordinates', fig, global_step=step)

    def nll(self,x=None):
        """ takes in train x=(xyz,vals,std) and outputs marginal log likelihood of GP
            (averaged over bs,n,c (batch size, number of training points, number of channels)) """
        xyz,vals,std = x or self._cached_x
        mean_x = gpytorch.means.ZeroMean()(xyz)
        covar_x = gpytorch.lazy.NonLazyTensor(self.opkernel.K(xyz,xyz)).add_diag((std.mean(-1)**2+1e-5))
        marginal_distribution = gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
        marginal_likelihoods = marginal_distribution.log_prob(vals.permute(2,0,1)) # (channels -> batch axis)
        return -marginal_likelihoods.mean()/vals.shape[1] # (average MLLs over channel axis, and divide by N)


class MaskedKxxwNoiseLazy(gpytorch.lazy.LazyTensor):
    """ A gpytorch lazy tensor that performs matrix multiplication with
        a kernel matrix Kxx (bs,n,n) which is identitcal over the channels
        with additional diagonal noise (bs,n,c) and a mask (bs,n,c) that is applied
        over the input. Meant for when different locations are observed per channel,
        and the ones that are not in the mask are just padded."""
    def __init__(self,Kxx,noise,mask,unc_scales):
        super().__init__(Kxx,noise,mask,unc_scales)
        self.Kxx = Kxx # shape (bs,n,n)
        self.noise=noise # shape (bs,n,c)
        self.unc_scales = unc_scales# torch.ones_like(self.noise[0,0]) if unc_scales is None else unc_scales
        assert mask is not None
        assert len(noise.shape)==3
        assert noise.shape[:2]==self.Kxx.shape[:2]
        #self.unc_scales = torch.ones_like(self.noise[0,0])
        self.mask = torch.ones_like(noise) if mask is None else mask
        self.mask = self.mask.reshape(self.noise.shape[0],-1,1) # (bs,n*c,1)

    def _matmul(self,M): # (bs,nc,e)
        """M -> [Kxx + diag(sigma^2)]M (with uncscales and masking for Kxx)"""
        #assert self.mask is None, "masking not yet supported"
        bs,n,c = self.noise.shape
        M0 = torch.where(self.mask>0,M,torch.zeros_like(M)) # set elems not in mask to 0
        noise_mvm = (self.noise.unsqueeze(-1)*M0.reshape(bs,n,c,-1)).reshape(bs,n*c,-1)
        unc_scaled_M0 = (M0.reshape(bs,n,c,-1)*self.unc_scales[:,None]).reshape(bs,n,-1)
        kernel_mvm = (self.Kxx@unc_scaled_M0).reshape(bs,n*c,-1)
        return (kernel_mvm+noise_mvm)*self.mask+1e-5*M # set elems not in mask to 0

    def diag(self):
        bs,n,c = self.noise.shape
        Kxxdiag = torch.diagonal(self.Kxx,dim1=-2,dim2=-1).unsqueeze(-1).repeat((1,1,c)).reshape(bs,-1)
        noisediag = self.noise.reshape(bs,-1)
        return (Kxxdiag+noisediag)*self.mask[...,0] + 1e-5

    def _size(self):
        bs,n,c = self.noise.shape
        return torch.Size((bs,n*c,n*c))
    # def _approx_diag(self):
    #     bs,n,c = self.noise.shape
    #     return (torch.diagonal(self.Kxx,dim1=1,dim2=2).unsqueeze(-1)+self.noise).reshape(bs,n*c)
    def _transpose_nonbatch(self):
        return self
    # def _inv_matmul_preconditioner(self):
    #     return gpytorch.lazy.DiagLazyTensor(1/self._approx_diag())
    def _solve(self, rhs, preconditioner, num_tridiag=0):
        return gpytorch.utils.linear_cg(
            self._matmul,
            rhs,
            n_tridiag=min(num_tridiag,gpytorch.settings.max_cg_iterations.value()),
            max_iter=gpytorch.settings.max_cg_iterations.value(),
            max_tridiag_iter=gpytorch.settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=preconditioner,
        )
    def inv_matmul(self, M):
        return self.mask*super().inv_matmul(self.mask*M)
    # def _solve(self,M,preconditioner,num_tridiag=0):
    #     # with torch.no_grad():
    #     #     preconditioner = self.detach()._inv_matmul_preconditioner()
    #     return gpytorch.utils.linear_cg(
    #         self._matmul,self.mask*M,n_tridiag=0,max_iter=10,
    #         max_tridiag_iter=0,preconditioner=preconditioner)*self.mask

class PosteriorCovarianceLazy(gpytorch.lazy.LazyTensor):
    """ Gpytorch Lazy tensor to compute matrix vector multiplies of the posterior
        covariance by sequentially multiplying through all of the terms.
        Advantages: memory consumption low even when noise is different per channel
            or observations are different per channel.
        Disadvantages: complexity, time cost"""
    def __init__(self,Kxxs,opiKmn,opiKopj,W,unc_scales,probes=20):
        super().__init__(Kxxs,opiKmn,opiKopj,W,unc_scales,probes=probes)
        self.Kxxs = Kxxs # (bs,n*cin,n*cin)
        self.opiKmn = opiKmn # (bs,k,m,n)
        self.opiKopj = opiKopj # (bs,ki,kj,m,m)
        self.W=W # (cout,cin,k)
        self.unc_scales = unc_scales #(cin)
        self.num_probes = probes

    def _size(self):
        c = self.W.shape[0]
        bs = self.opiKmn.shape[0]
        m = self.opiKopj.shape[-1]
        #print(m)
        return torch.Size((bs,m*c,m*c))

    def _transpose_nonbatch(self):
        return self

    def _matmul(self,M):
        """M -> (AKA - AK[Kxx+s^2]^{-1}KA)M"""
        # (bs,m*cout,a)
        bs,k,m,n = self.opiKmn.shape
        co, cin, _k = self.W.shape
        assert _k == k, "mismatched shapes"
        bs_,mc_ = M.shape[:2]
        assert (bs_,mc_)==(bs,m*co), f"{self.shape}, {(bs_,mc_)} and {(bs,m*co)} {self.opiKmn.shape} {self.W.shape}incompatible"
        a = M.shape[-1]
        #mask = self.Kxxs.mask
        #M0 = M * mask
                # (bs,k*n,m) @ (bs,m,co*a) -> (bs,k*n,co*a) -> (bs,k,n,co,a)
        AikM = (self.opiKmn.permute(0,1,3,2).reshape(bs,-1,m)@M.reshape(bs,m,-1)).reshape(bs,k,n,co,a)
         # (bs,n,a,k*cout)@(k*cout,cin) -> (bs,n,a,cin) -> (bs,n,cin,a)
        AM = ((AikM.permute(0,2,4,1,3).reshape(bs,n,a,-1)@self.W.permute(2,0,1).reshape(-1,cin))*self.unc_scales).permute(0,1,3,2)
        with gpytorch.settings.max_cg_iterations(100):
            KinvAM = self.Kxxs.inv_matmul(self.Kxxs.mask*AM.reshape(bs,n*cin,a)).reshape(bs,n,cin*a)
        AjKinvAM = (self.opiKmn.reshape(bs,k*m,n)@KinvAM).reshape(bs,k,m,cin,a)*self.unc_scales[:,None]
        # (bs,m,a,k*cin) @ (k*cin,co) -> (bs,m,a,co) -> (bs,m,co,a)
        AKinvAM=(AjKinvAM.permute(0,2,4,1,3).reshape(bs,m,a,k*cin)@self.W.permute(2,1,0).reshape(k*cin,co)).permute(0,1,3,2)
        WM = (M.reshape(bs,m,co,a).permute(0,1,3,2)@self.W.reshape(co,cin*k)).reshape(bs,m,a,cin,k).permute(0,2,3,4,1)
        # (bs,ki*mi,kj*mj) @ (bs,kj*mj,a*cin) -> (bs,ki*mi,a*cin)
        AikAWM = self.opiKopj.permute(0,1,3,2,4).reshape(bs,k*m,k*m)@WM.reshape(bs,a*cin,k*m).permute(0,2,1)
        # (bs,k,m,a,cin) -> (bs,m,a,k*cin)@(k*cin,cout) -> (bs,m,a,cout)
        AkAM = (AikAWM.reshape(bs,k,m,a,cin)*self.unc_scales).permute(0,2,3,1,4).reshape(bs,m,a,k*cin)@self.W.permute(2,1,0).reshape(k*cin,co)
        AkAM = AkAM.permute(0,1,3,2)
        return (AkAM - AKinvAM).reshape(*M.shape)

    def _approx_diag(self):
        """ stochastic diagonal estimator diag(C) = (1/n) sum z*(Az)"""
        bs,mco,_ = self._size()
        probes = torch.randn(bs,mco,self.num_probes,device=self.W.device,dtype=self.W.dtype)
        numerators = (probes*(self@probes))
        normalizations = (probes**2)
        diag_estimator = numerators.mean(-1)/normalizations.mean(-1)
        #print((diag_estimator.abs()).mean())
        #avg_rel_err = (((numerators.std(-1)/numerators.mean(-1).abs())/np.sqrt(N_probes))).mean()
        #print(f"diag estimate of {diag_estimator[0,8].cpu().data.item():.3f}+={[0,8].cpu().data.item()/np.sqrt(N_probes):.3f}")
        #print(f"avg_diag_rel_err: {avg_rel_err:.2f}")
        return diag_estimator # (bs,mco)

@export
class MaskedLinearOpGP(LinearOperatorGP):
    def __init__(self, cin, *args,probes=20,nounc=False,**kwargs):
        super().__init__(cin,*args,**kwargs)
        self._gp_unc_scales = nn.Parameter(.55*torch.ones(cin))#torch.ones(cin).cuda()#nn.Parameter(.55*torch.ones(cin))
        self.probes=probes
        self.nounc = nounc
    @property
    def unc_scales(self):
        return F.softplus(self._gp_unc_scales)

    def meanStd(self, f, sigin, x_in, x_out,in_mask):
        # Get shapes
        #print((sigin*in_mask).abs().mean()/in_mask.mean())
        assert not torch.isnan(sigin).any()
        if self.nounc: sigin = 1e-4*torch.ones_like(sigin)
        ksize = self.opkernel.ksize  # np.prod(self.opkernel.shape)
        bs, n, cin = f.shape
        bs, m, d = x_out.shape
        opiKmn = self.opkernel.opiK(x_out, x_in)  # (bs,ksize,m,n)
        assert sigin.shape == (bs, n, cin)
        Kxx = self.opkernel.K(x_in, x_in)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
        Kxxs = MaskedKxxwNoiseLazy(Kxx,sigin**2,in_mask,self.unc_scales) # (bs,n,cin)
        with gpytorch.settings.max_cg_iterations(100):
            masked_f = torch.where(in_mask>0,f,torch.zeros_like(f))
            assert not torch.isnan(masked_f).any(), masked_f
            Kinvf = Kxxs.inv_matmul(masked_f.reshape(bs,n*cin,-1)).reshape(bs,n,cin)*in_mask
            interpolated_varin = gpytorch.lazy.NonLazyTensor(Kxx).inv_matmul(sigin**2)*self.unc_scales #cancel out the uncscale inv factor in Kxx
        W = self.op_weight.reshape(-1, cin, ksize)
        cout = W.shape[0]
        out_mask = (in_mask.sum(-1) > 0).float() # (bs,n)
        #print(opiKmn.shape,(bs,ksize,m,cin))
        opiKinvf = (opiKmn.reshape(bs,-1,n)@Kinvf).reshape(bs,ksize,m,cin)*self.unc_scales # (bs,ksize,m,cin)
        mu = self.op_linear(opiKinvf.permute(0,2,3,1).reshape(bs,m,-1))#*out_mask.unsqueeze(-1))
        if self.nounc:
            return (mu,torch.zeros_like(mu),torch.ones_like(mu))
        # (bs,ki,kj,m,m)
        opiKopj_full = self.opkernel.opiKopjfull(x_out,x_out)#*out_mask[:,None,None,:,None]*out_mask[:,None,None,None,:]
        posteriorCov = PosteriorCovarianceLazy(Kxxs,opiKmn,opiKopj_full,W,self.unc_scales,probes=self.probes)
        # op_aleatoric = self.opkernel.opiKopjfull(x_out, x_in,left_both=True).reshape(bs, -1,n) @ interpolated_varin  # (bs,ijm,n)@(bs,n,c)->(bs,ijm,c)
        # aleatoric_var = op_aleatoric.reshape(bs, ksize ** 2, m, cin).permute(0, 2, 3, 1).reshape(bs, m, -1) @ (
        #             W[:, :, None, :] * W[:, :, :, None]).reshape(cout, -1).T
        gp_var = posteriorCov._approx_diag().reshape(*mu.shape)
        sigout = (gp_var+ 1e-6).clamp(min=1e-6).sqrt()
        #W2 = (W**2).mean(-1)#[:,:,0]
        #sigout = ((sigin**2)@W2.T +1e-6).sqrt() # mock up sigout for debugging right now
        mask_out = torch.ones_like(mu)#out_mask.unsqueeze(-1).repeat((1, 1, cout))
        return (mu,sigout,mask_out)

    def forward(self, x):
        coordsin, meanin, stdin,mask, ids = x
        assert mask is not None
        bs,n,d = coordsin.shape
        coordsout = coordsin
        self._cached_x = coordsin.detach(), meanin.detach(), stdin.detach(),mask
        self.saved_for_log = coordsin.detach(), meanin.detach(), stdin.detach(),mask
        meanout, stdout,mask_out = self.meanStd(meanin, stdin, coordsin, coordsout,mask)
        return (coordsout, meanout, stdout,mask_out,ids)

    def nll(self,x=None):
        """ takes in train x=(xyz,vals,std) and outputs marginal log likelihood of GP
            (averaged over bs,n,c (batch size, number of training points, number of channels)) """
        xyz,vals,std,mask = x or self._cached_x
        bs,n,c = vals.shape
        Kxx = self.opkernel.K(xyz, xyz)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
        Kxxs = MaskedKxxwNoiseLazy(Kxx, std ** 2, mask, self.unc_scales)  # (bs,n,cin)
        mean_x = torch.zeros_like(vals).reshape(bs,-1)
        with gpytorch.settings.max_cg_iterations(100):
            marginal_distribution = gpytorch.distributions.MultivariateNormal(mean_x,Kxxs)
            marginal_likelihoods = marginal_distribution.log_prob((mask*vals).reshape(bs,-1))
        return -marginal_likelihoods.mean()/vals.shape[1] # (average MLLs over channel axis, and divide by N)

    def log_data(self, logger, step, name):
        if self.saved_for_log is not None:
            # Also compute statistics for logging, the mean and uncertainty of a few channels with a wider sampling
            # Hack the layer to produce the GP interpolation (no blurring/derivatives)
            with torch.no_grad():
                xin, f, std, mask = self.saved_for_log
                if xin.shape[-1]>1: return
                old_op = self.opkernel
                old_W = self.op_linear
                bs = 6

                bs,n,cin = f.shape
                xout = torch.linspace(-2,2,200).unsqueeze(-1).to(xin.device,xin.dtype)[None].repeat((bs,1,1))
                # Kxx = self.opkernel.K(xin, xin)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
                # Kxxs = MaskedKxxwNoiseLazy(Kxx, std ** 2, mask, self.unc_scales)  # (bs,n,cin)
                # Kmn = self.opkernel.K(xout,xin)[:,None,:,:] # (bs,m,n) -> (bs,1,m,n)
                # Kmm = self.opkernel.K(xout,xout)[:,None,None,:,:] # (bs,m,m) -> (bs,1,1,m,m)
                self.op_linear = nn.Linear(cin, cin).to(f.device, f.dtype)
                self.op_weight.data = torch.eye(cin,device=xin.device)
                self.op_linear.bias.data *=0
                #PosteriorCovarianceLazy(Kxxs,Kmn,Kmm,W,self.unc_scales)
                self.opkernel = IdOpKernel(self.opkernel.scale,self.opkernel.length_scale)
                mask = torch.ones_like(f) if mask is None else mask
                muout,sigout,maskout = self.meanStd(f[:bs],std[:bs],xin[:bs],xout,mask[:bs])

                Tout = xout[0,:,0].cpu().data.numpy()
                Tin = xin[0,:,0].cpu().data.numpy()
                y = muout.cpu().data.numpy()
                yerr = sigout.cpu().data.numpy()
                x = f.cpu().data.numpy()
                xerr = std.cpu().data.numpy()
                xmask = mask.cpu().data.numpy()>0

                if cin>1:
                    fig, axes = plt.subplots(3, 1)
                    channels = [20,22,29]
                    channel_names = ["NIDias","NISys","Sys"]#[self.dataloaders['train'].dataset.vitals[ch] for ch in channels]
                    for i in range(3):
                        c = channels[i]
                        axes[i].errorbar(Tin[xmask[0, :, c]], x[0, :, c][xmask[0, :, c]],
                                         yerr=xerr[0, :, c][xmask[0, :, c]],fmt='r.',zorder=2,markersize=5,alpha=.8)
                        # axes[i].scatter(Tin[xmask[0, :, c]], x[0, :, c][xmask[0, :, c]],
                        #                 c='r',s=50,zorder=4,edgecolors=(0, 0, 0))
                        axes[i].plot(Tout,y[0,:,c],'k',lw=3,zorder=1)
                        axes[i].fill_between(Tout,(y-yerr)[0,:,c],(y+yerr)[0,:,c],alpha=.2,color='k',zorder=3)

                        axes[i].set_title(channel_names[i])
                    logger.add_figure(f'{name}_gppreds', fig, global_step=step)
                else:
                    fig = plt.figure()
                    plt.errorbar(Tin[xmask[0, :, 0]], x[0, :, 0][xmask[0, :, 0]],
                                         yerr=xerr[0, :, 0][xmask[0, :, 0]], fmt='r.', zorder=2, markersize=5, alpha=.8)
                    # axes[i].scatter(Tin[xmask[0, :, c]], x[0, :, c][xmask[0, :, c]],
                    #                 c='r',s=50,zorder=4,edgecolors=(0, 0, 0))
                    plt.plot(Tout, y[0, :, 0], 'k', lw=3, zorder=1)
                    plt.fill_between(Tout, (y - yerr)[0, :, 0], (y + yerr)[0, :, 0], alpha=.2, color='k',
                                         zorder=3)
                    plt.title("NISys")
                    plt.show()
                    logger.add_figure(f'{name}_gppreds', fig, global_step=step,close=False)
                # return kernel as before
                self.opkernel = old_op
                self.op_linear=old_W

@export
class ChannelSeperateLinOpGP(MaskedLinearOpGP):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self._blur_param = nn.Parameter(torch.tensor(5.))
    def forward(self, x):
        coordsin, meanin, stdin, ids = x
        mask = torch.ones_like(meanin)
        coordsout,meanout,stdout,maskout,ids = super().forward((coordsin, meanin, stdin,mask, ids))
        return coordsout,meanout,stdout,ids
    def meanStd(self, *inputs):
        (mu, sigout, mask_out) = super().meanStd(*inputs)
        #sigout = gaussian_blur(sigout,F.softplus(self._blur_param))
        return (mu,sigout,mask_out)

@export
class MultiChannelIntegralGP(nn.Module):
    def __init__(self,cin,kernel=RBF,d=2,nounc=False):
        super().__init__()
        self.opkernel=kernel(d=d)
        self._unc_scales = nn.Parameter(.55 * torch.ones(cin))
        self.nounc = nounc

    @property
    def unc_scales(self):
        return F.softplus(self._unc_scales)

    def forward(self,x):
        """ inputs: [x_in (bs,n,d), f (bs,n,cin), sigma (bs,n,cin)]
            outputs: [mu (bs,cin), sigma (bs,cin)]"""
        coords,meanin,stdin,mask,ids = x
        self._cached_x = coords.detach(), meanin.detach(), stdin.detach(),mask
        bs,n,cin = meanin.shape
        intKx = self.opkernel.integral_k(coords) # (bs,n)
        intKx_full = (intKx.unsqueeze(-1)*self.unc_scales)*mask
        Kxx = self.opkernel.K(coords, coords)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
        Kxxs = MaskedKxxwNoiseLazy(Kxx, stdin ** 2, mask, self.unc_scales)  # (bs,n,cin)
        with gpytorch.settings.max_cg_iterations(100):
            intKxKinv = Kxxs.inv_matmul(intKx_full.reshape(bs,n*cin,1)).reshape(bs, n, cin)*mask
        meanout = (intKxKinv*(mask*meanin)).sum(1) # (bs,cin)
        if self.nounc:
            return (coords,meanout,torch.zeros_like(meanout),mask,ids)
        stdout = (self.opkernel.integral_integral_k()*self.unc_scales-(intKxKinv*intKx_full).sum(1)+1e-7).clamp(min=1e-7).sqrt()
        return (coords,meanout,stdout,mask,ids)

    def nll(self,x=None):
        """ takes in train x=(xyz,vals,std) and outputs marginal log likelihood of GP
            (averaged over bs,n,c (batch size, number of training points, number of channels)) """
        xyz,vals,std,mask = x or self._cached_x
        bs,n,c = vals.shape
        Kxx = self.opkernel.K(xyz, xyz)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
        Kxxs = MaskedKxxwNoiseLazy(Kxx, std ** 2, mask, self.unc_scales)  # (bs,n,cin)
        mean_x = torch.zeros_like(vals).reshape(bs,-1)
        with gpytorch.settings.max_cg_iterations(100):
            marginal_distribution = gpytorch.distributions.MultivariateNormal(mean_x,Kxxs)
            marginal_likelihoods = marginal_distribution.log_prob((mask*vals).reshape(bs,-1))
        return -marginal_likelihoods.mean()/vals.shape[1] # (average MLLs over channel axis, and divide by N)


@export
class IntegralGP(nn.Module):
    """ A layer to be used in PNCNN that integrates the GP over R^d and produces the output mean and std.
        Integrating the RBF kernel over R^d yields a finite mean but infinite variance, so the output
        variance should not be used."""
    def __init__(self,opkernel=RBF,d=2,nounc=False):
        super().__init__()
        self.opkernel=opkernel(d=d)
        self.nounc=nounc
    def forward(self,x):
        """ inputs: [x_in (bs,n,d), f (bs,n,cin), sigma (bs,n,cin)]
            outputs: [mu (bs,cin), sigma (bs,cin)]"""
        coords,meanin,stdin = x[:3]
        self._cached_x = coords.detach(), meanin.detach(), stdin.detach()
        bs,n,cin = meanin.shape
        Kxx = self.opkernel.K(coords,coords)
        Kxxs = gpytorch.lazy.NonLazyTensor(Kxx).add_diag((stdin.mean(-1)**2)+1e-5)
        intKx = self.opkernel.integral_k(coords) # (bs,n)
        with gpytorch.settings.max_root_decomposition_size(20),gpytorch.settings.max_cg_iterations(20):
            probe = intKx
            KxxsInv = Kxxs.root_inv_decomposition(initial_vectors=probe, test_vectors=probe)
            intKxKinv = (KxxsInv@intKx.unsqueeze(-1)).squeeze(-1)
        # with gpytorch.settings.max_cg_iterations(10):
        #     intKxKinv = Kxxs.inv_matmul(intKx.unsqueeze(-1)).squeeze(-1) # (bs,n)
        meanout = (intKxKinv.unsqueeze(-1)*meanin).sum(1) # (bs,cin)
        if self.nounc:
            return (coords,meanout,torch.zeros_like(meanout))+x[3:]
        stdout = (self.opkernel.integral_integral_k()-(intKxKinv*intKx).sum(1)+1e-6).clamp(min=1e-6).sqrt()
        return (coords,meanout,stdout.unsqueeze(-1).repeat((1,cin))) + x[3:]

    def nll(self,x=None):
        """ takes in train x=(xyz,vals,std) and outputs marginal log likelihood of GP
            (averaged over bs,n,c (batch size, number of training points, number of channels)) """
        xyz,vals,std = x or self._cached_x
        mean_x = gpytorch.means.ZeroMean()(xyz)
        covar_x = gpytorch.lazy.NonLazyTensor(self.opkernel.K(xyz,xyz)).add_diag((std.mean(-1)**2+1e-6))
        marginal_distribution = gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
        marginal_likelihoods = marginal_distribution.log_prob(vals.permute(2,0,1)) # (channels -> batch axis)
        return -marginal_likelihoods.mean()/vals.shape[1] # (average MLLs over channel axis, and divide by N)s


@export
class GPinterpolateLayer(nn.Module):
    """ A layer implemented for time series that solely does GP interpolation onto a different sampling grid
        for use when not all input channels are observed simultaneously and one has to use masking (e.g. PhysIONet)."""
    def __init__(self, cin, d=2,nounc=False,res=100):
        super().__init__()
        self._gp_unc_scales = nn.Parameter(.55 * torch.ones(cin))
        self.kernel = RBF(d=d)
        self.nounc=nounc
        self.res=res

    @property
    def unc_scales(self):
        return F.softplus(self._gp_unc_scales)

    def forward(self, x):
        x_in, meanin, stdin, mask, ids = x
        self._cached_x = x_in.detach(), meanin.detach(), stdin, mask.detach()
        assert mask is not None
        bs, n, d = x_in.shape
        bs, n, c = meanin.shape
        assert d==1,"Implemented for time series"
        m = self.res
        x_out = torch.linspace(-2, 2, m).to(meanin.device, meanin.dtype)[None, :, None].repeat((bs, 1, 1))
        Kxy = self.kernel.K(x_in, x_out)  # (bs,n,m)
        Kyy = self.kernel.K0(x_out)[..., None] * self.unc_scales  # (bs,m,c)
        Kxx = self.kernel.K(x_in, x_in)  # (bs,n,n) #      -> (bs,n*c,n*c),
        Kxxs = MaskedKxxwNoiseLazy(Kxx, stdin ** 2, mask, self.unc_scales)  # (bs,n,c)

        maskedKxy = (Kxy[:, :, None, :].repeat((1, 1, c, 1)) * mask[..., None]).reshape(bs, n * c, m)
        with gpytorch.settings.max_cg_iterations(100):
            KinvKxy = (Kxxs.inv_matmul(maskedKxy).reshape(bs, n, c, m) * mask[..., None]) * self.unc_scales[:, None]
        meanout = (meanin[..., None] * KinvKxy).sum(1).permute(0, 2, 1)
        if self.nounc:
            std_out = torch.zeros_like(meanout)
            mask_out = torch.ones_like(meanout)
            self._cached_y = x_out.detach(), meanout.detach(), std_out.detach(), mask_out.detach()
            return (x_out,meanout,torch.zeros_like(meanout),ids)
        KyxKinvKxy = (Kxy[:, :, None, :] * KinvKxy).sum(1).permute(0, 2, 1) * self.unc_scales
        std_out = (Kyy - KyxKinvKxy +1e-7).clamp(min=1e-7).sqrt()
        mask_out = torch.ones_like(meanout)

        self._cached_y = x_out.detach(),meanout.detach(),std_out.detach(),mask_out.detach()
        return (x_out, meanout, std_out, ids)

    def nll(self, x=None):
        """ takes in train x=(xyz,vals,std) and outputs marginal log likelihood of GP
            (averaged over bs,n,c (batch size, number of training points, number of channels)) """
        xyz, vals, std, mask = x or self._cached_x
        bs, n, c = vals.shape
        Kxx = self.kernel.K(xyz, xyz)  # (bs,n,n) #      -> (bs,n*cin,n*cin),
        Kxxs = MaskedKxxwNoiseLazy(Kxx, std ** 2, mask, self.unc_scales)  # (bs,n,cin)
        mean_x = torch.zeros_like(vals).reshape(bs, -1)
        with gpytorch.settings.max_cg_iterations(100):
            marginal_distribution = gpytorch.distributions.MultivariateNormal(mean_x, Kxxs)
            marginal_likelihoods = marginal_distribution.log_prob((mask * vals).reshape(bs, -1))
        return -marginal_likelihoods.mean() / vals.shape[1]  # (average MLLs over channel axis, and divide by N)

    def log_data(self, logger, step, name):
        with torch.no_grad():#
            bs = 6
            xin, f, std, mask = self._cached_x
            xout,muout,sigout,maskout=self._cached_y
            Tout = xout[0, :, 0].cpu().data.numpy()
            Tin = xin[0, :, 0].cpu().data.numpy()
            y = muout.cpu().data.numpy()
            yerr = sigout.cpu().data.numpy()
            x = f.cpu().data.numpy()
            xerr = std.cpu().data.numpy()
            xmask = mask.cpu().data.numpy() > 0
            fig, axes = plt.subplots(3, 1)
            channels = [20, 22, 29]
            channel_names = ["NIDias", "NISys",
                             "Sys"]  # [self.dataloaders['train'].dataset.vitals[ch] for ch in channels]
            for i in range(3):
                c = channels[i]
                axes[i].errorbar(Tin[xmask[0, :, c]], x[0, :, c][xmask[0, :, c]],
                                 yerr=xerr[0, :, c][xmask[0, :, c]], fmt='r.', zorder=2, markersize=5, alpha=.8)
                # axes[i].scatter(Tin[xmask[0, :, c]], x[0, :, c][xmask[0, :, c]],
                #                 c='r',s=50,zorder=4,edgecolors=(0, 0, 0))
                axes[i].plot(Tout, y[0, :, c], 'k', lw=3, zorder=1)
                axes[i].fill_between(Tout, (y - yerr)[0, :, c], (y + yerr)[0, :, c], alpha=.2, color='k',
                                     zorder=3)

                axes[i].set_title(channel_names[i])
            logger.add_figure(f'{name}_gpinterp', fig, global_step=step)
