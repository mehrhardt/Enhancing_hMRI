# Auxilary functions needed for
# M. J. Ehrhardt, F. A. Gallagher, M. A. McLean, and C.-B. Schoenlieb, 
# Enhancing the Spatial Resolution of Hyperpolarized Carbon-13 MRI of Human 
# Brain Metabolism using Structure Guidance, 2021.

# models
import numpy as np
from matplotlib import pyplot as plt
import odl
from skimage.measure import block_reduce as sk_block_reduce

# select colormaps for plotting, case may be 'guide' for proton MRI, 0 for 
# Lactate and 3 for Pyruvate. This is consistent through the scripts as this is
# the order in which the data is stored: e.g.    
# 0 = Lactate
# 1 = Hydrate
# 2 = Alanine
# 3 = Pyruvate
# 4 = Bicarbonate
def cmap(case):
    if case == 'guide':
        smap = 'gray'
    elif case == 0:
        # smap = 'inferno'
        smap = 'Purples'
    elif case == 1:
        # smap = 'inferno'
        smap = 'Blues'
    elif case == 2:
        # smap = 'inferno'
        smap = 'Greens'
    elif case == 3:
        # smap = 'viridis'
        smap = 'Oranges'
    elif case == 4:
        smap = 'Reds'
    else:
        smap = None
        
    return smap
        
# auxiliary function to store images        
def save_image(x, filename, guide=False, molecule=None):
    if len(x.shape) == 2:
        if guide:
            c = cmap('guide')
        else:        
            c = cmap(molecule)  
        plt.imsave(filename + '.png', x, cmap=c, vmin=0, vmax=1) 
    elif len(x.shape) == 3:
        for i in range(x.shape[2]):
            save_image(x[:,:,i], '{}_{:02d}'.format(filename, i), guide, molecule)    


# Data and images are scaled for processing so that they are roughly between 0
# and 1. This is done via predefined constants for each data set.
# The scaling_list is has 5 values corresponding to
# 0 = Lactate
# 1 = Hydrate
# 2 = Alanine
# 3 = Pyruvate
# 4 = Bicarbonate
# def scaling(dataset, molecule):
#     if dataset == 'HV-109':
#         scaling_list = [1/700, np.nan, np.nan, 1/3500, 1/200]
    
#     elif dataset == 'HV-114':
#         scaling_list = [1/4000, np.nan, np.nan, 1/21000, 1/1000]

#     elif dataset == 'HV-117':
#         scaling_list = [1/2000, np.nan, np.nan, 1/11000, 1/1000]
        
#     elif dataset == 'HV-118':
#         scaling_list = [1/8000, np.nan, np.nan, 1/38000, 1/1000]
    
#     elif dataset == 'InSilico':
#         scaling_list = [1/4200, np.nan, np.nan, 1/11000, np.nan]
        
#     elif dataset == 'Phantom':
#         scaling_list = [1/700, np.nan, np.nan, 1/2000, np.nan]
        
#     return scaling_list[molecule]

def scaling(dataset, molecule, folder, data=None):
    filename = '{}/scaling_{}_{}.npy'.format(folder, dataset, molecule)
    
    import numpy as np
    import os.path
    if os.path.exists(filename):
        scaling = np.load(filename)
            
    else:
        if data is not None:
            scaling = 1 / np.quantile(data, 0.99)
            np.save(filename, scaling)
        else:
            print('could neither load nor compute scaling')
            
    return scaling
        
# Strings for all molecules
def s_molecule(molecule):
    list_molecule = ['Lactate', 'Hydrate', 'Alanine', 'Pyruvate', 'Bicarbonate']
    return list_molecule[molecule]

# Deform a 2D image with an affine transformation with 6 parameters
def deform_param(x, param):
    U = odl.uniform_discr([-1, -1], [1, 1], x.shape)
    
    shift = param[0:2]
    matrix = param[2:6]
    disp_vf = [
            lambda x: matrix[0] * x[0] + matrix[1] * x[1] + shift[0],
            lambda x: matrix[2] * x[0] + matrix[3] * x[1] + shift[1]]
    
    displacement = U.tangent_bundle.element(disp_vf)
 
    return odl.deform.linear_deform(U.element(x), displacement)

# Overlay 2 images
def overlay(x0, x1, molecule):
    x0 = np.squeeze(x0)
    x1 = np.squeeze(x1)
    
    from skimage.util import compare_images

    trans_gray = plt.get_cmap(cmap('guide'))
    trans_color = plt.get_cmap(cmap(molecule))
    x = np.zeros((x0.shape[0], x0.shape[1], 3))
    
    for i in range(3):
        x[:,:,i] = compare_images(trans_color(x0)[:,:,i], trans_gray(x1)[:,:,i], method='checkerboard', n_tiles=(3,5))

    return x

# Quickly visualize results
def show_result(x, molecule, A, data, guide):
    dim = len(guide.shape)
    
    if dim == 3:
        plt.clf()
        
        k = int((x.shape[2]-1)/2)
        
        plt.subplot(232)
        plt.imshow(x[:,:,k], cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('result')
        
        plt.subplot(233)
        plt.imshow(guide[:,:,k], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('guide')
        
        plt.subplot(234)
        plt.imshow(A(x), cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('estimated data')
        
        plt.subplot(235)
        plt.imshow(data, cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('data')
    
        plt.subplot(236)
        e = A(x) - data
        s = .2
        plt.imshow(e, vmin=-s, vmax=s, cmap='RdBu')
        plt.colorbar()
        plt.title('est - data')
        
    elif dim == 2:
        plt.figure(1)
        plt.clf()
        
        plt.subplot(232)
        plt.imshow(x, cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('result')
        
        plt.subplot(233)
        plt.imshow(guide, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('guide')
        
        plt.subplot(234)
        plt.imshow(A(x), cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('estimated data')
        
        plt.subplot(235)
        plt.imshow(data, cmap=cmap(molecule), vmin=0, vmax=1)
        plt.colorbar()
        plt.title('data')
    
        plt.subplot(236)
        e = A(x) - data
        s = .2
        plt.imshow(e, vmin=-s, vmax=s, cmap='RdBu')
        plt.colorbar()
        plt.title('est - data')
        
# Callback function to compute and store information during recon
class MyCallback(odl.solvers.Callback):

    def __init__(self, A, molecule, data, guide, seg, filename, suptitle_param, 
                 iter_save=[], iter_plot=[], prefix=None, obj_fun=None, 
                 gtruth=None, showimages=False):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter_count = 0
        self.molecule = molecule
        self.A = A
        self.guide = guide
        self.data = data
        self.seg = seg
        self.suptitle_param = suptitle_param
        self.filename = filename
        self.showimages = showimages
        
        if prefix is None:
            prefix = ''
        else:
            prefix += '_'
            
        self.prefix = prefix
        self.obj = []
        self.gtruth = gtruth
        self.obj_fun = obj_fun

    def __call__(self, x, **kwargs):

        if len(x) == 2:
            x = x[0]

        k = self.iter_count

        if k in self.iter_save:
            if self.obj_fun is not None:
                self.obj.append(self.obj_fun(x))

        if k in self.iter_plot:
            name = '{}{:04d}'.format(self.prefix, k)
            
            show_result(x, self.molecule, self.A, self.data, self.guide)

            if self.gtruth is not None:
                from odl.contrib.fom import psnr
                PSNR = psnr(x, self.gtruth)
            else:
                PSNR = np.nan
            
            seg0 = x.space.element(self.seg[...,0])
            seg1 = x.space.element(self.seg[...,1])
            
            meanGM = x.inner(seg0) / x.space.one().inner(seg0)
            meanWM = x.inner(seg1) / x.space.one().inner(seg1)
            plt.gcf().suptitle('{} {} {}\nPSNR: {:3.2f}, meanGM: {:3.2f}, meanWM: {:3.2f}'.format(
                name, self.suptitle_param, k, PSNR, meanGM, meanWM))
            plt.savefig('{}_overview_{:04d}.png'.format(self.filename, k))
            
            if self.showimages:
                save_image(x, '{}_{:04d}'.format(self.filename, k), molecule=self.molecule)

        self.iter_count += 1
        
# reconstruction helper for dtv
def recon_dtv(guide, data, seg, folder_out, alpha, eta, gamma, niter, molecule, 
              gtruth=None, scaling=1):
    
    filename = '{}/dtv_s{:3.2e}_e{:3.2e}_a{:3.2e}_g{:5.4f}'.format(
        folder_out, scaling, eta, alpha, gamma)
    suptitle_param = 'e{:3.2e}, a{:3.2e}'.format(eta, alpha)

    print(filename)
    
    dim = len(guide.shape)
    
    if dim == 3:
        U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], 
                              [1, 1, guide.shape[2]/150], guide.shape)
        V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
        S = Subsampling(U, V, margin=((0,0),(0,0), (1,1)))
        
    else:
        U = odl.uniform_discr([-1, -1], [1, 1], guide.shape)
        V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
        S = Subsampling(U, V, margin=((0,0),(0,0)))
    
    data = V.element(data)
    guide = U.element(guide)
    
    if gtruth is not None:
        gtruth = U.element(gtruth)

    grad = gradient(U, guide=guide, gamma=gamma, eta=eta)
    G, F, A = tv(S, data, alpha, grad=grad)

    norm_As = []
    for Ai in A:
        xs = odl.phantom.white_noise(Ai.domain, seed=1807)
        norm_As.append(Ai.norm(estimate=True, xstart=xs))
        
    Atilde = odl.BroadcastOperator(
            *[Ai / norm_Ai for Ai, norm_Ai in zip(A, norm_As)])
    Ftilde = odl.solvers.SeparableSum(
            *[Fi * norm_Ai for Fi, norm_Ai in zip(F, norm_As)])
    
    obj_fun = Ftilde * Atilde + G
    
    Atilde_norm = Atilde.norm(estimate=True)

    x = S.adjoint(data)
    sigma = scaling / Atilde_norm
    tau = 0.999 / (scaling * Atilde_norm)

    step = 200
    iter_save = []
    iter_plot = [0, 100, 250, 500, 1000, niter]
    cb = (odl.solvers.CallbackPrintIteration(step=step, end=', ') &
          odl.solvers.CallbackPrintTiming(step=step, cumulative=False, 
                                          end=', ') &
          odl.solvers.CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                          cumulative=True, end=', ') &
          odl.solvers.CallbackPrint(step=step, func=obj_fun, 
                                    fmt='obj={:3.2e}') &
          MyCallback(S, molecule, data, guide, seg, filename, suptitle_param, 
                     iter_save=iter_save, iter_plot=iter_plot, obj_fun=obj_fun, 
                     gtruth=gtruth, showimages=False))

    cb(x)
    odl.solvers.pdhg(x, G, Ftilde, Atilde, niter, tau, sigma, callback=cb)
    
    with open('{}.npz'.format(filename), 'wb') as file_out:
        np.savez(file_out, x=x)
        
# reconstruction helper for Dwork2021      
def recon_Dwork2021(guide, data, seg, folder_out, alpha, niter, molecule, 
                    gtruth=None, scaling=1):
    
    filename = '{}/Dwork2021_s{:3.2e}_a{:3.2e}'.format(folder_out, scaling, 
                                                       alpha)

    M = data.max()
    
    dim = len(guide.shape)
    if dim == 3:
        # recon space
        U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], 
                              [1, 1, guide.shape[2]/150], guide.shape)
        # data space
        V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
        # forward operator
        S = Subsampling(U, V, margin=((0,0),(0,0),(1,1)))
        
        raise RuntimeError('not defined for this dimension')
            
    else:
        # recon space
        U = odl.uniform_discr([-1, -1], [1, 1], guide.shape)
        # data space
        V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
        # forward operator
        S = Subsampling(U, V, margin=((0,0),(0,0)))
            
    data = V.element(data)
    guide = U.element(guide)
        
    data /= M
    guide /= guide.ufuncs.max()
    
    if gtruth is not None:
        gtruth = U.element(gtruth) / M
            
    recons = []
    for sign in [1, -1]:
        if sign == -1:
            guide = 1 - guide
            
        filename_ = '{}/Dwork2021_s{:3.2e}_a{:3.2e}_sign{}'.format(
            folder_out, scaling, alpha, sign)
        suptitle_param = 'a{:3.2e}'.format(alpha)
    
        print(filename)
        
        G, F, A, w = Dwork2021(S, data, alpha, guide)
    
        norm_As = []
        for Ai in A:
            xs = odl.phantom.white_noise(Ai.domain, seed=1807)
            norm_As.append(Ai.norm(estimate=True, xstart=xs))
            
        Atilde = odl.BroadcastOperator(
                *[Ai / norm_Ai for Ai, norm_Ai in zip(A, norm_As)])
        Ftilde = odl.solvers.SeparableSum(
                *[Fi * norm_Ai for Fi, norm_Ai in zip(F, norm_As)])
        
        obj_fun = Ftilde * Atilde + G
        
        Atilde_norm = Atilde.norm(estimate=True)
    
        x = S.adjoint(data)
        sigma = scaling / Atilde_norm
        tau = 0.999 / (scaling * Atilde_norm)
    
        step = 200
        iter_save = []
        iter_plot = [0, 100, 250, 500, 1000, niter]
        cb = (odl.solvers.CallbackPrintIteration(step=step, end=', ') &
              odl.solvers.CallbackPrintTiming(step=step, cumulative=False, end=', ') &
              odl.solvers.CallbackPrintTiming(step=step, fmt='total={:.3f} s',
                                              cumulative=True, end=', ') &
              odl.solvers.CallbackPrint(step=step, func=obj_fun, fmt='obj={:3.2e}') &
              MyCallback(S, molecule, data, guide, seg, filename_, 
                         suptitle_param, iter_save=iter_save, 
                         iter_plot=iter_plot, obj_fun=obj_fun, 
                         gtruth=gtruth, showimages=False))
    
        cb(x)
        odl.solvers.pdhg(x, G, Ftilde, Atilde, niter, tau, sigma, callback=cb)
        
        x *= M
        
        with open('{}.npz'.format(filename_), 'wb') as file_out:
            np.savez(file_out, x=x)

        recons.append(x)
        
    if (recons[0]-w).norm() < (recons[1]-w).norm():
        x = recons[0]
    else:
        x = recons[1]
    
    with open('{}.npz'.format(filename), 'wb') as file_out:
        np.savez(file_out, x=x)
        

def gradient(space, guide=None, gamma=1, eta=1e-2, prefix=None):
    
    grad = odl.Gradient(space, method='forward', pad_mode='symmetric')
    
    if guide is not None:
        norm = odl.PointwiseNorm(grad.range)
        grad_guide = grad(guide)
        ngrad_guide = norm(grad_guide)
          
        for i in range(len(grad_guide)):
            grad_guide[i] /= ngrad_guide.ufuncs.max()

        ngrad_guide = norm(grad_guide)            
        ngrad_guide_eta = np.sqrt(ngrad_guide ** 2 + eta ** 2)

        xi = grad.range.element([g / ngrad_guide_eta for g in grad_guide])
        
        Id = odl.operator.IdentityOperator(grad.range)
        xiT = odl.PointwiseInner(grad.range, xi)
        xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])
        
        grad = (Id - gamma * xixiT) * grad

    return grad


def tv(operator, data, alpha, grad=None, nonneg=True):

    space = operator.domain
    
    if grad is None:
        grad = gradient(space)
    
    A = odl.BroadcastOperator(operator, grad)
    
    F1 = odl.solvers.L2NormSquared(data.space).translated(data)
    F2 = alpha * odl.solvers.GroupL1Norm(grad.range)
    F = odl.solvers.SeparableSum(F1, F2)
    
    if nonneg:
        G = odl.solvers.IndicatorNonnegativity(space)
    else:
        G = odl.solvers.ZeroFunctional(space)        
            
    return G, F, A


def Dwork2021(operator, data, alpha, guide):
    space = operator.domain
    
    import skimage.transform as sktransform
    w = sktransform.resize(data, guide.shape, order=1)
    
    grad = gradient(space)
    M = odl.DiagonalOperator(odl.MultiplyOperator(space.element(w)), 2)
    Mgrad = M * grad
    A = odl.BroadcastOperator(operator, Mgrad)
    
    F1 = odl.solvers.L2NormSquared(data.space).translated(data)
    F2 = alpha * odl.solvers.L2NormSquared(grad.range).translated(Mgrad(guide))
    F = odl.solvers.SeparableSum(F1, F2)
    
    G = odl.solvers.IndicatorBox(space, lower=0, upper=1)     
            
    return G, F, A, w


class Subsampling(odl.Operator):
    '''  '''
    def __init__(self, domain, range, margin=None):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn((8, 8, 8))
        >>> Y = odl.rn((2, 2))  
        >>> S = myOperators.Subsampling(X, Y)
        >>> myOperators.test_adjoint(S)
        """
        domain_shape = np.array(domain.shape)
        range_shape = np.array(range.shape)
        
        len_domain = len(domain_shape)
        len_range = len(range_shape)
        
        if margin is None:
            margin = 0
                
        if np.isscalar(margin):
            margin = [(margin, margin)] * len_domain

        self.margin = np.array(margin).astype('int')

        self.margin_index = []
        for m in self.margin:
            m0 = m[0]
            m1 = m[1]
            
            if m0 == 0:
                m0 = None
            
            if m1 == 0:
                m1 = None
            else:
                m1 = -m1

            self.margin_index.append((m0, m1))
                        
        if len_domain < len_range:
            ValueError('TBC')
        else:
            if len_domain > len_range:
                range_shape = np.append(range_shape, np.ones(len_domain - len_range))
                
            self.block_size = tuple(((domain_shape - np.sum(self.margin, 1)) / range_shape).astype('int'))

        super(Subsampling, self).__init__(domain=domain, range=range, 
                                          linear=True)
        
    def _call(self, x, out):
        m = self.margin_index        
        if m is not None:
            if len(m) == 1:
                x = x[m[0][0]:m[0][1]]
            elif len(m) == 2:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1]]
            elif len(m) == 3:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]]
            else:
                ValueError('TBC') 
    
        out[:] = np.squeeze(sk_block_reduce(x, block_size=self.block_size, 
                                            func=np.mean))
                # block_reduce: returns Down-sampled image with same number of dimensions as input image.
                            
    @property
    def adjoint(self):
        op = self
            
        class SubsamplingAdjoint(odl.Operator):
            
            def __init__(self, op):
                """TBC
        
                Parameters
                ----------
                TBC
        
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> myOperators.test_adjoint(S)
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 15))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> myOperators.test_adjoint(S)
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 15))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y)
                >>> myOperators.test_adjoint(S)
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y, margin=1)
                >>> myOperators.test_adjoint(S)
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 21))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y, margin=((0, 0),(0, 0),(3, 3)))
                >>> myOperators.test_adjoint(S)
                """
                domain = op.range
                range = op.domain
                self.block_size = op.block_size
                self.margin = op.margin
                self.margin_index = op.margin_index
                
                x = range.zero()
                m = self.margin_index
                if m is not None:
                    if len(m) == 1:
                        x[m[0][0]:m[0][1]] = 1
                    elif len(m) == 2:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1]] = 1
                    elif len(m) == 3:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]] = 1
                    else:
                        ValueError('TBC')
                else:
                    x = range.one()
                
                self.factor = x.inner(range.one()) / domain.one().inner(domain.one())
                
                super(SubsamplingAdjoint, self).__init__(
                        domain=domain, range=range, linear=True)
                    
            def _call(self, x, out):
                for i in range(len(x.shape), len(self.block_size)):     
                    x = np.expand_dims(x, axis=i)
                                    
                if self.margin is None:
                    out[:] = np.kron(x, np.ones(self.block_size)) / self.factor                     
                else:      
                    y = np.kron(x, np.ones(self.block_size)) / self.factor                     
                    out[:] = np.pad(y, self.margin, mode='constant')

            @property
            def adjoint(self):
                return op
                    
        return SubsamplingAdjoint(self)
    
    @property
    def inverse(self):
        scaling = 1 / self.norm(estimate=True) ** 2
        return self.adjoint * scaling
    
# auxilary function to test the adjointness of an operator
def test_adjoint(A):
    import odl
    x = odl.phantom.white_noise(A.domain)
    y = odl.phantom.white_noise(A.range)
    print(A(x).inner(y)/x.inner(A.adjoint(y)))  
    
