# create figures of
# M. J. Ehrhardt, F. A. Gallagher, M. A. McLean, and C.-B. Schoenlieb, 
# Enhancing the Spatial Resolution of Hyperpolarized Carbon-13 MRI of Human 
# Brain Metabolism using Structure Guidance, MRM, 2021.

# load modules
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktransform
import misc
import os
import odl
from matplotlib import gridspec

formats = ['tiff', 'png']

# set and create output folders
folder_out = '../pics/processed'

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
     
def text(s, ax):    
    ax.text(0.5, -1.25, s, horizontalalignment='center', 
            verticalalignment='center', fontsize=12)    

def vtext(s, ax):    
    ax.text(0.5, 0.5, s, rotation=90, horizontalalignment='center', 
            verticalalignment='center', fontsize=12)    
    
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def colorbar(im, ax, label=None):
    fig = plt.gcf()
    axins = inset_axes(ax, # here using axis of the lowest plot
               width="100%",  # width = 5% of parent_bbox width
               height="100%",  # height : 340% good for a (4x4) Grid
               loc='lower left',
               bbox_to_anchor=(0.1, 0.1, 0.8, 0.8),
               bbox_transform=ax.transAxes,
               borderpad=0)

    fig.colorbar(im, cax=axins, orientation='horizontal', label=label)
    
    
def vcolorbar(im, ax, label=None):
    fig = plt.gcf()
    axins = inset_axes(ax, # here using axis of the lowest plot
               width="50%",  # width = 5% of parent_bbox width
               height="100%",  # height : 340% good for a (4x4) Grid
               loc='lower left',
               bbox_to_anchor=(0.05, 0.1, 0.8, 0.8),
               bbox_transform=ax.transAxes,
               borderpad=0)

    fig.colorbar(im, cax=axins, label=label)
    

def plot_pseudo3D(gs, irow, icol, ncols, data, vmin, vmax, aspect, cmap, 
                  title=None, cbar=False, label=None, vcbar=False, tcolor='r', 
                  cbarlabel=None):
    
    iy = data.shape[1] // 2
    ix = data.shape[1] // 2
        
    ax = plt.subplot(gs[icol+irow*ncols])
    ax.imshow(data, vmin=vmin, vmax=vmax, aspect=1, cmap=cmap)
    plt.axis('off')

    if label is not None:
        ax.text(0.5, 1.1, label, horizontalalignment='center', 
                verticalalignment='center', fontsize=12, 
                transform=ax.transAxes)    
        
    ax = plt.subplot(gs[icol+(irow+1)*ncols])
    ax.imshow(np.reshape(data[:, iy], (1, -1)), vmin=vmin, vmax=vmax, 
              aspect=aspect, cmap=cmap)
    plt.axis('off')

    ax = plt.subplot(gs[icol+(irow+2)*ncols])
    im = ax.imshow(np.reshape(data[ix, :], (1, -1)), vmin=vmin, vmax=vmax, 
                   aspect=aspect, cmap=cmap)
    plt.axis('off')

    if title is not None:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')
        text(title, ax)
    
    if cbar:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')   
        colorbar(im, ax, cbarlabel)
        
    if vcbar:
        ax = plt.subplot(gs[icol+1+irow*ncols])
        plt.axis('off')   
        vcolorbar(im, ax, cbarlabel)          
    
def plot_error(gs, irow, icol, ncols, data, vmin, vmax, title=None, cbar=False, 
               label=None, tcolor='r', bcolor=None, cbarlabel=None):
    ax = plt.subplot(gs[icol+irow*ncols])
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='bwr')
    plt.axis('off')

    if label is not None:
        ax.text(0.5, 1.1, label, horizontalalignment='center', 
                verticalalignment='center', fontsize=12, 
                transform=ax.transAxes)    
    
    if title is not None:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')
        text(title, ax)
    
    if cbar:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')
        colorbar(im, ax, cbarlabel)   
        
def plot_3D(gs, irow, icol, ncols, data, vmin, vmax, aspect, cmap, title=None, 
            cbar=False, vcbar=False, label=None, onlabel=None, tcolor='r', 
            bcolor=0, cbarlabel=None):
    
    iz = data.shape[2] // 2
    iy = data.shape[1] // 2
    ix = data.shape[1] // 2
    
    ax = plt.subplot(gs[icol+irow*ncols])
    im = ax.imshow(data[:, :, iz], vmin=vmin, vmax=vmax, aspect=1, cmap=cmap)
    plt.axis('off')

    if label is not None:
        ax.text(0.5, 1.1, label, horizontalalignment='center', 
                verticalalignment='center', fontsize=12, 
                transform=ax.transAxes)    

    if onlabel is not None:
        t = ax.text(0.05, 0.95, onlabel, color=tcolor, 
                    horizontalalignment='left', verticalalignment='top', 
                    fontsize=12, transform=ax.transAxes) 
        if bcolor is not None:
            import matplotlib
            c = matplotlib.cm.get_cmap(cmap)
            t.set_bbox(dict(facecolor=c(bcolor), alpha=0.8, 
                            edgecolor=c(bcolor), boxstyle="Round, pad=0.1"))
        
    ax = plt.subplot(gs[icol+(irow+1)*ncols])
    im = ax.imshow(np.flipud(data[:, iy, 1:-1].T), vmin=vmin, vmax=vmax, 
                   aspect=aspect, cmap=cmap)
    plt.axis('off')

    ax = plt.subplot(gs[icol+(irow+2)*ncols])
    im = ax.imshow(np.flipud(data[ix, :, 1:-1].T), vmin=vmin, vmax=vmax, 
                   aspect=aspect, cmap=cmap)
    plt.axis('off')

    if title is not None:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')
        text(title, ax)

    if cbar:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')  
        colorbar(im, ax, cbarlabel)    
        
    if vcbar:
        ax = plt.subplot(gs[icol+1+irow*ncols])
        plt.axis('off')   
        vcolorbar(im, ax, cbarlabel)    
        

def plot_3D_color(gs, irow, icol, ncols, data,  aspect, title=None, cbar=False, 
                  vcbar=False, label=None, cbarlabel=None, alpha=1): #, tcolor='r'):
    
    iz = data.shape[2] // 2
    iy = data.shape[1] // 2
    ix = data.shape[1] // 2
        
    ax = plt.subplot(gs[icol+irow*ncols])
    im = ax.imshow(data[:, :, iz, :], aspect=1, alpha=alpha)
    plt.axis('off')

    if label is not None:
        ax.text(0.5, 1.1, label, horizontalalignment='center', 
                verticalalignment='center', fontsize=12, 
                transform=ax.transAxes)

    if vcbar:
        ax = plt.subplot(gs[icol+1+irow*ncols])
        plt.axis('off')   
        vcolorbar(im, ax)   
        
    ax = plt.subplot(gs[icol+(irow+1)*ncols])
    x = np.moveaxis(data[:, iy, 1:-1, :], [1], [0])
    x = np.flip(x, 0)
    im = ax.imshow(x, aspect=aspect, alpha=alpha)
    plt.axis('off')

    ax = plt.subplot(gs[icol+(irow+2)*ncols])
    x = np.moveaxis(data[ix, :, 1:-1, :], [1], [0])
    x = np.flip(x, 0)
    im = ax.imshow(x, aspect=aspect, alpha=alpha)
    plt.axis('off')

    if title is not None:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')
        text(title, ax)

    if cbar:
        ax = plt.subplot(gs[icol+(irow+3)*ncols])
        plt.axis('off')   
        colorbar(im, ax, cbarlabel)    
    
    
#%% Figure 1: idea
name = '3DHV-109'
param = 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'
folder_data = '../processed_data'

file_data = '{}/{}_Lactate_40.npz'.format(folder_data, name)
dic = np.load(file_data)
guide = dic['guide']
mol = dic['molecule']
scaling = misc.scaling(name[2:], mol, folder=folder_data) * 1000
vmax = 1 / scaling
data = dic['data'] / scaling

aspecthigh = 2/1.5
aspectlow = 30/6

folder_result1 = '../pics/figure7/{}_Lactate_40'.format(name)
result = np.load(folder_result1 + "/" + param + ".npz")['x'] / scaling

plt.close(1)
plt.figure(1, figsize=(5, 2.15))
ncols = 5
nrows = 3
figheight = 8 # 240 / 30
gs = gridspec.GridSpec(nrows, ncols, height_ratios=[figheight, 1, 1], 
                       width_ratios=[1, 0.2, 1, 0.2, 1], wspace=0.02, 
                       hspace=0.05, left=0, right=1, top=0.87, bottom=0) 

plot_pseudo3D(gs, 0, 0, ncols, data, 0, vmax, aspectlow, misc.cmap(mol), 
              label='${}^{13}$C-MRI', tcolor='w')
plot_3D(gs, 0, 4, ncols, result, 0, vmax, aspecthigh, misc.cmap(mol), 
        label='proposed output', tcolor='w')
plot_3D(gs, 0, 2, ncols, guide, 0, 1, aspecthigh, misc.cmap('guide'), 
        label='${}^1$H-MRI', tcolor='w')

ax = plt.subplot(gs[1])
plt.axis('off')
ax.text(0.5, 0.5, '+', color='k', horizontalalignment='center', 
        verticalalignment='center', fontsize=20, transform=ax.transAxes) 

ax = plt.subplot(gs[3])
plt.axis('off')
ax.text(0.5, 0.5, '=', color='k', horizontalalignment='center', 
        verticalalignment='center', fontsize=20, transform=ax.transAxes) 
    
plt.show()
for f in formats:
    plt.savefig('{}/figure1.{}'.format(folder_out, f))

#%% Figure 2: 2d v 3d data
import pydicom
import matplotlib.patches as patches

name = 'HV-109'
folder_data = '../data/{}'.format(name)

slice = 84
fov_sinfo = 256
fov_data = 240
    
slices = range(slice-8, slice+9)
slices_all = range(1, 125)

image = np.zeros((160,160,len(slices_all)))
for j, s in enumerate(slices_all):
    file_sinfo = '{}/002_Stealth_3D_Bravo_/{:04d}.dcm'.format(folder_data, s)
    sinfo_dcm = pydicom.read_file(file_sinfo)    
    sinfo_raw = sinfo_dcm.pixel_array.astype('float64')
    image[:,:,j] = sktransform.resize(sinfo_raw, (160, 160))
 
mri = image / image.max() * 3.5
   
plt.close(1)
plt.figure(1, figsize=(4.8, 3.3))
plt.clf()
gs = gridspec.GridSpec(2, 1, wspace=0.05, hspace=0.02, left=0.05, right=1, 
                       top=0.9, bottom=0) 

aspect = 2/1.5

iy = image.shape[1] // 2

fov_full = np.array([240, 240, 248])
fov = np.array([240, 240, 30])

simage = mri.shape
vsize = fov_full / np.array(mri.shape)

x0, x1 = 0, simage[0] - 1
y0, y1 = 0, simage[1]
z0, z1 = 61, 100

zmris = np.sort([(z1-z0) - (s - z0) for s in slices])
zcmri = [zmris[0]-1, zmris[-1]+1]

ax = plt.subplot(gs[0])
im = ax.imshow(np.rot90(mri[:, iy, z0:z1], 1), vmin=0, vmax=1, aspect=aspect, 
               cmap='gray')
plt.axis('off')

rect = patches.Rectangle((x0, zcmri[0]), x1-x0, zcmri[1]-zcmri[0], linewidth=1, 
                         edgecolor='r', facecolor='none', label='13C-MRI')
ax.add_patch(rect)
rect = patches.Rectangle((x0, zcmri[0]), x1-x0, zcmri[1]-zcmri[0], 
                         edgecolor='none', facecolor='r', alpha=0.3)
ax.add_patch(rect)

c = len(zmris) // 2
ax.plot([x0, x1], [zmris[c], zmris[c]], linewidth=1, color='b', label='1H-MRI')

plt.legend(ncol=2, bbox_to_anchor=(1, 1.25), loc='upper right', frameon=False)

ax.text(-0.025, 0.5, '2D', rotation=90, horizontalalignment='center', 
        verticalalignment='center', fontsize=12, transform=ax.transAxes)    

ax = plt.subplot(gs[1])
im = ax.imshow(np.rot90(mri[:, iy, z0:z1], 1), vmin=0, vmax=1, aspect=aspect, 
               cmap='gray')
plt.axis('off')

rect = patches.Rectangle((x0, zcmri[0]), x1-x0, zcmri[1]-zcmri[0], linewidth=1, 
                         edgecolor='r', facecolor='none')
ax.add_patch(rect)
rect = patches.Rectangle((x0, zcmri[0]), x1-x0, zcmri[1]-zcmri[0], 
                         edgecolor='none', facecolor='r', alpha=0.3)
ax.add_patch(rect)

for z in zmris:
    ax.plot([x0, x1], [z, z], linewidth=1, color='b')

ax.text(-0.025, 0.5, '3D', rotation=90, horizontalalignment='center',
        verticalalignment='center', fontsize=12, transform=ax.transAxes)    
   
name_ = 'figure2'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))


#%% Figure 3: guide
name = '3DHV-118'
folder_data = '../processed_data'

file_data = '{}/{}_Pyruvate_40.npz'.format(folder_data, name)
dic = np.load(file_data, allow_pickle=True)
guide = dic['guide']
seg = dic['seg']
mol = dic['molecule']

U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], 
                      [1, 1, guide.shape[2]/150], guide.shape)

plt.close(1)
ncols = 4
nrows = 3
figheight = 8 # 240 / 30
plt.figure(1, figsize=(5, 2.33))
gs = gridspec.GridSpec(nrows, ncols, width_ratios=[1, .02, 1, 1],
                       height_ratios=[figheight, 1, 1], wspace=0.05, 
                       hspace=0.02, left=0, right=1, top=0.90, bottom=0) 

aspecthigh = 2/1.5

def vecfield(guide):
    grad = odl.Gradient(guide.space, method='forward', pad_mode='symmetric')
    eta=1e-1
    norm = odl.PointwiseNorm(grad.range)
    grad_guide = grad(guide)
    ngrad_guide = norm(grad_guide)
      
    for i in range(len(grad_guide)):
        grad_guide[i] /= ngrad_guide.ufuncs.max()
    
    ngrad_guide = norm(grad_guide)            
    ngrad_guide_eta = np.sqrt(ngrad_guide ** 2 + eta ** 2)
    
    xi = grad.range.element([g / ngrad_guide_eta for g in grad_guide])
    
    return xi

ic = 0
plot_3D(gs, 0, ic, ncols, guide, 0, 1, aspecthigh, misc.cmap('guide'), 
        label='${}^1$H-MRI')

ic += 2
x = guide[:,:,8]
x2 = np.repeat(x[:, :, np.newaxis], 17, axis=2)
xi = vecfield(U.element(x2))
x = np.moveaxis(xi.asarray(), [0], [3])
x = np.abs(x)
plot_3D_color(gs, 0, ic, ncols, x, aspecthigh, 
              label='vector field $\\xi$ (2D)')

ic += 1
xi = vecfield(U.element(guide))
x = np.moveaxis(xi.asarray(), [0], [3])
x = np.abs(x)
plot_3D_color(gs, 0, ic, ncols, x, aspecthigh, 
              label='vector field $\\xi$ (3D)')

plt.show()
name_ = 'figure3'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))

    
#%% Figure 4: data
file_data1 = '3DInSilico_Pyruvate_40.npz'
file_data2 = '3DPhantom_Lactate_40.npz'
file_data3 = '3DHV-118_Lactate_40.npz'
folder_data = '../processed_data'

ff1 = np.load('{}/{}'.format(folder_data, file_data1), allow_pickle=True)
ff2 = np.load('{}/{}'.format(folder_data, file_data2), allow_pickle=True)
ff3 = np.load('{}/{}'.format(folder_data, file_data3), allow_pickle=True)

guide1 = ff1['guide']
mol1 = ff1['molecule']
scaling1 = misc.scaling('InSilico', mol1, folder=folder_data)
vmax1 = 1 / scaling1
data1 = ff1['data'] / scaling1
gtruth1 = ff1['gt'] / scaling1

guide2 = ff2['guide']
seg2 = ff2['seg']
mol2 = ff2['molecule']
scaling2 = misc.scaling('Phantom', mol2, folder=folder_data)
vmax2 = 1 / scaling2
data2 = ff2['data'] / scaling2

guide3 = ff3['guide']
mol3 = ff3['molecule']
scaling3 = misc.scaling('Phantom', mol3, folder=folder_data)
vmax3 = 1 / scaling3
data3 = ff3['data'] / scaling3

aspecthigh = 2/1.5
aspectlow = 30/6    

plt.close(1)
ncols = 4
nrows = 10
figheight = 8 # 240 / 30
plt.figure(1, figsize=(5, 6.2))
gs = gridspec.GridSpec(nrows, ncols,  
                       width_ratios=[0.12, 1, 1, 1],
                       height_ratios=[1.3, figheight, 1, 1, figheight, 1, 1, 
                                      figheight, 1, 1], 
                       wspace=0.02, hspace=0.02, left=0, right=1, top=1, 
                       bottom=0) 
    
ax = plt.subplot(gs[1])
plt.axis('off')
ax.text(0.5, 0.5, '${}^{13}$C-MRI', horizontalalignment='center', 
        verticalalignment='center', fontsize=12)    
ax = plt.subplot(gs[2])
plt.axis('off')
ax.text(0.5, 0.5, '${}^1$H-MRI', horizontalalignment='center', 
        verticalalignment='center', fontsize=12)    
ax = plt.subplot(gs[3])
plt.axis('off')  
ax.text(0.5, 0.5, 'registration', horizontalalignment='center', 
        verticalalignment='center', fontsize=12)

ic = 0
ax = plt.subplot(gs[ic+1*ncols])
plt.axis('off')
vtext('$\it{in~silico}$', ax)
ax = plt.subplot(gs[ic+4*ncols])
plt.axis('off')
vtext('$\it{in~vitro}$', ax)
ax = plt.subplot(gs[ic+7*ncols])
plt.axis('off')
vtext('$\it{in~vivo}$ example', ax)

ic += 1
plot_pseudo3D(gs, 1, ic, ncols, data1, 0, vmax1, aspectlow, misc.cmap(mol1))
plot_pseudo3D(gs, 4, ic, ncols, data2, 0, vmax2, aspectlow, misc.cmap(mol2))
plot_pseudo3D(gs, 7, ic, ncols, data3, 0, vmax3, aspectlow, misc.cmap(mol3))

ic += 1
plot_3D(gs, 1, ic, ncols, guide1, 0, 1, aspecthigh, misc.cmap('guide'))
plot_3D(gs, 4, ic, ncols, guide2, 0, 1, 2/1.5 * (15/4), misc.cmap('guide'))
plot_3D(gs, 7, ic, ncols, guide3, 0, 1, aspecthigh, misc.cmap('guide'))


def plot_color(data1, guide1, scaling1, aspecthigh, ncols, ir):
    U = odl.uniform_discr([-1, -1, -guide1.shape[2]/150], 
                          [1, 1, guide1.shape[2]/150], guide1.shape)
    V = odl.uniform_discr([-1, -1], [1, 1], data1.shape)
    S3d = misc.Subsampling(U, V, margin=((0,0),(0,0), (1,1)))
    x = np.zeros((160, 160, guide1.shape[2], 3))
    d = guide1
    x[:,:,:,0] = d
    d = S3d.adjoint(data1)
    s = S3d.norm(estimate=True)
    x[:,:,:,2] = d * scaling1 / s**2
    plot_3D_color(gs, ir, 3, ncols, x, aspecthigh)
    
ic += 1

plot_color(data1, guide1, 1.5*scaling1, aspecthigh, ncols, 1)
plot_color(data2, guide2, 2*scaling2, 2/1.5 * (15/4), ncols, 4)
plot_color(data3, guide3, scaling3, aspecthigh, ncols, 7)

plt.show()
name_ = 'figure4'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))


#%% Figure 5: Comparison on in silico phantom with Dwork2021
name = '3DInSilico'
folder_data = '../processed_data'

file_data = '{}/{}_Pyruvate_40.npz'.format(folder_data, name)
dic = np.load(file_data, allow_pickle=True)
guide = dic['guide']
seg = dic['seg']
mol = dic['molecule']
scaling = misc.scaling(name[2:], mol, folder=folder_data) * 1000
gtruth = dic['gt'] / scaling
vmax = 1 / scaling
data = dic['data'] / scaling

aspecthigh = 2/1.5
aspect2d = aspecthigh * 15
    
name2d = '2' + name[1:]
base = '../pics/figure5/{}_Pyruvate_40/'.format(name)
base2d = '../pics/figure5/{}_Pyruvate_40/'.format(name2d)
result2d_1 = np.load(base2d + 'dtv_s1.00e+02_e1.00e-02_a1.00e-02_g0.9995.npz')['x'] / scaling
result3d_1 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a1.00e-02_g0.9995.npz')['x'] / scaling
resultDworkOpt = np.load(base2d + 'Dwork2021_s1.00e+02_a1.00e-04.npz')['x'] / scaling
resultDworkLarge = np.load(base2d + 'Dwork2021_s1.00e+02_a1.00e-02_sign-1.npz')['x'] / scaling

# 3d recon space
U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], 
                      [1, 1, guide.shape[2]/150], guide.shape)
# data space
V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
# 3d sampling operator
S3d = misc.Subsampling(U, V, margin=((0,0),(0,0), (1,1)))

# 2d guide (central slice)
guide2 = guide.copy()
guide2 = guide2[:,:,8]
# 2d recon space
U2d = odl.uniform_discr([-1, -1], [1, 1], guide2.shape)
# 3d sampling operator
S2d = misc.Subsampling(U2d, V, margin=((0,0),(0,0)))

plt.close(1)
ncols = 7
nrows = 6
figheight = 8 # 240 / 30
plt.figure(1, figsize=(8.2, 4))
gs = gridspec.GridSpec(nrows, ncols, width_ratios=[0.12, 1, 1, 1, 1, 1, .3],
                       height_ratios=[figheight, 1, 1, figheight, 1, 1], 
                       wspace=0.05, hspace=0.02, left=0, right=.96, top=0.94, 
                       bottom=0) 

if gtruth is None:
    vmine = -0.2 / scaling
    vmaxe = -vmine
else:
    vmine = -1 / scaling
    vmaxe = -vmine
    c = gtruth.size * np.max(gtruth)
    error = lambda x: np.linalg.norm(x.ravel()-gtruth.ravel())** 2 / c

ic = 1

x = gtruth
plot_3D(gs, 0, ic, ncols, x, 0, vmax, aspecthigh, misc.cmap(mol), 
        label='ground truth')
plot_3D(gs, 3, ic, ncols, x - gtruth, vmine, vmaxe, aspecthigh, cmap='RdBu', 
        onlabel='MSE: {:2.2f}'.format(error(gtruth)), tcolor='k', bcolor=None)

ax = plt.subplot(gs[0])
plt.axis('off')
vtext('pyruvate', ax)

ax = plt.subplot(gs[0+3*ncols])
plt.axis('off')
vtext('error', ax)

ic += 1
x = resultDworkOpt
plot_pseudo3D(gs, 0, ic, ncols, x, 0, vmax, aspect2d, misc.cmap(mol), 
              label='[22] (optimal $\\alpha$)')
r2d_1 = np.repeat(x[:, :, np.newaxis], 17, axis=2)
plot_3D(gs, 3, ic, ncols, r2d_1 - gtruth, vmine, vmaxe, aspecthigh, cmap='bwr', 
        onlabel='MSE: {:2.2f}'.format(error(r2d_1)), tcolor='k', bcolor=None)
        
ic += 1
x = resultDworkLarge
plot_pseudo3D(gs, 0, ic, ncols, x, 0, vmax, aspect2d, misc.cmap(mol), 
              label='[22] (very large $\\alpha$)')
r2d_1 = np.repeat(x[:, :, np.newaxis], 17, axis=2)
plot_3D(gs, 3, ic, ncols, r2d_1 - gtruth, vmine, vmaxe, aspecthigh, cmap='bwr', 
        onlabel='MSE: {:2.2f}'.format(error(r2d_1)), vcbar=True, tcolor='k', 
        bcolor=None)

ic += 1
x = result2d_1
plot_pseudo3D(gs, 0, ic, ncols, x, 0, vmax, aspect2d, misc.cmap(mol), 
              label='2D-dTV')
r2d_1 = np.repeat(x[:, :, np.newaxis], 17, axis=2)
plot_3D(gs, 3, ic, ncols, r2d_1 - gtruth, vmine, vmaxe, aspecthigh, cmap='bwr', 
        onlabel='MSE: {:2.2f}'.format(error(r2d_1)), tcolor='k', bcolor=None)
        
ic += 1
x = result3d_1
plot_3D(gs, 0, ic, ncols, x, 0, vmax, aspecthigh, misc.cmap(mol), 
        label='3D-dTV (proposed)', vcbar=True, 
        cbarlabel='signal intensity [a.u.]')
plot_3D(gs, 3, ic, ncols, x - gtruth, vmine, vmaxe, aspecthigh, cmap='bwr', 
        onlabel='MSE: {:2.2f}'.format(error(x)), vcbar=True, tcolor='k', 
        bcolor=None, cbarlabel='signal intensity [a.u.]')

plt.show()
name_ = 'figure5'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))

    
#%% Figure 6: with Dwork 2021
name, molecule = '3DHV-114', 'Lactate'
folder_data = '../processed_data'

namelong = '{}_{}_40'.format(name, molecule)
file_data = '{}/{}.npz'.format(folder_data, namelong)
dic = np.load(file_data, allow_pickle=True)
guide = dic['guide']
mol = dic['molecule']
scaling = misc.scaling(name[2:], mol, folder=folder_data) * 1000
vmax = 1 / scaling
data = dic['data'] / scaling
        
aspecthigh = 2/1.5
aspect2d = aspecthigh * 15

name2d = '2' + namelong[1:]
base = '../pics/figure6/{}/'.format(name2d)
result2d_1 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-04_g0.9995.npz')['x'] / scaling
result2d_2 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-03_g0.9995.npz')['x'] / scaling
result2d_3 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995.npz')['x'] / scaling
result2d_4 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-01_g0.9995.npz')['x'] / scaling

base = '../pics/figure6/{}/'.format(namelong)
result3d_1 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-03_g0.9995.npz')['x'] / scaling
result3d_2 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995.npz')['x'] / scaling
result3d_3 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995.npz')['x'] / scaling
result3d_4 = np.load(base + 'dtv_s1.00e+02_e1.00e-02_a5.00e-01_g0.9995.npz')['x'] / scaling

base = '../pics/figure6/{}/'.format(name2d)
resultDwork_1 = np.load(base + 'Dwork2021_s1.00e+02_a5.00e-04.npz')['x'] / scaling
resultDwork_2 = np.load(base + 'Dwork2021_s1.00e+02_a5.00e-03.npz')['x'] / scaling
resultDwork_3 = np.load(base + 'Dwork2021_s1.00e+02_a5.00e-02.npz')['x'] / scaling
resultDwork_4 = np.load(base + 'Dwork2021_s1.00e+02_a5.00e-01.npz')['x'] / scaling

# 3d recon space
U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], 
                      [1, 1, guide.shape[2]/150], guide.shape)
# data space
V = odl.uniform_discr([-1, -1], [1, 1], data.shape)
# 3d subsampling operator
S3d = misc.Subsampling(U, V, margin=((0,0),(0,0),(1,1)))

# 2d guide
guide2 = guide.copy()
guide2 = guide2[:,:,8]
# 2d recon space
U2d = odl.uniform_discr([-1, -1], [1, 1], guide2.shape)
# 2d subsamping operator
S2d = misc.Subsampling(U2d, V, margin=((0,0),(0,0)))

plt.close(1)
ncols = 9
nrows = 13
figheight = 8 # 240 / 30
plt.figure(1, figsize=(10, 8.6))
gs = gridspec.GridSpec(nrows, ncols,  
                       width_ratios=[0.18, 1, 1, 0.02, 1, 1, 0.02, 1, 1],
                       height_ratios=[figheight, 1, 1, figheight, 1, 1, 
                                      figheight, 1, 1, figheight, 1, 1, 1.5], 
                       wspace=0.05, hspace=0.03, left=0, right=1, top=0.965, 
                       bottom=0.06) 

vmine = -0.25 / scaling
vmaxe = -vmine
    
ic = 0

ax = plt.subplot(gs[ic+0*ncols])
plt.axis('off')
vtext('$\\alpha=5 \cdot 10^{-4}$', ax)
ax = plt.subplot(gs[ic+3*ncols])
plt.axis('off')
vtext('$\\alpha=5 \cdot 10^{-3}$', ax)
ax = plt.subplot(gs[ic+6*ncols])
plt.axis('off')
vtext('$\\alpha=5 \cdot 10^{-2}$', ax)
ax = plt.subplot(gs[ic+9*ncols])
plt.axis('off')
vtext('$\\alpha=5 \cdot 10^{-1}$', ax)

cmap = misc.cmap(mol)
ic += 1
plot_pseudo3D(gs, 0, ic, ncols, resultDwork_1, 0, vmax, aspect2d, cmap, 
              label='[22]', tcolor='w')
plot_pseudo3D(gs, 3, ic, ncols, resultDwork_2, 0, vmax, aspect2d, cmap)
plot_pseudo3D(gs, 6, ic, ncols, resultDwork_3, 0, vmax, aspect2d, cmap)
plot_pseudo3D(gs, 9, ic, ncols, resultDwork_4, 0, vmax, aspect2d, cmap)

ic += 1
plot_error(gs, 0, ic, ncols, S2d(resultDwork_1) - data, vmine, vmaxe, 
           label='residual', tcolor='k')
plot_error(gs, 3, ic, ncols, S2d(resultDwork_2) - data, vmine, vmaxe)
plot_error(gs, 6, ic, ncols, S2d(resultDwork_3) - data, vmine, vmaxe)
plot_error(gs, 9, ic, ncols, S2d(resultDwork_4) - data, vmine, vmaxe)   

ic += 2
plot_pseudo3D(gs, 0, ic, ncols, result2d_1, 0, vmax, aspect2d, cmap, 
              label='2D-dTV', tcolor='w')
plot_pseudo3D(gs, 3, ic, ncols, result2d_2, 0, vmax, aspect2d, cmap)
plot_pseudo3D(gs, 6, ic, ncols, result2d_3, 0, vmax, aspect2d, cmap)
plot_pseudo3D(gs, 9, ic, ncols, result2d_4, 0, vmax, aspect2d, cmap)

ic += 1
plot_error(gs, 0, ic, ncols, S2d(result2d_1) - data, vmine, vmaxe, 
           label='residual', tcolor='k')
plot_error(gs, 3, ic, ncols, S2d(result2d_2) - data, vmine, vmaxe)
plot_error(gs, 6, ic, ncols, S2d(result2d_3) - data, vmine, vmaxe)
plot_error(gs, 9, ic, ncols, S2d(result2d_4) - data, vmine, vmaxe)   
    
ic += 2
plot_3D(gs, 0, ic, ncols, result3d_1, 0, vmax, aspecthigh, cmap, 
        label='3D-dTV (proposed)', tcolor='w')
plot_3D(gs, 3, ic, ncols, result3d_2, 0, vmax, aspecthigh, cmap)
plot_3D(gs, 6, ic, ncols, result3d_3, 0, vmax, aspecthigh, cmap)
plot_3D(gs, 9, ic, ncols, result3d_4, 0, vmax, aspecthigh, cmap, 
        cbar=True, cbarlabel='signal intensity [a.u.]')

ic += 1
plot_error(gs, 0, ic, ncols, S3d(result3d_1) - data, vmine, vmaxe, 
           label='residual', tcolor='k')
plot_error(gs, 3, ic, ncols, S3d(result3d_2) - data, vmine, vmaxe)
plot_error(gs, 6, ic, ncols, S3d(result3d_3) - data, vmine, vmaxe)
plot_error(gs, 9, ic, ncols, S3d(result3d_4) - data, vmine, vmaxe, 
           cbar=True, cbarlabel='signal intensity [a.u.]')

plt.show()
name_ = 'figure6'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))

    
#%% Figure 7
folder_data = '../processed_data'

def plot_comparison(name, param1, param2, suffix=None, figname='7'):
    file_data1 = '{}/{}_Lactate_40.npz'.format(folder_data, name)
    file_data2 = '{}/{}_Pyruvate_40.npz'.format(folder_data, name)
    ff1 = np.load(file_data1)
    ff2 = np.load(file_data2)
    guide = ff1['guide']
    mol1 = ff1['molecule']
    mol2 = ff2['molecule']

    scaling1 = misc.scaling(name[2:], mol1, folder=folder_data) * 1000
    scaling2 = misc.scaling(name[2:], mol2, folder=folder_data) * 1000
    vmax1 = 1 / scaling1
    vmax2 = 1 / scaling2
    data1 = ff1['data'] / scaling1
    data2 = ff2['data'] / scaling2
    
    if name == '3DPhantom':
        vminratio, vmaxratio = 0, .8
        ratiothresh = 0.2
        aspecthigh = 2/1.5 * (15/4)

    else:
        vminratio, vmaxratio = 0.05, 0.3
        ratiothresh = 0.04
        aspecthigh = 2/1.5
        
    aspectlow = 30/6   
    mapratio = 'gist_heat'
    mapratio = 'plasma'
    
    folder_result1 = '../pics/figure7/{}_Lactate_40'.format(name)
    result1 = np.load(folder_result1 + "/" + param1 + ".npz")['x'] / scaling1

    folder_result2 = '../pics/figure7/{}_Pyruvate_40'.format(name)
    result2 = np.load(folder_result2 + "/" + param2 + ".npz")['x'] / scaling2


    plt.close(1)
    ncols = 3
    nrows = 8
    figheight = 8 # 240 / 30
    plt.figure(1, figsize=(5, 5))
    gs = gridspec.GridSpec(nrows, ncols, height_ratios=[figheight, 1, 1, 0.1, 
                                                        figheight, 1, 1, 1.5], 
                           wspace=0.05, hspace=0.05, left=0, right=1, top=0.95, 
                           bottom=0.08) 
    
    plot_pseudo3D(gs, 0, 0, ncols, data1, 0, vmax1, aspectlow, misc.cmap(mol1), 
                  label= 'lactate', tcolor='w')
    plot_pseudo3D(gs, 0, 1, ncols, data2, 0, vmax2, aspectlow, misc.cmap(mol2), 
                  label= 'pyruvate', tcolor='w')

    plot_3D(gs, 4, 0, ncols, result1, 0, vmax1, aspecthigh, misc.cmap(mol1), 
            cbar=True, cbarlabel='signal intensity [a.u.]')
    plot_3D(gs, 4, 1, ncols, result2, 0, vmax2, aspecthigh, misc.cmap(mol2), 
            cbar=True, cbarlabel='signal intensity [a.u.]')

    mask = guide > ratiothresh
    ratio = (mask * result1 / (result1 + result2))
    plot_3D(gs, 4, 2, ncols, ratio, vminratio, vmaxratio, aspecthigh, mapratio, 
            cbar=True, cbarlabel='ratio')
    
    mask = np.squeeze(sktransform.resize(guide, (40, 40, 1))) > ratiothresh
    ratio = (mask*data1 / (data1 + data2))
    plot_pseudo3D(gs, 0, 2, ncols, ratio, vminratio, vmaxratio, aspectlow, 
                  mapratio, label='lac/(lac+pyr)', tcolor='w')
        
    plt.show()
    name_ = 'figure' + figname + '_' + name
    for f in formats:
        plt.savefig('{}/{}.{}'.format(folder_out, name_, f))
  
cases = [{'name': '3DHV-118',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'},
         {'name': '3DPhantom',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'}] 
    
for c in cases:
    plot_comparison(c['name'], c['param1'], c['param2'], c['suffix'])
    
    
cases = [{'name': '3DHV-109',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'},
         {'name': '3DHV-114',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'},
         {'name': '3DHV-117',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'},
         {'name': '3DHV-118',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'},
         {'name': '3DPhantom',
          'suffix': 'dtv',
          'param1': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995',
          'param2': 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'}] 
    
for c in cases:
    plot_comparison(c['name'], c['param1'], c['param2'], c['suffix'], 'S1')

    
#%% Figure 8: line plots
folder_data = '../processed_data'

def plot_lines(name, gs, ic, ncol, labels=['GM', 'WM', 'GM', 'WM'], i1=13, 
               legend=False, ylabel=None, title=False, vertical=False, wim=0.18):
    
    file_data1 = '{}/{}_Lactate_40.npz'.format(folder_data, name)
    file_data2 = '{}/{}_Pyruvate_40.npz'.format(folder_data, name)
    
    param1 = 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'
    folder_result1 = '../pics/figure7/{}_Lactate_40'.format(name)
    folder_result2 = '../pics/figure7/{}_Pyruvate_40'.format(name)
        
    ff1 = np.load(file_data1, allow_pickle=True)
    ff2 = np.load(file_data2, allow_pickle=True)
    
    guide = ff1['guide']
    mol1 = ff1['molecule']
    mol2 = ff2['molecule']
    scaling1 = misc.scaling(name[2:], mol1, folder=folder_data) * 1000
    scaling2 = misc.scaling(name[2:], mol2, folder=folder_data) * 1000
    data1 = ff1['data'] / scaling1
    data2 = ff2['data'] / scaling2  
    result1 = np.load(folder_result1 + "/" + param1 + ".npz")['x'] / scaling1
    result2 = np.load(folder_result2 + "/" + param1 + ".npz")['x'] / scaling2
    
    def plot_line(ax, low, high, guide, i1, ylabel=None, title=None):
        x1 = np.linspace(0,1,len(low))
        x2 = np.linspace(0,1,len(high))
        
        i2 = i1 * 4
        c = high.shape[2] // 2
        
        if vertical:
            ax2 = ax.twinx() 
            ax2.step(x2, guide[:, i2, c], color='lightgray', linestyle=':')
            ax2.tick_params(axis='y', labelcolor='lightgray')
            ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
            
            ax.step(x1, low[:, i1], color='salmon', linestyle='--')
            ax.step(x2, high[:, i2, c], color='cornflowerblue')
            ax.tick_params(axis='y', labelcolor='black')
            ax.get_xaxis().set_visible(False)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            
        else:
            ax2 = ax.twinx() 
            ax2.step(x2, guide[i2, :, c], color='lightgray', linestyle=':')
            ax2.tick_params(axis='y', labelcolor='lightgray')
            ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
            
            ax.step(x2, high[i2, :, c], color='cornflowerblue')
            ax.step(x1, low[i1, :], color='salmon', linestyle='--')
            ax.tick_params(axis='y', labelcolor='black')
            ax.get_xaxis().set_visible(False)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        if title is not None:
            plt.title(title)
            
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    
        plt.show()
        
        return ax
    
    if title:
        stitle = 'lactate'
    else:
        stitle = None
    ax = plot_line(plt.subplot(gs[ic]), data1, result1, guide, i1, title=stitle, 
                    ylabel=ylabel)
    
    if title:
        stitle = 'pyruvate'
    else:
        stitle = None
    plot_line(plt.subplot(gs[ic+1]), data2, result2, guide, i1, title=stitle)
    
    if title:
        stitle = 'lac/(lac+pyr)'
    else:
        stitle = None
    plot_line(plt.subplot(gs[ic+2]), data1/(data1+data2), 
              result1/np.maximum(result1+result2, 1e-6), 
              guide, i1, title=stitle)
        
    p = ax.get_position()
    w = wim
    a = plt.axes([p.x1-0.55*w, p.y1-0.92*w, w, w])
    xx = guide.copy()
    i2 = i1 * 4
    c = xx.shape[2] // 2
    
    r = 2
    rr = r + 1
    if vertical:
        xx[:, i2-r:i2+rr, c] = 1
    else:
        xx[i2-r:i2+rr, :, c] = 1
    a.imshow(xx[:, :, c], cmap='gray', vmax=1)
    plt.axis('off')

plt.close(1)
ncol = 3
nrow = 2
fig = plt.figure(1, figsize=(10,4))
gs = fig.add_gridspec(nrow, ncol, left=0.07, right=0.965, top=0.94, 
                      bottom=0.09, hspace=0.1, wspace=0.3)
    

plot_lines('3DPhantom', gs=gs, ic=0, ncol=ncol, ylabel='$\it{in~vitro}$', i1=16, title=True)
plot_lines('3DHV-118', gs=gs, ic=3, ncol=ncol, ylabel='$\it{in~vivo}$ example', i1=28)

from matplotlib.lines import Line2D

custom_lines = [Line2D([0,0],[0,0],color='lightgray', linestyle=':', 
                       label='guide'),
                Line2D([0,0],[0,0],color='salmon', linestyle='--', 
                       label='data'),
                Line2D([0,0],[0,0],color='cornflowerblue', 
                       label='recon')]

fig.legend(custom_lines, ['guide', 'data', 'recon'], ncol=3, loc='lower right')

plt.show()
name_ = 'figure8'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))

plt.close(1)
ncol = 3
nrow = 5
fig = plt.figure(1, figsize=(10,13))
gs = fig.add_gridspec(nrow, ncol, left=0.07, right=0.965, top=0.97, 
                      bottom=0.05, hspace=0.1, wspace=0.3)
    

plot_lines('3DPhantom', gs=gs, ic=0, ncol=ncol, ylabel='$\it{in~vitro}$', i1=16, title=True, wim=0.1)
plot_lines('3DHV-109', gs=gs, ic=3, ncol=ncol, ylabel='$\it{in~vivo}~1$', i1=28, wim=0.1)
plot_lines('3DHV-114', gs=gs, ic=6, ncol=ncol, ylabel='$\it{in~vivo}~2$', i1=28, wim=0.1)
plot_lines('3DHV-117', gs=gs, ic=9, ncol=ncol, ylabel='$\it{in~vivo}~3$', i1=28, wim=0.1)
plot_lines('3DHV-118', gs=gs, ic=12, ncol=ncol, ylabel='$\it{in~vivo}~4$', i1=28, wim=0.1)

from matplotlib.lines import Line2D

custom_lines = [Line2D([0,0],[0,0],color='lightgray', linestyle=':', 
                       label='guide'),
                Line2D([0,0],[0,0],color='salmon', linestyle='--', 
                       label='data'),
                Line2D([0,0],[0,0],color='cornflowerblue', 
                       label='recon')]

fig.legend(custom_lines, ['guide', 'data', 'recon'], ncol=3, loc='lower right')
plt.show()
name_ = 'figureS2'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))


#%% Figure 9: Box plots
folder_data = '../processed_data'

def plot_boxes(name, gs, ic, ncol, labels=['GM', 'WM', 'GM', 'WM', 'GM', 'WM'], legend=False, 
               ylabel=False, title=None):
    
    file_data1 = '{}/{}_Lactate_40.npz'.format(folder_data, name)
    file_data2 = '{}/{}_Pyruvate_40.npz'.format(folder_data, name)
    
    param1 = 'dtv_s1.00e+02_e1.00e-02_a5.00e-02_g0.9995'
    folder_result1 = '../pics/figure7/{}_Lactate_40'.format(name)
    folder_result2 = '../pics/figure7/{}_Pyruvate_40'.format(name)
        
    ff1 = np.load(file_data1, allow_pickle=True)
    ff2 = np.load(file_data2, allow_pickle=True)
    
    guide = ff1['guide']
    mol1 = ff1['molecule']
    mol2 = ff2['molecule']
    scaling1 = misc.scaling(name[2:], mol1, folder=folder_data) * 1000
    scaling2 = misc.scaling(name[2:], mol2, folder=folder_data) * 1000
    data1 = ff1['data'] / scaling1
    data2 = ff2['data'] / scaling2
    seg = ff1['seg']
    
    result1 = np.load(folder_result1 + "/" + param1 + ".npz")['x'] / scaling1
    result2 = np.load(folder_result2 + "/" + param1 + ".npz")['x'] / scaling2
    
    U = odl.uniform_discr([-1, -1, -guide.shape[2]/150], [1, 1, guide.shape[2]/150], guide.shape)
    V = odl.uniform_discr([-1, -1], [1, 1], data1.shape)
    S3d = misc.Subsampling(U, V, margin=((0,0),(0,0), (1,1)))
    
    seglow = [S3d(seg[:,:,:,i]).asarray() for i in range(seg.shape[3])]
    seghigh = [seg[:,:,:,i] for i in range(seg.shape[3])]
    seghigh2 = [S3d.inverse(S3d(seg[:,:,:,i])).asarray() for i in range(seg.shape[3])]
    
    def plot_box(ax, low, high, seglow, seghigh, seghigh2, ylabel=None, color=None, labels=None):
        low_gm = low[seglow[0]>0.7]
        low_wm = low[seglow[1]>0.7]
        
        high_gm = high[seghigh[0]>0.7]
        high_wm = high[seghigh[1]>0.7]

        high_gm2 = high[seghigh2[0]>0.7]
        high_wm2 = high[seghigh2[1]>0.7]
        
        data_to_plot = [low_gm, low_wm, high_gm2, high_wm2, high_gm, high_wm]
        
        if labels is None:
            labels = ['', '', '', '']
            
        bp = ax.boxplot(data_to_plot, showfliers=False, whis=(10, 90), patch_artist=True, labels=labels)
        if ylabel is not None:
            plt.ylabel(ylabel)
         
        colors = ['salmon', 'salmon', 'orchid', 'orchid', 'cornflowerblue', 'cornflowerblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        for patch in bp['medians']:
            patch.set_color('k')
            patch.set_linewidth(3)
    
        plt.show()
        
        return low_gm, low_wm, high_gm2, high_wm2, high_gm, high_wm
        
    llgm, llwm, lhgm2, lhwm2, lhgm, lhwm = plot_box(plt.subplot(gs[ic]), data1, result1, seglow, seghigh, seghigh2, ylabel=ylabel,
             labels=labels)
    if title:
        plt.title('lactate')
    plgm, plwm, phgm2, phwm2, phgm, phwm = plot_box(plt.subplot(gs[ic+1]), data2, result2, seglow, seghigh, seghigh2,
             labels=labels)
    if title:
        plt.title('pyruvate')
    plot_box(plt.subplot(gs[ic+2]), data1/(data1+data2), result1/np.maximum(result1+result2, 1e-6), seglow, seghigh, seghigh2,
             labels=labels)
    if title:
        plt.title('lac/(lac+pyr)')

    return [llgm, llwm, lhgm2, lhwm2, lhgm, lhwm, plgm, plwm, phgm2, phwm2, phgm, phwm]

ncol = 3
nrow = 2

plt.close(1)
fig = plt.figure(1, figsize=(10,4))
gs = fig.add_gridspec(nrow, ncol, left=0.06, right=0.995, top=0.93, bottom=0.15, hspace=0.2, wspace=0.2)
    
plot_boxes('3DPhantom', labels=['80-U', '40-U', '80-U', '40-U', '80-U', '40-U'], gs=gs, ic=0, ncol=ncol, title=True, ylabel='$\it{in~vitro}$')
plot_boxes('3DHV-118', gs=gs, ic=3, ncol=ncol, ylabel='$\it{in~vivo}$ example')

from matplotlib.patches import Patch

custom_lines = [Patch(facecolor='salmon', edgecolor='k', label='data'),
                Patch(facecolor='orchid', edgecolor='k', label='recon+LR seg'),
                Patch(facecolor='cornflowerblue', edgecolor='k', label='recon+HR seg')]

fig.legend(custom_lines, ['data+LR seg', 'recon+LR seg', 'recon+HR seg'], ncol=3, loc='lower right')
plt.show()
name_ = 'figure9'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))


ncol = 3
nrow = 5

plt.close(1)
fig = plt.figure(1, figsize=(10,13))
gs = fig.add_gridspec(nrow, ncol, left=0.06, right=0.995, top=0.97, bottom=0.08, hspace=0.25, wspace=0.2)

stats = [None] * 4

plot_boxes('3DPhantom', labels=['80-U', '40-U', '80-U', '40-U', '80-U', '40-U'], 
           gs=gs, ic=0, ncol=ncol, title=True, ylabel='$\it{in~vitro}$')
stats[0] = plot_boxes('3DHV-109', gs=gs, ic=3, ncol=ncol, ylabel='$\it{in~vivo}$ 1')
stats[1] = plot_boxes('3DHV-114', gs=gs, ic=6, ncol=ncol, ylabel='$\it{in~vivo}$ 2') 
stats[2] = plot_boxes('3DHV-117', gs=gs, ic=9, ncol=ncol, ylabel='$\it{in~vivo}$ 3')
stats[3] = plot_boxes('3DHV-118',legend=True, gs=gs, ic=12, ncol=ncol, ylabel='$\it{in~vivo}$ 4')

from matplotlib.patches import Patch

custom_lines = [Patch(facecolor='salmon', edgecolor='k', label='data+LR seg'),
                Patch(facecolor='orchid', edgecolor='k', label='recon+LR seg'),
                Patch(facecolor='cornflowerblue', edgecolor='k', label='recon+HR seg')]

fig.legend(custom_lines, ['data', 'recon+low res seg', 'recon'], ncol=3, loc='lower right')
plt.show()
name_ = 'figureS3'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))


#%% Figure 10
n = 4 # number of subjects
k = 6

stats2 = [[stats[p][q] for p in range(n)] for q in range(2*k)]

# add ratio to stats
for i in range(k):
        s = [stats2[i][j]/(stats2[i][j]+stats2[i+k][j]) for j in range(n)]
        stats2.append(s)

# compute means
mean_stats2 = []
for i in range(9):
    s = [np.mean(stats2[2*i][j])/np.mean(stats2[2*i+1][j]) for j in range(n)]
    mean_stats2.append(s)
        
plt.close(1)
fig = plt.figure(1, figsize=(10,2.5))
plt.clf()
gs = fig.add_gridspec(1, 3, left=0.05, right=0.995, bottom=0.2, top=0.88, 
                      wspace=0.3)
         
def draw_dots(ax, where, data, color, label=None):
    x = np.array(data)
    ax.scatter(where*np.ones(x.shape), x, marker='o', color=color, edgecolor='black', label=label)
    ax.plot([where-0.2, where+0.2], [np.mean(x), np.mean(x)], linewidth=2, 
            color='black')
    print('{:2.2f}+-{:2.2f}'.format(np.mean(x), np.std(x)))
    
def draw_plot(ax, data1, title, ylim=None, labels=False):
    
    if labels:
        draw_dots(ax, 1, data1[0], 'salmon', label='data+LR seg')
        draw_dots(ax, 2, data1[1], 'orchid', label='recon+LR seg')
        draw_dots(ax, 3, data1[2], 'cornflowerblue', label='recon+HR seg')
    else:
        draw_dots(ax, 1, data1[0], 'salmon')
        draw_dots(ax, 2, data1[1], 'orchid')
        draw_dots(ax, 3, data1[2], 'cornflowerblue')
        
    plt.xticks(ticks=[1, 2, 3], labels=['', '', ''])
    
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
          
draw_plot(plt.subplot(gs[0]), mean_stats2[0:3], 'lactate', 
          ylim=[0.8, 2.1], labels=True)
draw_plot(plt.subplot(gs[1]), mean_stats2[3:6], 'pyruvate', ylim=[0.8, 2.1])
draw_plot(plt.subplot(gs[2]), mean_stats2[6:9], 'lac/(lac+pyr)', ylim=[0.8, 2.1])
plt.show()

fig.legend(ncol=3, loc='lower right')

plt.show()
name_ = 'figure10'
for f in formats:
    plt.savefig('{}/{}.{}'.format(folder_out, name_, f))