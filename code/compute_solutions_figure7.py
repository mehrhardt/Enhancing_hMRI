# create results as shown in figure 7 of
# M. J. Ehrhardt, F. A. Gallagher, M. A. McLean, and C.-B. Schoenlieb, 
# Enhancing the Spatial Resolution of Hyperpolarized Carbon-13 MRI of Human 
# Brain Metabolism using Structure Guidance, MRM, https://doi.org/10.1002/mrm.29045, 2021.

# load modules
import numpy as np
import os
import misc

# select data sets
names = ['3DHV-117_Lactate_40', 
         '3DHV-117_Pyruvate_40',
         '3DHV-114_Lactate_40', 
         '3DHV-114_Pyruvate_40',
         '3DHV-118_Lactate_40', 
         '3DHV-118_Pyruvate_40',
         '3DHV-109_Lactate_40',
         '3DHV-109_Pyruvate_40',
         '3DPhantom_Lactate_40',
         '3DPhantom_Pyruvate_40']

# select parameters
alphas = [5e-2] # regularization parameter
gammas = [0.9995] # parameter in dtv
etas = [1e-2] # parameter in dtv
scalings = [100] # algorithm parameter
niter = 1000 # algorithm parameter
        
# process all data for all parameters
for name in names:
    print('=== Dataset {}'.format(name))
    
    # set and create out folder
    folder_out = '../pics/figure7/{}'.format(name)    
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # load data
    dic = np.load('../processed_data/{}.npz'.format(name), allow_pickle=True)
    data = dic['data']
    guide = dic['guide']
    molecule = dic['molecule']
    seg = dic['seg']
    
    # run recon for all parameters
    for gamma in gammas:
        for alpha in alphas: 
            for eta in etas:
                for scaling in scalings:
                    misc.recon_dtv(guide, data, seg, folder_out, alpha, eta, 
                                   gamma, niter, molecule, scaling=scaling)    
