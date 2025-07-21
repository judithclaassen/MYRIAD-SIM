# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:28:43 2025

@author: jcl202
"""
#convert to uniform margins

import xarray as xr
import scipy.stats as st
import numpy as np
import ast

#%% code to create scipy distribution format
def create_distribution(name):
    try:
        return getattr(st, name)
    except AttributeError:
        return None

#%%
names =[ 't2m', 'pr', 'ws']
files = ['meant2m.nc', 'sumpr.nc', 'meanws.nc']
filepath =  '..../MYRIAD-SIM/data/' 
modelfilepath =   '..../MYRIAD-SIM/varpar/' 
#%%loading data function
def loadselected(fn):
    inter = xr.open_dataset(fn).load()
    return inter 
#%%
num = list(range(0,16))
for namei in names:
    fn = filepath + namei  + 'coarse.nc'
    trial = loadselected(fn)
    keyi = list(trial.keys())[0]
    trial = trial.rename({keyi: "u"})
    img = trial.u.values
    img2 = img.copy()
    mask2 = ~np.isnan(img[0])
    ys, xs = np.where(mask2 == True)
    n1 =  mask2.shape[1]
    n2 = mask2.shape[0]
    distsdata2 = loadselected(modelfilepath + 'distdata' + namei + '.nc')
    distsi = distsdata2.distname.values
    dists = np.empty((13,n2, n1), dtype=object)
    for j in list(range(1,13)):
        for i in range(len(ys)):
            dists[j, ys[i], xs[i]] = create_distribution(distsi[j, ys[i], xs[i]])
            
    globals()['dists' + namei] = dists
    distsdata2 =  loadselected(modelfilepath+ 'distdataparams' + namei + '.nc')
    params = np.empty((13,n2, n1), dtype=object)
    paramsi = distsdata2.distparams.values
    for j in list(range(1,13)):
      for i in range(len(ys)):
        params[j, ys[i], xs[i]] = ast.literal_eval(paramsi[j, ys[i], xs[i]].replace("np.float64", ""))
    globals()['params' + namei] = params  
    months = list(trial['time'].dt.month.values)
    locals()[namei + '_img2'] = img2.copy()
    img = trial.u.values
    img2 = img.copy()
    n3 = trial.u.shape[0]

    for j in list(range(1,13)):
        print(j)
        for i in range(len(ys)):
            dsx = dists[j, ys[i], xs[i]]
            paramx = params[j, ys[i], xs[i]]
            img2[np.where(months == np.int64(j)), ys[i], xs[i]] =  dsx.cdf(img[np.where(months == np.int64(j)), ys[i], xs[i]], *paramx )

    trial.u.values = img2.copy()
    trial.to_netcdf(filepath + namei +'coarse_u2.nc')