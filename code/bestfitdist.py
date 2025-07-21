# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:52:45 2025

@author: jcl202
"""


# code  to fit best dist
import xarray as xr
import sys
import scipy.stats as st
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata
import matplotlib.pyplot as plt

#%% opening the netcdfdata
def create_distribution(name):
    try:
        return getattr(st, name)
    except AttributeError:
        return None
    
#%% data information

data_folder = '..../MYRIAD-SIM/data/' #file where data is stored
model_file_folder = '..../MYRIAD-SIM/varpar/' #where variable distributions should be saved 
names = ['t2m','pr','ws'] #list of variables
varnum = 2 # which variable to fit to, where 0 is the first variable in names list

#%% load in data
fn =data_folder + names[varnum] + 'coarse.nc'
namei = names[varnum]
def loadselected(fn): #load function
    inter = xr.open_dataset(fn).load()
    return inter
print(fn)
inter = loadselected(fn)
#%%
if namei == 'pr':
  inter.tp.values[inter.tp.values<0] = 0
def neededinfo(inter): #function to obtain needed information
    lon = inter.longitude
    lat = inter.latitude
    keyi = list(inter.keys())[0]
    mask = ~np.isnan(inter[keyi][5].values) #the gridcells where there are values
    ys, xs = np.where(mask == True) #the x and y values where the gridcells are values
    n2, n1 = inter[keyi][5].values.shape
    return lon, lat,  ys, xs, n1, n2, keyi

lon, lat, ys, xs, n1, n2, keyi = neededinfo(inter)
#%% get key of variable
print(namei, ' and ', keyi)
#%% function to fit bet distribution adapted from the VineCopulas package
def  best_fit_distribution(data, dists = []):
    """
    Fits the best continuous distribution to data.

    Arguments:
        *data* : The data which has to be fit as a 1-d numpy array.
        
        *dists* : Specify specific distributions if only specific distributions need to be tested, provided as a list.

    Returns:
     *bestdist* : the best distribution and its parameters.
    """
    # distributions
    distributions = {
        "Beta": st.beta,
        "Birnbaum-Saunders": st.burr,
        "Exponential": st.expon,
        "Extreme value": st.genextreme,
        "Gamma": st.gamma,
        "Generalized Pareto": st.genpareto,
        "Inverse Gaussian": st.invgauss,
        "Logistic": st.logistic,
        "Log-logistic": st.fisk,
        "Lognormal": st.lognorm,
        "Nakagami": st.nakagami,
        "Normal": st.norm,
        "Rayleigh": st.rayleigh,
        "Rician": st.rice,
        "t location-scale": st.t,
        "Weibull": st.weibull_min,
    }
    
    if len(dists)> 0:
        keys_list = list(distributions.keys())
    
        keys_list2 = []
        for i in dists:
            keys_list2.append(keys_list[i])
            
        distributions = dict((k, distributions[k]) for k in keys_list2
               if k in distributions)
    
   
        
    best_distributions = []
    ranks = np.apply_along_axis(rankdata, axis=0, arr=data)
    n = data.shape[0]
    u = (ranks - 1) / (n - 1)
    u[u==1] = 0.9999
    u[u ==0 ] = 0.0001

    disn = -1
    for name, distribution in distributions.items():
        disn = disn + 1
        try:
            #= Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                try:
                    params1 = distribution.fit(data,floc =int(min(data)), method='mse')

                except:
                    params1 = distribution.fit(data, floc =int(np.floor(min(data))))
                try:
                    params2 = distribution.fit(data, method='mse')
                except:
                    params2 = distribution.fit(data)
                convert = distribution.ppf(u, *params1)
                if np.any(np.isinf(convert)):

                    
                    criterion_value1 = np.inf
                else:
                    criterion_value1 = mean_squared_error(data, convert) + mean_squared_error([np.nanmax(data), np.nanmin(data)], [np.nanmax(convert), np.nanmin(convert)])
                    
                convert = distribution.ppf(u, *params2)
                if np.any(np.isinf(convert)):

                    
                    criterion_value2 = np.inf
                else:
                    criterion_value2 = mean_squared_error(data, convert) + mean_squared_error([np.nanmax(data), np.nanmin(data)], [np.nanmax(convert), np.nanmin(convert)])
                    
                try:
                    params3 = distribution.fit(data,floc =int(min(data)), method='mm')
                except:
                    params3 = distribution.fit(data, floc =int(np.floor(min(data))))
                try:
                    params4= distribution.fit(data, method='mm')
                except:
                    params4 = distribution.fit(data)

                convert = distribution.ppf(u, *params3)
                if np.any(np.isinf(convert)):

     
                    criterion_value3 = np.inf
                else:
                    criterion_value3 = mean_squared_error(data, convert) + mean_squared_error([np.nanmax(data), np.nanmin(data)], [np.nanmax(convert), np.nanmin(convert)])

                convert = distribution.ppf(u, *params4)
                if np.any(np.isinf(convert)):

                    criterion_value4 = np.inf
                else:
                    criterion_value4 = mean_squared_error(data, convert) + mean_squared_error([np.nanmax(data), np.nanmin(data)], [np.nanmax(convert), np.nanmin(convert)])
   
                cv = [ criterion_value1,  criterion_value2, criterion_value3, criterion_value4]    
                mincv = np.where(cv == np.min(cv))[0][0] + 1
                params = locals()['params' + str(mincv)]
                criterion_value = locals()['criterion_value' + str(mincv)]
                best_distributions.append((distribution, params, criterion_value))
        except Exception:
            
            pass

    return sorted(best_distributions, key=lambda x: x[2])[0]

#%% dimension of grid for dists and params, run if need to be fit
dists = np.empty((13,n2, n1), dtype=object)
params = np.empty((13, n2, n1), dtype=object)

#%%#%%distribution per month

dis0f = False
start_time2 = time.time()
for j in range(1,13):
    for i in range(len(ys)):
                start_time2 = time.time()
                latj = lat.values[ys[i]]
                lonk = lon.values[xs[i]]
                if namei == 'pr':
                    if (sum(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values) == 0) and (dis0f == True):
                        dis = dis0
                    elif (sum(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values) == 0):
                        dis0f = True
                        dis0 =  best_fit_distribution(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values,dists = list(range(16)))
                        dis = dis0
                    else:
                        dis = best_fit_distribution(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values,dists =  list(range(16)))
                elif namei == 'ws':
                    dis = best_fit_distribution(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values, dists = list(range(16)))
                elif namei == 't2m':
                    dis = best_fit_distribution(inter.sel(latitude=latj, longitude=lonk, method='nearest')[keyi].sel(time=inter.time.dt.month.isin(j)).values,dists = list(range(16)))

                dists[j,ys[i], xs[i]] = dis[0]
                params[j, ys[i], xs[i]] = dis[1]
                
print((time.time() - start_time2))    
#%% put distis into a frame to save
dists2 = dists.copy()
for j in range(1,13):
    for i in range(len(ys)):
        latj = lat.values[ys[i]]
        lonk = lon.values[xs[i]]
        dists2[j,ys[i], xs[i]] = dists[j,ys[i], xs[i]].name

    
#%% put dists into a format to save
dists2 = dists2.astype(str)

#%% put dists into netcdf
distsdata = xr.DataArray(
        dists2,
        dims=('months', "latitude", "longitude"),
        coords={'months' : list(range(13)), "latitude": inter.latitude.values, "longitude": inter.longitude.values}
    )
distsdata.name = "distname"

distsdata.to_netcdf(model_file_folder +'distdata' + namei + '.nc')
#%% format params to save and save as netcdf

params2 = params.copy()

for j in range(1,13):
    for i in range(len(ys)):
        latj = lat.values[ys[i]]
        lonk = lon.values[xs[i]]
        params2[j, ys[i], xs[i]] = str(params2[j,  ys[i], xs[i]])

       
params2 = params2.astype(str)       
distsdata = xr.DataArray(
        params2,
        dims=('months', "latitude", "longitude"),
        coords={'months' : list(range(13)), "latitude":  inter.latitude.values, "longitude": inter.longitude.values}
    )
distsdata.name = "distparams"

distsdata.to_netcdf(model_file_folder + 'distdataparams' + namei + '.nc')  


