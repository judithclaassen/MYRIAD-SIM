# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 09:58:37 2025

@author: jcl202
"""
#fit model

import xarray as xr
import sys
import numpy as np
from vinecopulas.marginals import *
from vinecopulas.bivariate import *
from vinecopulas.vinecopula import *
import time
from itertools import combinations
from joblib import Parallel, delayed
#%%
names =[ 't2m','pr', 'ws']
files = ['meant2m.nc','meandpt.nc', 'sumpr.nc', 'meanws.nc', 'meanwd.nc']
filepath = '..../MYRIAD-SIM/data/'
modelfilepath2 = '.../MYRIAD-SIM/modelpar/'
varnum = 2

#%% load files function
def loadselected(fn):
    inter = xr.open_dataset(fn).load()
    return inter
#%% open files to fit
for k1 in range(varnum+1):
    fn = filepath+ names[k1] + 'coarse_u2.nc'
    namei = names[k1]
    locals()[namei + '_u2'] =loadselected(fn)
    keyi = list(locals()[namei + '_u2'].keys())[0]
    locals()[namei+'_key'] = keyi


#%% dimensions info

n1 =  locals()[namei + '_u2'][keyi][5].values.shape[1]
n2 = locals()[namei + '_u2'][keyi][5].values.shape[0]
mask = ~np.isnan(locals()[namei + '_u2'][keyi][5].values) #the gridcells where there are values
ys, xs = np.where(mask == True) #the x and y values where the gridcells are value
lon = locals()[namei + '_u2'].longitude
lat = locals()[namei + '_u2'].latitude
months = list(locals()[namei + '_u2']['time'].dt.month.values)

#%% computes information on sampling order and variables needed for each grid
aseq = [0] 
for i in range(1,int(max(n1,n2)/2)):
    aseq.append(aseq[-1] + i)

my_array = np.zeros((n2, n1), dtype=int)

for j in range(n1):
    for i in range(n2):
        distance_y = abs(j - (n1 // 2))
        distance_x = abs(i - (n2 // 2))
        for k in range(int(max(n1,n2)/2)):
            if distance_y > k or distance_x > k:
                my_array[i, j] =distance_x + distance_y + aseq[k]


my_array2  = my_array.copy()
my_array[abs(n2 // 2)-1, abs((n1 // 2))] = 0
my_array[abs(n2 // 2)+1, abs((n1 // 2))] = 0
my_array= my_array.astype(float)
my_array2= my_array2.astype(float)
steps = np.unique(my_array2).astype(int)
steps = steps[~np.isnan(steps)] 
my_array = np.empty((n2, n1))
my_array[:] = np.nan
my_array[my_array2 == 0] = 0


k=0

for i in steps:
    ys, xs = np.where(my_array2 == i)
    for j in range(len(ys)):
        
        if mask[ys[j], xs[j]] == True:
            my_array[ys[j], xs[j]] = k
            k = k +1
           
steps = np.unique(my_array).astype(int)[:-1]
steps = steps[~np.isnan(steps)]
modelrows = []
modelnum = []

sampl = np.zeros((n2, n1))
elements = [0, 1, 2, 3, 4, 5, 6, 7]


all_combinations = []
for r in range(2, len(elements) + 1):
    all_combinations.extend(combinations(elements, r))

ni = 1
nin = ni*-1
l = list(range(nin,ni + 1))
combinations1 = []
for j in l:
    for k in l:
        if k == 0 and j == 0:
            continue
        if (k in [nin,ni]) or (j in [nin,ni]):
            combinations1.append([[j,k]])
         
combinations2 = []   
modelnum = []       
for i in steps:
    y, x = np.where(my_array == i)
    if i == 0:
        modelnum.append(0)
        sampl[y,x] = 1
        combinations2.append(names)
    else:
        ys, xs= np.where(sampl== 1)
        mask3 = (ys != y) | (xs != x)
        ys, xs = ys[mask3], xs[mask3]
        distances = np.sqrt((ys - y)**2 + (xs - x)**2)
        closest_idx = np.argmin(distances)
        closest_y, closest_x = ys[closest_idx], xs[closest_idx]
        ni = max([abs(closest_y-y), abs(closest_x-x)])[0]
        nin = ni*-1
        l = list(range(nin,ni + 1))
        combinations1 = []
        for j in l:
            for k in l:
                if k == 0 and j == 0:
                    continue
                if (k in [nin,ni]) or (j in [nin,ni]):
                    combinations1.append([[j,k]])

        l = []
        for s2 in combinations1:
            try:
                if y+s2[0][0] < 0 or x+s2[0][1] < 0:
                    continue
                elif sampl[y+s2[0][0], x+s2[0][1]] == 1:
                    l = l +  s2
            except:
                continue
        if len(l) > 0:
            sampl[y,x] = 1
            if l in combinations2:
                for s in range(len(combinations2)):
                    if combinations2[s] == l:
                        modelnum.append(s)

                        break
            else:
                combinations2.append(l)
                modelnum.append(len(combinations2)-1)
        else:
            break
                    
                
for i in range(1,len(combinations2)):
    l = []
    for j in combinations2[i]:
        for s in names[:varnum+1]:
            l = l + [[s] + j]
    combinations2[i] = l   
    combinations2[i] =  names[:varnum+1] + combinations2[i]                
        

modelspatial = np.zeros((n2, n1), dtype=int)* np.nan


for i in steps:
    yy,xx = np.where(my_array ==i)
    modelspatial[yy,xx] =  modelnum[i]

#%% set up variables and matrices 

lon = locals()[namei+'_u2'].longitude
lat = locals()[namei+'_u2'].latitude
cops = [1,6,7,8,9,10] #deifines copulas to fit
variab = ['M', 'P', 'C', 'orders']
for i in variab:
    for j in range(1,13):
        globals()[i + str(j)] = np.empty(modelspatial.shape, dtype=object)
orders = np.empty(modelspatial.shape, dtype=object)
mask2 = ~np.isnan(locals()[namei + '_u2' ][keyi][5].values) #the gridcells where there are values
ys, xs = np.where(mask2 == True) #the x and y values where the gridcells are values

#%% fit vinecopulas for t=1 in parallel for one month


def fitcopulat1(yi, xi, m, t2m_u2, pr_u2, ws_u2,t2m_key, pr_key, ws_key):
    latj = lat.values[yi]
    lonk = lon.values[xi]
    s = combinations2[int(modelspatial[yi, xi])].copy()
    u = np.empty((0, len(s)))
    x = locals()[s[0]+'_u2'][keyi].loc[dict(latitude=latj, longitude=lonk)].sel(time=locals()[s[0]+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)
    for s2 in s[1:]:
        if type(s2) != str:
            if len(s2) == 2:
                if yi+s2[0] >= n2 or xi+s2[1] >= n1:
                    continue
                elif sum(np.isnan(locals()[namei+'_u2'][keyi].values[:, yi + s2[0], xi+s2[1]])) > 1: 
                    break
                else:
                    x = np.hstack((x, locals()[namei+'_u2'][keyi].sel(time=locals()[namei+'_u2'].time.dt.month.isin(m)).values[:, yi + s2[0], xi+s2[1]].reshape(-1, 1)))
            elif len(s2) == 3:
                if yi+s2[1] >= n2 or xi+s2[2] >= n1:
                    continue
                else:
                    x = np.hstack((x, locals()[s2[0]+'_u2'][locals()[s2[0]+'_key']].sel(time=locals()[namei+'_u2'].time.dt.month.isin(m)).values[:, yi + s2[1], xi+s2[2]].reshape(-1, 1)))
                 
        else:
            x = np.hstack((x, locals()[s2+'_u2'][locals()[s2+'_key']].loc[dict(latitude=latj, longitude=lonk)].sel(time=locals()[namei+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)))
    if x.shape[1] == len(s):
        u =  np.vstack([u, x])
        u = u[~np.isnan(u).any(axis=1)]
    Mc, Pc, Cc = fit_conditionalvine(u, list(range(0,varnum+1)), cops ,vine = 'D', condition = 1, printing = False)
    inte = np.array(list(np.diag(Mc[::-1])[::-1])).astype(int) 
    orders= [s[ind] for ind in inte]
    result = {
        'C': Cc, 
        'P': Pc, 
        'M': Mc, 
        'orders': orders
    }
    return (yi, xi, result)

m = 1 #the month you want to fit, where 1 is January and 12 is December
runthis = True #false if t=1 does not need to be fit
if runthis == True:
    start_time2 = time.time() 
    print(f"Processing month {m}")
    results = Parallel(n_jobs=-1)(
        delayed(fitcopulat1)(ys[i], xs[i], m, t2m_u2, pr_u2, ws_u2, t2m_key, pr_key, ws_key) for i in range(len(ys))
    )

    for yi, xi, result in results:
        globals()['C' + str(m)][yi, xi] = result['C']
        globals()['P' + str(m)][yi, xi] = result['P']
        globals()['M' + str(m)][yi, xi] = result['M']
        globals()['orders' + str(m)][yi, xi] = result['orders']
    print((time.time() - start_time2))
    
    variab = ['M', 'P', 'C', 'orders']
#save files
    for k in variab:
        C_copy  =  locals()[k + str(m)].copy()
        for j in range(len(ys)):
            if type(C_copy[ys[j], xs[j]]) == int:
                C_copy[ys[j], xs[j]] =str(C_copy[ys[j], xs[j]])
            
                
            elif type(C_copy[ys[j], xs[j]]) == np.ndarray:
                C_copy[ys[j], xs[j]] = str(C_copy[ys[j], xs[j]].tolist())

            else: C_copy[ys[j], xs[j]] =str(C_copy[ys[j], xs[j]])
        C_copy = C_copy.astype(str)  
        C_copy = xr.DataArray(
            C_copy,
            dims=("latitude", "longitude"),
            coords={"latitude": lat.values, "longitude":lon.values}
        )
        
        C_copy.to_netcdf(modelfilepath2 + 'all' +  k + str(m) + '.nc')

#%%
order2 = np.empty(modelspatial.shape, dtype=object)
variab = ['M2_', 'P2_', 'C2_', 'orders2_']
for i in variab:
    for j in range(1,13):
        globals()[i + str(j)] = np.empty(modelspatial.shape, dtype=object)
#%% add previous timestep to samples
l2 = []
for j in names[:varnum+1]:
    l2 = l2 + [[j] + [-1]]


#%% fit vinecopulas for t>1 in parallel for one month
def fitcopulata1(yi, xi, m, t2m_u2, pr_u2, ws_u2,t2m_key, pr_key, ws_key):
    latj = lat.values[yi]
    lonk = lon.values[xi]
    s = combinations2[int(modelspatial[yi,xi])].copy()
    s = s + l2
    u = np.empty((0, len(s)))
    x = locals()[s[0]+'_u2'][keyi].loc[dict(latitude=latj, longitude=lonk)].sel(time=locals()[s[0]+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)[1:]
    for s2 in s[1:]:
        if type(s2) != str:
            if len(s2) == 2:
                x = np.hstack((x, locals()[s2[0]+'_u2'][keyi].loc[dict(latitude=latj, longitude=lonk)].sel(time=locals()[s2[0]+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)[:-1]))
    
            elif len(s2) == 3:
                if yi+s2[1] < 0 or xi+s2[2] < 0:
                    continue
                elif yi+s2[1] >= n2 or xi+s2[2] >= n1:
                    continue
                else:
                    x = np.hstack((x, locals()[s2[0]+'_u2'][locals()[s2[0]+'_key']].loc[dict(latitude=lat.values[yi+s2[1]], longitude=lon.values[xi+s2[2]])].sel(time=locals()[namei+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)[1:]))
                
            
        else:
            x = np.hstack((x, locals()[s2+'_u2'][locals()[s2+'_key']].loc[dict(latitude=latj, longitude=lonk)].sel(time=locals()[namei+'_u2'].time.dt.month.isin(m)).values.reshape(-1, 1)[1:]))
        
    if x.shape[1] == len(s):
        u =  np.vstack([u, x])
        u = u[~np.isnan(u).any(axis=1)]
        Mc, Pc, Cc = fit_conditionalvine(u,  list(range(0,varnum+1)), cops ,vine = 'D', condition = 1, printing = False)
        inte = np.array(list(np.diag(Mc[::-1])[::-1])).astype(int) 
        orders= [s[ind] for ind in inte]
        result = {
            'C': Cc, 
            'P': Pc, 
            'M': Mc, 
            'orders': orders
        }
    return (yi, xi, result)
runthis = True #false if t>1 does not need to be fit
if runthis == True:
    start_time2 = time.time() 
    print(f"Processing month {m}")
    results = Parallel(n_jobs=-1)(
        delayed(fitcopulata1)(ys[i], xs[i], m, t2m_u2, pr_u2, ws_u2, t2m_key, pr_key, ws_key) for i in range(len(ys))
    )

    for yi, xi, result in results:
        globals()['C2_' + str(m)][yi, xi] = result['C']
        globals()['P2_' + str(m)][yi, xi] = result['P']
        globals()['M2_' + str(m)][yi, xi] = result['M']
        globals()['orders2_' + str(m)][yi, xi] = result['orders']
    print((time.time() - start_time2))
 
    variab = ['M2_', 'P2_', 'C2_', 'orders2_']
#save files
    for k in variab:
        C_copy  =  locals()[k + str(m)].copy()
        for j in range(len(ys)):
            if type(C_copy[ys[j], xs[j]]) == int:
                C_copy[ys[j], xs[j]] =str(C_copy[ys[j], xs[j]])
            
                
            elif type(C_copy[ys[j], xs[j]]) == np.ndarray:
                C_copy[ys[j], xs[j]] = str(C_copy[ys[j], xs[j]].tolist())

            else: C_copy[ys[j], xs[j]] =str(C_copy[ys[j], xs[j]])
        C_copy = C_copy.astype(str)  
        C_copy = xr.DataArray(
            C_copy,
            dims=("latitude", "longitude"),
            coords={"latitude": lat.values, "longitude":lon.values}
        )
        
        C_copy.to_netcdf(modelfilepath2 + 'all' +  k + str(m) + '.nc')

