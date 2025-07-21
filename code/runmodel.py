# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:24:20 2025

@author: jcl202
"""

#code to run model

import xarray as xr
import sys
import numpy as np
from vinecopulas.marginals import *
from vinecopulas.bivariate import *
from vinecopulas.vinecopula import *
import time
from itertools import combinations
#%%
names =[ 't2m','pr', 'ws']
filepath = '...../MYRIAD-SIM/data/' #path to data
modelfilepath = '..../MYRIAD-SIM/modelpar/' #path to model parameters
outputfilepath = '..../MYRIAD-SIM/outputall/' #path to output of model
varnum = 0
#%%
def loadselected(fn):
    inter = xr.open_dataset(fn).load()
    return inter
#%%
for k1 in range(varnum+1):
    fn = filepath + names[k1] + 'coarse_u2.nc'
    namei = names[k1]
    locals()[namei + '_u2'] =loadselected(fn)
    keyi = list(locals()[namei + '_u2'].keys())[0]
    locals()[namei+'_key'] = keyi
#%%
months = list(locals()[namei + '_u2']['time'].dt.month.values)
n1 =  locals()[namei + '_u2'][keyi][5].values.shape[1]
n2 = locals()[namei + '_u2'][keyi][5].values.shape[0]
mask = ~np.isnan(locals()[namei + '_u2'][keyi][5].values) #the gridcells where there are values
ys, xs = np.where(mask == True) #the x and y values where the gridcells are value
lon = locals()[namei + '_u2'].longitude
lat = locals()[namei + '_u2'].latitude
timei  = locals()[namei+'_u2'].time.values
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
        

#%% load model files
k1 = 0
variab = ['M2_', 'P2_', 'C2_', 'orders2_']
ys, xs = np.where(mask == True)
for k in variab:
    for m in range(1,13):
        Ci = xr.open_dataset(modelfilepath + 'all' +  k + str(m) + '.nc').load().__xarray_dataarray_variable__.values
        Ci = Ci.astype(object)
        Ci[Ci == 'None'] = None
        for j in range(len(ys)):
            if Ci[ys[j], xs[j]] == None:
                continue 
            elif '[' in Ci[ys[j], xs[j]]:
                Ci[ys[j], xs[j]] = eval(Ci[ys[j], xs[j]].replace('nan', 'np.nan'))
                

    # Convert the string to a Python list
                if k in  ['M', 'P', 'C', 'M2_', 'P2_', 'C2_']:
                    Ci[ ys[j], xs[j]]= np.array(Ci[ys[j], xs[j]])
            elif '.' in Ci[ys[j], xs[j]]:
                Ci[ys[j], xs[j]] = float(Ci[ys[j], xs[j]])
            else:
                Ci[ys[j], xs[j]] = int(Ci[ys[j], xs[j]])

            
        globals()[k + str(m)] = Ci.copy()  
    
        
k1 = 0
variab = ['M', 'P', 'C', 'orders']
ys, xs = np.where(mask == True)
for k in variab:
    for m in range(1,13):
    
        Ci = xr.open_dataset(modelfilepath + 'all' +  k + str(m) + '.nc').load().__xarray_dataarray_variable__.values
        Ci = Ci.astype(object)
        Ci[Ci == 'None'] = None
        for j in range(len(ys)):
            if Ci[ys[j], xs[j]] == None:
                continue 
            elif '[' in Ci[ys[j], xs[j]]:
                Ci[ys[j], xs[j]] = eval(Ci[ys[j], xs[j]].replace('nan', 'np.nan'))
                

    # Convert the string to a Python list
                if k in  ['M', 'P', 'C', 'M2_', 'P2_', 'C2_']:
                    Ci[ ys[j], xs[j]]= np.array(Ci[ys[j], xs[j]])
            elif '.' in Ci[ys[j], xs[j]]:
                Ci[ys[j], xs[j]] = float(Ci[ys[j], xs[j]])
            else:
                Ci[ys[j], xs[j]] = int(Ci[ys[j], xs[j]])

            
        globals()[k + str(m)] = Ci.copy()          
#%% set duration
n3 = len(locals()[names[0] + '_u2'][locals()[names[0]+'_key']]) #these are all time steps in original data
n3 = 3 #n3 can also be adjusted to a different length for a different duration of samples, for example 3 would compute 3 days
#%% create array per variable for simulated data
for i in names:
    locals()[i + 'img']=  np.empty((n3, n2, n1)) * np.nan
   
#%%
start_time = time.time() 
print('starting model')    
modelnums = np.unique(modelnum)
for n in range(0, n3):
    mn = months[n]
    if n == 0:
        for j in steps:
            
            yj = np.where(my_array== j)[0][0]
            xj = np.where(my_array== j)[1][0]
            
            if globals()['orders' + str(mn)][yj,xj] == 3:
                sample1 = sample_vinecop(globals()['M' + str(mn)][yj,xj] , globals()['P' + str(mn)][yj,xj] ,globals()['C' + str(mn)][yj,xj] , 1)[0]
                for i in range(len(names)):
                    
                    locals()[names[i] + 'img'][n,yj,xj] =  sample1[i]
            else:
                xc = []
                io = np.where(modelnums == modelnum[j])[0][0]
                
                for k in globals()['orders' + str(mn)][yj,xj] [:-3]:
                    if len(k) == 3:
                        xc.append(locals()[k[0] + 'img'][n,yj+k[1],xj+k[2]])
                        
                sample1 = sample_vinecopconditional(globals()['M' + str(mn)][yj,xj] , globals()['P' + str(mn)][yj,xj] ,globals()['C' + str(mn)][yj,xj] , 1, xc)[0]
                for i in range(len(names)):
                    
                    locals()[names[i] + 'img'][n,yj,xj] =  sample1[i]
                
    else:
        for j in steps:
            yj = np.where(my_array== j)[0][0]
            xj = np.where(my_array== j)[1][0]
            xc = []
            io = np.where(modelnums == modelnum[j])[0][0]
            
            for k in globals()['orders2_' + str(mn)][yj,xj] [:-3]:
                if len(k) == 3:
                    xc.append(locals()[k[0] + 'img'][n,yj+k[1],xj+k[2]])
                if len(k) == 2:
                    xc.append(locals()[k[0] + 'img'][n-1,yj,xj])
                    
            sample1 = sample_vinecopconditional(globals()['M2_' + str(mn)][yj,xj] , globals()['P2_' + str(mn)][yj,xj] ,globals()['C2_' + str(mn)][yj,xj] , 1, xc)[0]
            for i in range(len(names)):
                
                locals()[names[i] + 'img'][n,yj,xj] =  sample1[i]
print('run time in seconds: ', (time.time() - start_time))
            
#%% save data 
for namei in names:

    trial= xr.DataArray(
        locals()[namei + 'img'],
        dims=("time", "latitude", "longitude"),
        coords={"time": timei[:n3], "latitude": lat.values, "longitude": lon.values}
    )
    trial.name = 'u'
    
    trial = xr.Dataset({trial.name: trial})
    trial.to_netcdf(outputfilepath + namei + '.nc') 
          
 
    