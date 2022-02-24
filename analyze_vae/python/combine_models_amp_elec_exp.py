# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:03:41 2021

@author: Joseph Sombeck
"""

import global_vars as gl # import global variables first. I'm not ecstatic about this approach but oh well
gl.global_vars()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

import vae_utils 
import vae_model_code as mdl
import glob
import stim_exp_utils
import osim_utils
import scipy as sp

from sklearn.metrics.pairwise import cosine_similarity

import pickle
import datetime
import torch
%matplotlib qt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# load in hand velocity data
training_data_folder = r'D:\Lab\Data\StimModel\training_data'
path_to_data_mask = glob.glob(training_data_folder + '\\*TrialMask_uniform*')[0]
trial_mask = np.genfromtxt(path_to_data_mask,delimiter=',')[:]
path_to_hand_vels = glob.glob(training_data_folder + '\\*RawHandVel_uniform*')[0]
hand_vels = np.genfromtxt(path_to_hand_vels,delimiter=',')[trial_mask==1,0:2]


#%% set global plot style
sns.set_style('ticks') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('font', size=14)          # controls default text sizes


#%% get data

fpath = r'D:\Lab\Data\StimModel\models'

folders = glob.glob(fpath + '\\Han_*')

# init vars
thresh_vals = [np.pi/2,np.pi/4]
# load in a base input data
pkl_file = glob.glob(folders[0] + '\\multi_loc_repetition_few_conditions_exp*')
f=open(pkl_file[0],'rb')  
temp = pickle.load(f)
f.close()

input_data = temp[3]
        
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

mean_ang_all = np.zeros((len(folders),input_data['n_sets'],len(input_data['amp_list'])))
mean_mag_all = np.zeros_like(mean_ang_all)
std_ang_all = np.zeros_like(mean_ang_all)
keep_mask = np.zeros((len(folders),))

for i_folder in range(len(folders)):
    print(i_folder)
    # find amp_elec_exp_ratemult10_
    pkl_file = glob.glob(folders[i_folder] + '\\multi_loc_repetition_few_conditions_exp*')
    hash_pd_file = glob.glob(folders[i_folder] + '\\PD_multiunit_hash_pow2*')
    if(len(pkl_file)>0):
        #load data 
        idx_look = 0
        keep_mask[i_folder] = 1
        f=open(pkl_file[0],'rb')  
        temp = pickle.load(f)
        f.close()
    
        single_loc_exp = temp[0]
        loc_idx = temp[1]
        amp = temp[2]
        input_data = temp[3]
        stim_chan_list = temp[4]
        
        # load hash PDs
        f=open(hash_pd_file[0],'rb')
        temp = pickle.load(f)
        hash_PDs = temp[0]
        f.close()
    
        for i_set in range(input_data['n_sets']):
            for i_amp in range(len(input_data['amp_list'])):
                # angular error data
                delta_dir_stim = single_loc_exp[idx_look][-2]
                #pred_delta_dir = single_loc_exp[idx_look][-1]
                stim_chan_pds = hash_PDs[stim_chan_list[idx_look].astype(int)]
                pred_delta_dir = sp.stats.circmean(stim_chan_pds,axis=1,high=np.pi,low=-np.pi)
                
                error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
                mean_ang_all[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
                std_ang_all[i_folder,i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
            
                
                # magnitude data
                delta_mag_stim = single_loc_exp[idx_look][1]
                data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
                mean_mag_all[i_folder,i_set,i_amp] = np.mean(data)
                
                idx_look = idx_look + 1
                    
mean_ang_all = mean_ang_all[keep_mask==1,:,:]                    
mean_mag_all = mean_mag_all[keep_mask==1,:,:]
std_ang_all = std_ang_all[keep_mask==1,:,:]
  
# the map I used for every other experiment               
used_map_idx = 1
   
#%% magnitude of effect, error bars across trials of the same location
colors = plt.cm.inferno([0,90,180])
offset = [-1,-0.5,0,0.5,1]*1

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.4,0.6])

# plot mean and error bars across all maps
mean_mag_map = np.mean(mean_mag_all,axis=1)

ax.errorbar(np.array(input_data['amp_list']),np.mean(mean_mag_map,axis=0), np.std(mean_mag_map,axis=0), capsize=5,elinewidth=2, \
                     color=colors[0],linewidth=0,marker='o',markersize=10,markerfacecolor='none',markeredgewidth=3)

# plot mean for each map
for i_map in range(mean_mag_all.shape[0]):
    if(i_map==1): # map I used for other experiments
        colors = '#e41a1c'
        markersize=8
        marker='s'
    elif(i_map==27): # an example map in the figure
        colors = '#377eb8'
        markersize=10
        marker='v'
    elif(i_map==26): # an example map in the figure
        colors = '#65463E'
        markersize=10
        marker='d'
    else:
        colors = 'black'
        markersize=8
        marker='.'
        
    mean_to_plot = np.mean(mean_mag_all[i_map,:,:],axis=0) # mean over sets
    offset = np.random.choice([-1,1])*(np.random.rand()*1.0 + 0.75)
    ax.plot(np.array(input_data['amp_list'])+offset,mean_to_plot, \
                     color=colors,linewidth=0,marker=marker,markersize=markersize,alpha=0.5)

plt.xlim([-2.5,22.5])
plt.ylim([0,4.5])
plt.xticks(ticks=np.arange(0,30,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')


#%% angular error
colors = plt.cm.inferno([0,90,180])
offset = [-1,-0.5,0,0.5,1]*1

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.4,0.6])

# plot mean and error bars across all maps
mean_ang_map = np.mean(mean_ang_all,axis=1)

ax.errorbar(np.array(input_data['amp_list']),np.mean(mean_ang_map,axis=0), np.std(mean_ang_map,axis=0), capsize=6,elinewidth=2, \
                     color=colors[0],linewidth=0,marker='o',markersize=10,markerfacecolor='none',markeredgewidth=3)

# plot mean for each map
for i_map in range(mean_ang_map.shape[0]):
    if(i_map==1): # map I used for other experiments
        colors = '#e41a1c'
        markersize=8
        marker='s'
    elif(i_map==27): # an example map in the figure
        colors = '#377eb8'
        markersize=10
        marker='v'
    elif(i_map==26): # an example map in the figure
        colors = '#65463E'
        markersize=10
        marker='d'
    else:
        colors = 'black'
        markersize=8
        marker='.'
        marker='.'
        
    mean_to_plot = np.mean(mean_ang_all[i_map,:,:],axis=0) # mean over sets
    offset = np.random.choice([-1,1])*(np.random.rand()*1.0 + 0.75)
    ax.plot(np.array(input_data['amp_list'])+offset,mean_to_plot, \
                     color=colors,linewidth=0,linestyle='none',marker=marker,markersize=markersize,alpha=0.5)
    
    
plt.xlim([-3,23])
plt.ylim([40,95])
plt.xticks(ticks=np.arange(0,30,10))
plt.yticks(ticks=np.arange(40,100,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')
  


#%% does angular error correlate with bimodalness of the PD distribution?

#%% get data

fpath = r'D:\Lab\Data\StimModel\models'

folders = glob.glob(fpath + '\\Han_*')

PD_hist=np.array([])
non_uniform_meas=np.array([])
chi_square=np.array([])
for i_folder in range(len(folders)):
    # find amp_elec_exp_ratemult10_
    pkl_file = glob.glob(folders[i_folder] + '\\PD_distribution_uniformity_multiUnitPDs_v4*')
    
    if(len(pkl_file)>0):
        #load data 
        
        f=open(pkl_file[0],'rb')  
        temp = pickle.load(f)
        f.close()
    
        PD_hist = np.append(PD_hist,temp[0])
        non_uniform_meas = np.append(non_uniform_meas,temp[1])
        chi_square = np.append(chi_square, temp[2])
        
PD_hist = PD_hist.reshape((len(folders),-1))
        
        

mean_map = np.zeros((len(folders),mean_ang_all.shape[-1]))
for i_map in range(mean_ang_all.shape[0]):
    mean_map[i_map,:] = np.mean(mean_ang_all[i_map,:,:],axis=0) # mean over sets


fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.4,0.6])

for i_map in range(len(chi_square)):
    if(i_map==1): # map I used for other experiments
        colors = '#e41a1c'
        markersize=8
        marker='s'
    elif(i_map==27): # an example map in the figure
        colors = '#377eb8'
        markersize=10
        marker='v'
    elif(i_map==26): # an example map in the figure
        colors = '#65463E'
        markersize=10
        marker='d'
    else:
        colors = 'black'
        markersize=8
        marker='.'
    plt.plot(chi_square[i_map],mean_map[i_map,1],color=colors,markersize=markersize,marker=marker,alpha=0.5)

reg=LinearRegression()
reg.fit(chi_square.reshape(-1,1),mean_map[:,1].reshape(-1,1))

X = np.arange(0,1.2,0.05).reshape(-1,1)
Y = reg.predict(X)

plt.plot(X,Y,linestyle='--',linewidth=2,color='black')

sns.despine()
plt.ylim([40,95])
ax.yaxis.set_minor_locator(MultipleLocator(2.5))

plt.ylabel('Mean Angular Error (deg)')
plt.xlabel('Non-Uniformity Measure')
#%%       


X = chi_square
Y = mean_map[:,1]
X2 = sm.add_constant(X)
est = sm.OLS(Y,X2)
est2 = est.fit()

print(est2.summary())















