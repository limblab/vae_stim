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
prop_below_thresh = np.zeros((len(thresh_vals),2,7,len(folders))) # thresholds, n_chans, n_amps, n_maps
median_mag = np.zeros((2,7,len(folders))) # n_chans, n_amps, n_maps
mag_all = np.zeros((2,7,len(folders),1000))
ang_diff_all = np.zeros_like(mag_all)
keep_map_mask = np.zeros(len(folders))

for i_folder in range(len(folders)):
    print(i_folder)
    # find amp_elec_exp_ratemult10_
    pkl_file = glob.glob(folders[i_folder] + '\\amp_elec_exp_ratemult10_*')
    
    if(len(pkl_file)>0):
        pkl_file = pkl_file[0]
        keep_map_mask[i_folder] = 1
        f=open(pkl_file,'rb')
        pkl_data = pickle.load(f)
        f.close()
        # extract vars
        amp_elec_exp_out = pkl_data[0]
        n_chans = pkl_data[1]
        amp = pkl_data[2]
        input_data = pkl_data[3]
        
        # get proportion below threshold(s) for each file
        idx_look = 0
        for i_chan in range(len(input_data['n_stim_chans_list'])):
            for i_amp in range(len(input_data['amp_list'])):
                delta_dir_stim = amp_elec_exp_out[idx_look][-2]
                pred_delta_dir = amp_elec_exp_out[idx_look][-1]
                ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
                ang_diff_all[i_chan,i_amp,i_folder,:] = ang_diff
                for i_thresh in range(len(thresh_vals)):
                    prop_below_thresh[i_thresh,i_chan,i_amp,i_folder] = np.sum(ang_diff<thresh_vals[i_thresh])/len(ang_diff)
                    
                idx_look = idx_look+1
        
        
        # get magnitude for each file
        idx_look = 0
        for i_chan in range(len(input_data['n_stim_chans_list'])):
            for i_amp in range(len(input_data['amp_list'])):
                delta_mag_stim = amp_elec_exp_out[idx_look][1]
                data = delta_mag_stim[:,0]
                median_mag[i_chan,i_amp,i_folder] = np.median(data)
                mag_all[i_chan,i_amp,i_folder,:] = data
                idx_look=idx_look+1
    
median_mag = median_mag[:,:,keep_map_mask==1]
prop_below_thresh = prop_below_thresh[:,:,:,keep_map_mask==1]
mag_all = mag_all[:,:,keep_map_mask==1,:]
ang_diff_all = ang_diff_all[:,:,keep_map_mask==1,:]
#%% plot prop below thresh for each threshold
            
# for each threshold, amp, n_chans: plot mean proportion and individual props connected
mean_prop = np.mean(prop_below_thresh,axis=3)
#std_prop = np.std(prop_below_thresh,axis=3)
std_prop=np.zeros_like(mean_prop)

colors = plt.cm.inferno([0,170,30,200])
colors = plt.cm.inferno([170,200])

ls = ['dashed','dotted']
m = ['.','s']
ms = [20,8]
offset = [-0.5,0.5]

amp_vals = np.array(input_data['amp_list'])

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
color_idx=0
line_list = []
for i_thresh in range(len(thresh_vals)):
    for i_chan in [1]:#range(len(input_data['n_stim_chans_list'])):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_prop[i_thresh,i_chan,:], std_prop[i_thresh,i_chan,:], capsize=0,elinewidth=0, \
                     color=colors[color_idx], linestyle=ls[i_thresh],marker=m[i_thresh],markersize=ms[i_thresh],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset
        for i_folder in range(prop_below_thresh.shape[-1]):
            plt.plot(amp_vals+offset[i_chan],prop_below_thresh[i_thresh,i_chan,:,i_folder],color=colors[color_idx], \
                 linestyle='none',marker=m[i_thresh],markersize=ms[i_thresh]/2,alpha=0.5)
        color_idx = color_idx+1

plt.ylim([0.,0.85])
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predicted')
            
#plt.legend(line_list,['thresh=90deg,1chan','thresh=90deg,4chan','thresh=45deg,1chan','thresh=45deg,4chan'],frameon=0,loc='upper right')
         
#%% plot prop below thresh for different thresholds for one condition
            
new_thresh_vals = [np.pi/16,np.pi/8,np.pi/4,np.pi/2]

colors = plt.cm.inferno([170,200])

ls = ['dashed','dotted']
m = ['.','s']
ms = [20,8]
offset = [-0.5,0.5]

amp_vals = np.array(input_data['amp_list'])

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
color_idx=0
line_list = []
i_chan = 1

for i_amp in range(len(input_data['amp_list'])):
    prop_below = np.zeros((len(new_thresh_vals),ang_diff_all.shape[2]))
    for i_thresh in range(len(new_thresh_vals)):
        prop_below[i_thresh,:] = np.sum(ang_diff_all[i_chan,i_amp,:,:] < new_thresh_vals[i_thresh],axis=-1)/ang_diff_all.shape[-1]
        
    
    mean_prop_all = np.mean(prop_below,axis=-1)
    std_prop_all = np.std(prop_below,axis=-1)
    # plot mean
    line = plt.errorbar(thresh_vals,mean_prop_all[:], std_prop_all[:], capsize=0,elinewidth=0, \
                 )
    #%%
    line_list.append(line)
    # plot each point from each map with some offset
    for i_folder in range(mean_prop_map.shape[-1]):
        plt.plot(amp_vals+offset[i_chan],mean_prop_map[:,i_folder],alpha=0.5)
    color_idx = color_idx+1


#%%
plt.ylim([0.,0.85])
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predicted')

   
#%% plot magnitude across maps


act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

median_mag_use = median_mag/max_hand_vel_mag*100
mean_mag = np.mean(median_mag_use,axis=-1)
std_mag = np.zeros_like(mean_mag)



amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,170])

ls = ['dashed']
m = ['.','s']
ms = [20]
offset = [-0.25,0.25]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []
for i_chan in range(len(input_data['n_stim_chans_list'])):
    # plot mean
    line = plt.errorbar(amp_vals+offset[i_chan],mean_mag[i_chan,:], std_mag[i_chan,:], capsize=0,elinewidth=0, \
                 color=colors[i_chan], linestyle=ls[0],marker=m[0],markersize=ms[0],linewidth=2)
    line_list.append(line)
    # plot each point from each map with some offset
    for i_folder in range(prop_below_thresh.shape[-1]):
        plt.plot(amp_vals+offset[i_chan],median_mag_use[i_chan,:,i_folder],color=colors[i_chan], \
             linestyle='none',marker=m[0],markersize=ms[0]/2,alpha=0.5)
    
    
    
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% high speed)')
            
plt.legend(line_list,['1chan','4chan'],frameon=0)


#%%
ang_thresh = np.pi/2;
idx_base = 0

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mag_use = mag_all/max_hand_vel_mag*100
mag_base = mag_use[:,0,:,:].reshape(-1,1) # n_chans, n_amps, n_maps, n_data
min_mag = np.percentile(mag_base,95)

prop_above_min_mag = np.zeros_like(prop_below_thresh)
prop_predicted = np.zeros_like(prop_above_min_mag)

amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,60,120,180])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_chan in [1]:#range(len(n_chans)):
    is_large = mag_use[i_chan,:,:,:] > min_mag
    is_directed = ang_diff_all[i_chan,:,:,:] < ang_thresh

    prop_above_min_mag[i_chan,0,:,:] = np.sum(is_large,axis=-1)/is_large.shape[-1]
    prop_predicted[i_chan,0,:,:] = np.sum(np.logical_and(is_large,is_directed),axis=-1)/is_large.shape[-1]
    
    mean_pred = np.mean(prop_predicted,axis=-1)
    std_pred = np.std(prop_predicted,axis=-1)
    mean_above_min = np.mean(prop_above_min_mag,axis=-1)
    
    line = plt.errorbar(amp_vals+offset[i_chan],mean_pred[i_chan,0,:], std_pred[i_chan,0,:], capsize=0,elinewidth=0, \
                 color=colors[i_chan], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
    line_list.append(line)
    plt.plot(amp_vals+offset[i_chan],mean_above_min[i_chan,0,:]*0.5, \
             color=colors[i_chan], linestyle=ls[i_chan],marker='',alpha=0.5,markersize=ms[i_chan],linewidth=2)
        
        
        
plt.xlim([-5,85])
plt.ylim([0,0.65])
plt.xticks(ticks=np.arange(0,85,10))
plt.yticks(ticks=np.arange(0,0.7,0.1))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predictable and large')
            
plt.legend(line_list,['50 Hz','100 Hz','200 Hz'],frameon=0,loc='lower right')
