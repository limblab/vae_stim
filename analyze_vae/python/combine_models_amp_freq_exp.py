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

n_chans=[1,4]
# init vars
thresh_vals = [np.pi/2]
prop_below_thresh = np.zeros((len(n_chans),len(thresh_vals),3,7,len(folders))) # n_chans, thresholds, n_freqs,n_amps, n_maps
keep_map_mask = np.zeros(len(folders))
median_mag = np.zeros((len(n_chans),3,7,len(folders))) # n_chans,n_freqs, n_amps, n_maps
mag_all = np.zeros((len(n_chans),3,7,len(folders),1000)) # n_chans, n_freqs, n_amps, n_maps, n_data
ang_diff_all = np.zeros_like(mag_all)

for i_chan in range(len(n_chans)):
    for i_folder in range(len(folders)):
        print(i_folder)
        # find amp_elec_exp_ratemult10_
        pkl_file = glob.glob(folders[i_folder] + '\\amp_freq_exp_ratemult10_nchans' + str(n_chans[i_chan]) + '*')
        
        if(len(pkl_file)>0):
            pkl_file = pkl_file[0]
            keep_map_mask[i_folder] = 1
            f=open(pkl_file,'rb')
            pkl_data = pickle.load(f)
            f.close()
            # extract vars
            amp_freq_exp_out = pkl_data[0]
            freq = pkl_data[1]
            amp = pkl_data[2]
            input_data = pkl_data[3]
            
            # get proportion below threshold(s) for each file
            idx_look = 0
            for i_freq in range(len(input_data['freq_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_dir_stim = amp_freq_exp_out[idx_look][-2]
                    pred_delta_dir = amp_freq_exp_out[idx_look][-1]
                    ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
                    ang_diff_all[i_chan,i_freq,i_amp,i_folder,:] = ang_diff
                    for i_thresh in range(len(thresh_vals)):
                        prop_below_thresh[i_chan,i_thresh,i_freq,i_amp,i_folder] = np.sum(ang_diff<thresh_vals[i_thresh])/len(ang_diff)
                        
                    idx_look = idx_look+1
            
            
            # get magnitude for each file
            idx_look = 0
            for i_freq in range(len(input_data['freq_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_mag_stim = amp_freq_exp_out[idx_look][1]
                    data = delta_mag_stim[:,0]
                    median_mag[i_chan,i_freq,i_amp,i_folder] = np.median(data)
                    mag_all[i_chan,i_freq,i_amp,i_folder,:] = data
                    idx_look=idx_look+1

      
median_mag = median_mag[:,:,:,keep_map_mask==1]
prop_below_thresh = prop_below_thresh[:,:,:,:,keep_map_mask==1]
mag_all = mag_all[:,:,:,keep_map_mask==1,:]
ang_diff_all = ang_diff_all[:,:,:,keep_map_mask==1,:]

#%% plot prop below thresh for each threshold
            
# for each threshold, amp, n_chans: plot mean proportion and individual props connected
mean_prop = np.mean(prop_below_thresh,axis=-1)
std_prop = np.std(prop_below_thresh,axis=-1)

colors = plt.cm.inferno([0,60,120,180])

ls = ['solid','dashed']
m = ['.','s']
ms = [20,8]
offset = [-0.5,0.5]

amp_vals = np.array(input_data['amp_list'])

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
color_idx=0
line_list = []

for i_freq in range(len(input_data['freq_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_prop[i_chan,0,i_freq,:], std_prop[i_chan,0,i_freq,:], capsize=5,elinewidth=2, \
                     color=colors[color_idx], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset

    color_idx = color_idx+1

plt.ylim([0.45,0.7])
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,75,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion Actual - Predicted < 90 deg')
            
#plt.legend(line_list,['1chan,50Hz','4chan,50Hz','1chan,100Hz','4chan,100Hz','1chan,200Hz','4chan,200Hz','1chan,400Hz','4chan,400Hz'],frameon=0)
plt.legend(line_list,['50 Hz','100 Hz','200 Hz'],loc='upper right',frameon=0)
  

#%% plot magnitude across maps

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mag_use = 100*mag_all.reshape((mag_all.shape[0],mag_all.shape[1],mag_all.shape[2],-1))#/max_hand_vel_mag*100
mean_mag = np.mean(mag_use,axis=-1)
std_mag = np.std(mag_use,axis=-1)


amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,60,120,180])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []
for i_freq in range(len(input_data['freq_list'])):
    for i_chan in range(len(n_chans)):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_mag[i_chan,i_freq,:], std_mag[i_chan,i_freq,:], capsize=0,elinewidth=0, \
                     color=colors[i_freq], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset
        #for i_folder in range(median_mag_use.shape[-1]):
        #    plt.plot(amp_vals+offset[i_chan],median_mag_use[i_chan,i_freq,:,i_folder],color=colors[i_freq], \
        #         linestyle='none',marker=m[i_chan],markersize=ms[i_chan]/2,alpha=0.5)
    
    
    
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
#plt.yticks(ticks=np.arange(0,15,5))
ax.xaxis.set_minor_locator(MultipleLocator(5))
#ax.yaxis.set_minor_locator(MultipleLocator(1))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Length of difference vector \n(cm/s)')
            
plt.legend(line_list,['1chan,50Hz','4chan,50Hz','1chan,100Hz','4chan,100Hz','1chan,200Hz','4chan,200Hz','1chan,400Hz','4chan,400Hz'],frameon=0,loc='upper left')
#plt.legend(line_list,['50 Hz','100 Hz','200 Hz'],loc='upper left',frameon=0)

#%% plot angular error and proportion of trials
amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,40,80,120,160,200,240])

bin_edges = np.arange(0,190,20)
bin_centers = bin_edges[0:-1] + np.mean(np.diff(bin_edges))/2

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_freq in [2]:#range(len(input_data['freq_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        for i_amp in [0,1,2,3,4,-1]:#range(len(input_data['amp_list'])):
            error_all = ang_diff_all[i_chan,i_freq,i_amp,:,:]*180/np.pi
            error_all = error_all.reshape(-1,1)
            hist_vals = np.histogram(error_all,bins=bin_edges)[0]/len(error_all)
            
            
            line = plt.plot(bin_centers,hist_vals, \
                         color=colors[i_amp], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)[0]
            line_list.append(line)

plt.ylim([0,0.2])
sns.despine()

plt.xlabel('Actual - Predicted (deg)')     
plt.ylabel('Proportion of trials')
            
plt.legend(line_list,['0 uA','5 uA','10 uA','15 uA','20 uA','80 uA'],frameon=0)


#%% plot mean and std. of angular error

amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,60,120,180])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_freq in range(len(input_data['freq_list'])):
    for i_chan in range(len(n_chans)):
        error_all = ang_diff_all[i_chan,i_freq,:,:,:]
        error_all = error_all*180/np.pi
        mean_error = np.mean(np.mean(error_all,axis=-1),axis=-1)
        std_error = np.std(np.mean(error_all,axis=-1),axis=-1)
        
        line = plt.errorbar(amp_vals+offset[i_freq],mean_error, std_error, capsize=5,elinewidth=2, \
                     color=colors[i_freq], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        #line_list.append(line)

plt.ylim([70,100])
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')
            
#plt.legend(line_list,['50 Hz','100 Hz','200 Hz'],loc='upper left',frameon=0)

#%% plot mean projection onto PD
amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,60,120,180])

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)
mag_use = mag_all/max_hand_vel_mag*100

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_freq in [2]:#range(len(input_data['freq_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        error_all = ang_diff_all[i_chan,i_freq,:,1,:]
        error_all = error_all.reshape(error_all.shape[0],-1)
        mag = mag_use[i_chan,i_freq,:,1,:]
        
        proj_all = np.multiply(mag,np.cos(error_all))
        
        mean_error = np.mean(proj_all,axis=-1)
        std_error = np.std(proj_all,axis=-1)
        
        line = plt.errorbar(amp_vals+offset[i_freq],mean_error, std_error, capsize=0,elinewidth=0, \
                     color=colors[i_freq], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)

sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Mean projection onto PD (cm/s)')
            
plt.legend(line_list,['200 Hz'],loc='upper left',frameon=0)



#%% plot proportion above minimum magnitude

ang_thresh = np.pi/2;
idx_base = 0

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mag_use = mag_all/max_hand_vel_mag*100
mag_base = mag_use[:,:,0,:,:].reshape(-1,1) # n_chans, n_freqs, n_amps, n_maps, n_data
min_mag = np.percentile(mag_base,95)

prop_above_min_mag = np.zeros_like(prop_below_thresh)
prop_predicted = np.zeros_like(prop_above_min_mag)
sanity_check = np.zeros_like(prop_above_min_mag)

amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,60,120,180])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_freq in range(len(input_data['freq_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        is_large = mag_use[i_chan,i_freq,:,:,:] > min_mag
        is_directed = ang_diff_all[i_chan,i_freq,:,:,:] < ang_thresh

        prop_above_min_mag[i_chan,0,i_freq,:,:] = np.sum(is_large,axis=-1)/is_large.shape[-1]
        prop_predicted[i_chan,0,i_freq,:,:] = np.sum(np.logical_and(is_large,is_directed),axis=-1)/is_large.shape[-1]
        
        mean_pred = np.mean(prop_predicted,axis=-1)
        std_pred = np.std(prop_predicted,axis=-1)
        mean_above_min = np.mean(prop_above_min_mag,axis=-1)
        
        line = plt.errorbar(amp_vals+offset[i_chan],mean_pred[i_chan,0,i_freq,:], std_pred[i_chan,0,i_freq,:], capsize=0,elinewidth=0, \
                     color=colors[i_freq], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        plt.plot(amp_vals+offset[i_chan],mean_above_min[i_chan,0,i_freq,:]*0.5, \
                 color=colors[i_freq], linestyle=ls[i_chan],marker='',alpha=0.5,markersize=ms[i_chan],linewidth=2)
        
        
        
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


