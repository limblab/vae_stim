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
prop_below_thresh = np.zeros((len(n_chans),len(thresh_vals),5,6,len(folders))) #  n_chans, n_thresholds, n_steps, n_amps, n_maps
keep_map_mask = np.ones(len(folders))
median_mag = np.zeros((len(n_chans),5,6,len(folders))) # n_chans, n_steps, n_amps, n_maps
mag_all = np.zeros((len(n_chans),5,6,len(folders),500))
ang_diff_all = np.zeros_like(mag_all)

for i_chan in range(len(n_chans)):
    for i_folder in range(len(folders)):
        print(i_folder)
        # find amp_elec_exp_ratemult10_
        pkl_file = glob.glob(folders[i_folder] + '\\amp_highsim_exp_ratemult10_nchans' + str(n_chans[i_chan]) + '*')
        
        if(len(pkl_file)>0):
            pkl_file = pkl_file[0]
            f=open(pkl_file,'rb')
            pkl_data = pickle.load(f)
            f.close()
            # extract vars
            amp_highsim_exp_out = pkl_data[0]
            amp = pkl_data[1]
            step = pkl_data[2]
            stim_chan_list = pkl_data[3]
            neigh_sim = pkl_data[4]
            input_data = pkl_data[5]
            
            # get proportion below threshold(s) for each file
            idx_look = 0
            for i_step in range(len(input_data['step_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_dir_stim = amp_highsim_exp_out[idx_look][-2]
                    pred_delta_dir = amp_highsim_exp_out[idx_look][-1]
                    ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
                    ang_diff_all[i_chan,i_step,i_amp,i_folder,:]= ang_diff
                    for i_thresh in range(len(thresh_vals)):
                        prop_below_thresh[i_chan,i_thresh,i_step,i_amp,i_folder] = np.sum(ang_diff<thresh_vals[i_thresh])/len(ang_diff)
                        
                    idx_look = idx_look+1
            
            
            # get magnitude for each file
            idx_look = 0
            for i_step in range(len(input_data['step_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_mag_stim = amp_highsim_exp_out[idx_look][1]
                    data = delta_mag_stim[:,0]
                    mag_all[i_chan,i_step,i_amp,i_folder,:] = data
                    median_mag[i_chan,i_step,i_amp,i_folder] = np.median(data)
                    idx_look=idx_look+1
        else:
            keep_map_mask[i_folder] = 0
      
median_mag = median_mag[:,:,:,keep_map_mask==1]
prop_below_thresh = prop_below_thresh[:,:,:,:,keep_map_mask==1]
ang_diff_all = ang_diff_all[:,:,:,keep_map_mask==1,:]
mag_all = mag_all[:,:,:,keep_map_mask==1,:]
#%% plot prop below thresh for each threshold
            
# for each threshold, amp, n_chans: plot mean proportion and individual props connected
mean_prop = np.mean(prop_below_thresh,axis=-1)
std_prop = np.std(prop_below_thresh,axis=-1)

colors = plt.cm.inferno([0,50,100,150,200])

ls = ['solid','dashed']
m = ['.','s']
ms = [20,8]
offset = [-0.5,0.5]

amp_vals = np.array(input_data['amp_list'])

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_step in range(len(input_data['step_list'])):
    for i_chan in [1]:#   ONLY PLOT 4CHAN    range(len(n_chans)):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_prop[i_chan,0,i_step,:], std_prop[i_chan,0,i_step,:], capsize=5,elinewidth=2, \
                     color=colors[i_step], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset


plt.ylim([0.,0.9])
plt.xlim([-5,45])
plt.xticks(ticks=np.arange(0,45,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predicted')
            
#plt.legend(line_list,['1chan,0step','4chan,0step','1chan,1step','4chan,1step','1chan,2step','4chan,2step','1chan,4step','4chan,4step','1chan,8step','4chan,8step'],frameon=0)
plt.legend(line_list,['0step','1step','2step','4step','8step'],frameon=0)
     
#%%  require min distance
ang_thresh = np.pi/2;
idx_base = 0

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mag_use = mag_all/max_hand_vel_mag*100
mag_base = mag_use[:,:,0,:,:].reshape(-1,1) # n_chans, n_steps, n_amps, n_maps, n_data
min_mag = np.percentile(mag_base,95)

prop_above_min_mag = np.zeros_like(prop_below_thresh)
prop_predicted = np.zeros_like(prop_above_min_mag)

amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([0,50,100,150,200])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.5,0.5]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []

for i_step in range(len(input_data['step_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        is_large = mag_use[i_chan,i_step,:,:,:] > min_mag
        is_directed = ang_diff_all[i_chan,i_step,:,:,:] < ang_thresh

        prop_above_min_mag[i_chan,0,i_step,:,:] = np.sum(is_large,axis=-1)/is_large.shape[-1]
        prop_predicted[i_chan,0,i_step,:,:] = np.sum(np.logical_and(is_large,is_directed),axis=-1)/is_large.shape[-1]
        
        mean_pred = np.mean(prop_predicted,axis=-1)
        std_pred = np.std(prop_predicted,axis=-1)
        mean_above_min = np.mean(prop_above_min_mag,axis=-1)
        
        line = plt.errorbar(amp_vals+offset[i_chan],mean_pred[i_chan,0,i_step,:], std_pred[i_chan,0,i_step,:], capsize=5,elinewidth=2, \
                     color=colors[i_step], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        plt.plot(amp_vals+offset[i_chan],mean_above_min[i_chan,0,i_step,:]*0.5, \
                 color=colors[i_step], linestyle=ls[i_chan],marker='',alpha=0.25,markersize=ms[i_chan],linewidth=2)
        
        
        
plt.xlim([-5,85])
plt.ylim([0,0.9])
plt.xticks(ticks=np.arange(0,85,10))
plt.yticks(ticks=np.arange(0,1,0.1))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predictable and large')
            
plt.legend(line_list,['0step','1step','2step','4step','8step'],frameon=0)

