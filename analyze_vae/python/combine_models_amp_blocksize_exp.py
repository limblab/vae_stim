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
prop_below_thresh = np.zeros((len(n_chans),len(thresh_vals),4,7,len(folders))) # n_chans, thresholds, n_freqs,n_amps, n_maps
keep_map_mask = np.ones(len(folders))
median_mag = np.zeros((len(n_chans),4,7,len(folders))) # n_chans,n_freqs, n_amps, n_maps
mag_all = np.zeros((len(n_chans),4,7,len(folders),1000))

for i_chan in range(len(n_chans)):
    for i_folder in range(len(folders)):
        print(i_folder)
        # find amp_elec_exp_ratemult10_
        pkl_file = glob.glob(folders[i_folder] + '\\amp_blocksize_exp_ratemult10_nchans' + str(n_chans[i_chan]) + '*')
        
        if(len(pkl_file)>0):
            pkl_file = pkl_file[0]
            f=open(pkl_file,'rb')
            pkl_data = pickle.load(f)
            f.close()
            # extract vars
            amp_blocksize_exp_out = pkl_data[0]
            amp = pkl_data[2]
            input_data = pkl_data[3]

            # get proportion below threshold(s) for each file
            idx_look = 0
            for i_size in range(len(input_data['block_size_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_dir_stim = amp_blocksize_exp_out[idx_look][-2]
                    pred_delta_dir = amp_blocksize_exp_out[idx_look][-1]
                    ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
                    
                    for i_thresh in range(len(thresh_vals)):
                        prop_below_thresh[i_chan,i_thresh,i_size,i_amp,i_folder] = np.sum(ang_diff<thresh_vals[i_thresh])/len(ang_diff)
                        
                    idx_look = idx_look+1
            
            
            # get magnitude for each file
            idx_look = 0
            for i_size in range(len(input_data['block_size_list'])):
                for i_amp in range(len(input_data['amp_list'])):
                    delta_mag_stim = amp_blocksize_exp_out[idx_look][1]
                    data = delta_mag_stim[:,0]
                    median_mag[i_chan,i_size,i_amp,i_folder] = np.median(data)
                    idx_look=idx_look+1
        else:
            keep_map_mask[i_folder] = 0
      
median_mag = median_mag[:,:,:,keep_map_mask==1]
prop_below_thresh = prop_below_thresh[:,:,:,:,keep_map_mask==1]
    
#%% plot prop below thresh for each threshold
            
# for each threshold, amp, n_chans: plot mean proportion and individual props connected
mean_prop = np.mean(prop_below_thresh,axis=-1)
std_prop = np.std(prop_below_thresh,axis=-1)

colors = plt.cm.inferno([180,120,60,0])

ls = ['solid','dashed']
m = ['.','s']
ms = [20,8]
offset = [-0.5,0.5]

amp_vals = np.array(input_data['amp_list'])

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
color_idx=0
line_list = []

for i_size in range(3):#len(input_data['block_size_list'])):
    for i_chan in [1]:#range(len(n_chans)):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_prop[i_chan,0,i_size,:], std_prop[i_chan,0,i_size,:], capsize=5,elinewidth=2, \
                     color=colors[color_idx], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset

    color_idx = color_idx+1

plt.ylim([0.4,1.0])
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Proportion predicted')
            
plt.legend(line_list,['33um','50um','67um'],frameon=0)
            
#%% plot magnitude across maps

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

median_mag_use = median_mag/max_hand_vel_mag*100
mean_mag = np.mean(median_mag_use,axis=-1)
std_mag = np.zeros_like(mean_mag)


amp_vals = np.array(input_data['amp_list'])
colors = plt.cm.inferno([180,120,60,0])

ls = ['solid','dashed']
m = ['.','s']
ms = [16,6]
offset = [-0.25,0.25]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
line_list = []
for i_size in range(3):#len(input_data['block_size_list'])):
    for i_chan in [1]: #range(len(n_chans)):
        # plot mean
        line = plt.errorbar(amp_vals+offset[i_chan],mean_mag[i_chan,i_size,:], std_mag[i_chan,i_size,:], capsize=0,elinewidth=0, \
                     color=colors[i_size], linestyle=ls[i_chan],marker=m[i_chan],markersize=ms[i_chan],linewidth=2)
        line_list.append(line)
        # plot each point from each map with some offset
        for i_folder in range(median_mag_use.shape[-1]):
            plt.plot(amp_vals+offset[i_chan],median_mag_use[i_chan,i_size,:,i_folder],color=colors[i_size], \
                 linestyle='none',marker=m[i_chan],markersize=ms[i_chan]/2,alpha=0.5)
    
    
    
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
plt.yticks(ticks=np.arange(0,25,5))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% high speed)')
            
plt.legend(line_list,['30um','50um','67um'],frameon=0)


