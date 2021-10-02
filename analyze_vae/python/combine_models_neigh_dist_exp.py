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
bin_edges = np.arange(0,190,10)/180*np.pi
bin_centers = bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2
neigh_dist_list= [3.0] # in blocksize, to convert to um, do 150/neigh_dist <- neigh_dist represents the recording distance of 1 electrode

neigh_bin_counts = np.zeros((len(neigh_dist_list),len(bin_edges)-1,len(folders)))
non_neigh_bin_counts = np.zeros_like(neigh_bin_counts)
keep_map_mask = np.zeros(len(folders))


for i_folder in [0,1,2]:#range(len(folders)):
    print(i_folder)
    # find amp_elec_exp_ratemult10_
    pkl_file = glob.glob(folders[i_folder] + '\\neigh_dist_*')
    if(len(pkl_file)>0):
        pkl_file = pkl_file[0]
        keep_map_mask[i_folder] = 1
        f=open(pkl_file,'rb')
        pkl_data = pickle.load(f)
        f.close()
        # extract vars
        pd_diff = pkl_data[0]
        unit_dist = pkl_data[1]
        for i_dist in range(len(neigh_dist_list)):
    
            # separate into neighbor and non-neighbor based on distance
            is_neigh = unit_dist <= neigh_dist_list[i_dist]
            is_same_unit = unit_dist == 0
            neigh_pd_diff=pd_diff[np.logical_and(is_neigh==1,is_same_unit==0)]
            non_neigh_pd_diff=pd_diff[np.logical_and(is_neigh==0,is_same_unit==0)]
    
            # bin, store bin counts
            neigh_bin_counts[i_dist,:,i_folder] = np.histogram(neigh_pd_diff,bins=bin_edges)[0]
            non_neigh_bin_counts[i_dist,:,i_folder] = np.histogram(non_neigh_pd_diff,bins=bin_edges)[0]
        
neigh_bin_counts = neigh_bin_counts[:,:,keep_map_mask==1]
non_neigh_bin_counts = non_neigh_bin_counts[:,:,keep_map_mask==1]    
    
#%% plot neighbor PD_dist distribution for each neighbor distance (in blocks)
            
# for each threshold, amp, n_chans: plot mean proportion and individual props connected
num_units = np.expand_dims(np.sum(neigh_bin_counts,axis=1),axis=1)
norm_neigh_bin_counts = neigh_bin_counts/np.repeat(num_units,neigh_bin_counts.shape[1],axis=1)*100
        
mean_neigh_counts = np.mean(norm_neigh_bin_counts,axis=-1)
#std_prop = np.std(prop_below_thresh,axis=3)
std_prop=np.std(norm_neigh_bin_counts,axis=-1)

colors = plt.cm.inferno([0,60,120,180])

offset = np.array([-1,-0.6,-0.2,0.2,0.6,1.0])*0
ls = ['dashed','dotted']
m = ['.','s']
ms = [18,8]

plt.figure(figsize=(5,5),tight_layout=True)
ax = plt.axes()
color_idx=0
line_list = []
for i_dist in range(len(neigh_dist_list)):
    # plot mean
    line = plt.errorbar(bin_centers*180/np.pi + offset[i_dist],mean_neigh_counts[i_dist,:], std_prop[i_dist,:], capsize=6,elinewidth=2, \
                 color=colors[i_dist], linestyle='none',marker=m[0],markersize=ms[0],linewidth=1)
    line_list.append(line)
    # plot each point from each map with some offset
    #for i_folder in range(len(folders)):
    #    plt.plot(amp_vals+offset[i_chan],prop_below_thresh[i_thresh,i_chan,:,i_folder],color=colors[color_idx], \
    #         linestyle='none',marker=m[i_thresh],markersize=ms[i_thresh]/2,alpha=0.5)


plt.xlim([0,180])
plt.xticks(ticks=np.arange(0,185,30))
ax.xaxis.set_minor_locator(MultipleLocator(15))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
sns.despine()

plt.xlabel('Difference in PD (degrees)')     
plt.ylabel('Percentage')
            
plt.legend(line_list,['4000 \u03BCm'],frameon=0)
            
