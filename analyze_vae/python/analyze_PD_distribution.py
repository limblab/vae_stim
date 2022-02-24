# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:05:22 2021

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

PD_hist=np.array([])
non_uniform_meas=np.array([])
for i_folder in range(len(folders)):
    # find amp_elec_exp_ratemult10_
    pkl_file = glob.glob(folders[i_folder] + '\\PD_distribution_uniformity*')
    
    if(len(pkl_file)>0):
        #load data 
        
        f=open(pkl_file[0],'rb')  
        temp = pickle.load(f)
        f.close()
    
        PD_hist = np.append(PD_hist,temp[0])
        non_uniform_meas = np.append(non_uniform_meas,temp[1])
    plt.figure()
    plt.plot(temp[0])
           
        
PD_hist = PD_hist.reshape((len(folders),-1))
        
        
        
        
#%%   
     

   
        
        