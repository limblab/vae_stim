# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:22:30 2021

@author: Joseph Sombeck
"""

import global_vars as gl # import global variables first. I'm not ecstatic about this approach but oh well
gl.global_vars()

import numpy as np
import matplotlib.pyplot as plt

import vae_utils 
import vae_model_code as mdl
import glob
import stim_exp_utils
import osim_utils
import scipy as sp

from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import cmasher

import pickle
import datetime
import torch
%matplotlib qt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#%% set global plot style
sns.set_style('ticks') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('font', size=14)          # controls default text sizes

#%% load in pretrained model and joint velocity data, both normalized and raw
# get project folder, file with vae_params and training parameters
project_folder = r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801'
training_data_folder = r'D:\Lab\Data\StimModel\training_data'
path_to_model_dict = glob.glob(project_folder + r'\*model_params*')[0]
path_to_model_yaml = glob.glob(project_folder + r'\*.yaml')[0]
path_to_norm_joint_vels = glob.glob(training_data_folder + r'\*NormalizedJointVel_uniform*.txt')[0]
path_to_hand_vels = glob.glob(training_data_folder + '\\*RawHandVel_uniform*')[0]
path_to_data_mask = glob.glob(training_data_folder + '\\*TrialMask_uniform*')[0]
path_to_norm_muscle_vel = glob.glob(training_data_folder + '\\*NormalizedMuscleVel_uniform*')[0]
path_to_joint_angs = glob.glob(training_data_folder + '\\*JointAng_uniform*')[0]
path_to_joint_vels = glob.glob(training_data_folder + r'\*RawJointVel_uniform*.txt')[0]
path_to_norm_hand_vels = glob.glob(training_data_folder + r'\\*RawHandVel_uniform*')[0]

# get data, only use times when trial mask is 1
trial_mask = np.genfromtxt(path_to_data_mask,delimiter=',')[:]
joint_vels_norm = np.genfromtxt(path_to_norm_joint_vels, delimiter=',')[trial_mask==1,:]
joint_vels = np.genfromtxt(path_to_joint_vels,delimiter=',')[trial_mask==1,:]
joint_angs = np.genfromtxt(path_to_joint_angs,delimiter=',')[trial_mask==1,:]
hand_vels = np.genfromtxt(path_to_hand_vels,delimiter=',')[trial_mask==1,0:2]
muscle_vels_norm = np.genfromtxt(path_to_norm_muscle_vel,delimiter=',')[trial_mask==1,:]
hand_vels_norm = np.genfromtxt(path_to_norm_hand_vels,delimiter=',')[trial_mask==1,:]

# load parameter file. This is currently convoluted, but works
gl.Params.load_params(gl.params,path_to_model_yaml)

# set params['cuda'] to false since my computer doesn't have a GPU ( :( )
gl.params.params['cuda']=False

kin_var_norm = joint_vels_norm

# load in vae weights
vae = vae_utils.load_vae_parameters(fpath=path_to_model_dict,input_size=kin_var_norm.shape[1]) 


#%% train linear decoder for this network from kin_var_norm to actual hand_vels
#%% either train new decoder or load one in

path_to_dec = glob.glob(project_folder + '\\*hand_vel_dec*')
if(len(path_to_dec)>0):
    f=open(path_to_dec[0],'rb')
    dec = pickle.load(f)
    f.close()
else:
    dec = vae_utils.make_linear_decoder(x=kin_var_norm, y=hand_vels, drop_rate=0.,n_iters=500,lr=0.01)

    f=open(project_folder + '\\hand_vel_dec.pkl','wb')
    pickle.dump(dec,f)
    f.close()
 
# load a vae_decoder if we retrained it
path_to_dec = glob.glob(project_folder + '\\*retrained_vae_dec*')
if(len(path_to_dec)>0):
    f=open(path_to_dec[0],'rb')
    vae_dec = pickle.load(f)
    f.close()
    print("loaded retrained vae dec; use input_data['vae_dec']=vae_dec")
    
    # evaluate decoder by plotting joint velocities against actual joint vel
    rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)
    kin_var_hat = vae_utils.linear_dec_forward(dec=vae_dec,x=rates)
    hand_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=kin_var_hat)
    hand_vaf = []
    
    for hand in range(hand_vels.shape[1]):
        hand_vaf.append(mdl.vaf(hand_vels[:,hand],hand_vels_hat[:,hand]))
    print(hand_vaf)



#%% get PD similarity matrix and correlation similarity matrix. This can take awhile (minutes)
corr_sim_mat = vae_utils.get_correlation_similarity(vae,kin_var_norm)

path_to_PDs = glob.glob(project_folder + '\\*PD_calc*')
if(len(path_to_PDs)>0):
    f=open(path_to_PDs[0],'rb')
    hand_vel_PDs = pickle.load(f)
    f.close()
    
else:
    hand_vel_PDs, hand_vel_params = vae_utils.get_PD_similarity(vae,kin_var_norm,hand_vels)
    
    f=open(project_folder + '\\PD_calc.pkl','wb')
    pickle.dump(hand_vel_PDs,f)
    f.close()
    
    
    
#%% train decoder but constrain direction to be in PD
rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)

constrained_dec = vae_utils.make_constrained_linear_decoder(x=rates,y=hand_vels,PDs=hand_vel_PDs,drop_rate=0.99,n_iters=2000,lr=0.001)

y_hat = vae_utils.linear_constrained_dec_forward(dec=constrained_dec,x=rates,PDs=hand_vel_PDs)

hand_vaf = []
for i in range(y_hat.shape[1]):
    hand_vaf.append(mdl.vaf(hand_vels[:,i],y_hat[:,i]))
    
print(hand_vaf)
#%% save constrained dec
f=open(project_folder + '\\constrained_straight_to_hand_dec.pkl','wb')
pickle.dump(constrained_dec,f)
f.close()

#%% retrain vae decoder and plot weights vs. PD

dec_weights = vae.decoder.state_dict()['layer1.0.weight'].numpy()
vae_dec_new = vae_utils.make_linear_decoder(x=rates, y=kin_var_norm, drop_rate=0.995,n_iters=2,lr=0.001,init_weights=dec_weights)   
hand_dec_weights = dec.state_dict()['layer1.0.weight'].numpy()
whole_decoder_weights = np.matmul(np.transpose(vae_dec_new.state_dict()['layer1.0.weight'].numpy()),np.transpose(hand_dec_weights))
whole_decoder_dir = np.arctan2(whole_decoder_weights[:,1],whole_decoder_weights[:,0])

PD_err = vae_utils.circular_diff(whole_decoder_dir, hand_vel_PDs)
    
plt.figure()
plt.hist(np.abs(PD_err)*180/np.pi,50)   

weight_mag=np.sqrt(np.sum(np.square(whole_decoder_weights),axis=1))
plt.figure()
plt.plot(whole_decoder_dir, np.sqrt(np.sum(np.square(whole_decoder_weights),axis=1)),'.')

#%% make a decoder based on the PDs of neurons
rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)
PDs_as_weights = np.vstack((np.cos(hand_vel_PDs), np.sin(hand_vel_PDs)))/6400

# normalize PDs_as_weights based on PD distribution
PD_distribution, bin_edges = np.histogram(hand_vel_PDs,20)
bin_idx = np.digitize(hand_vel_PDs, bin_edges) - 1
bin_idx[bin_idx < 0] = 0
bin_idx[bin_idx >= len(PD_distribution)] = len(PD_distribution)-1
n_bin = PD_distribution[bin_idx]

PDs_as_weights = PDs_as_weights/n_bin

straight_to_hand_dec = vae_utils.make_linear_decoder(x=rates, y=hand_vels, drop_rate=0.,n_iters=1,lr=0.01)
straight_to_hand_dec.state_dict()['layer1.0.weight'][:] = torch.Tensor(PDs_as_weights)
