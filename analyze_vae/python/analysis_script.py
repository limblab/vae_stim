# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:25:36 2021

@author: Joseph Sombeck
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

import vae_utils 
import vae_model_code as mdl
import glob

vae_utils.global_vars();

#%% test code to load in a pre-trained model and run the forward model


# get project folder, file with vae_params and training parameters
project_folder = r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801'
path_to_model_dict = glob.glob(project_folder + r'\*model_params*')[0]
path_to_model_yaml = glob.glob(project_folder + r'\*.yaml')[0]
path_to_norm_joint_vels = glob.glob(project_folder + r'\*NormalizedJointVel*.txt')[0]
path_to_joint_vels = glob.glob(project_folder + r'\*RawJointVel*.txt')[0]

joint_vels_norm = np.genfromtxt(path_to_norm_joint_vels, delimiter=',')[1:25000,:]
joint_vels = np.genfromtxt(path_to_joint_vels,delimiter=',')[1:25000,:]

# load parameter file. This is currently convoluted, but works
vae_utils.Params.load_params(vae_utils.params,path_to_model_yaml)

# set params['cuda'] to false since my computer doesn't have a GPU ( :( )
vae_utils.params.params['cuda']=False

# setup base vae,  there are 7 joint angles. I need to figure out how to automatically load this parameter....
vae = vae_utils.load_vae_parameters(fpath=path_to_model_dict,input_size=joint_vels_norm.shape[1]) 


#%% get rates
rates = vae_utils.vae_get_rates(vae,joint_vels_norm,vae_utils.bin_size)

#%% train linear decoder from rates to joint_vels
dec = vae_utils.make_linear_decoder(x=rates, y=joint_vels, drop_rate=0.95,n_iters=500,lr=0.01)
dec_weights, dec_bias = vae_utils.get_decoder_params(dec=dec)

#%% plot joint vels and predicted joint vels
joint_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=rates)
vaf_list = []
for joint in range(joint_vels_norm.shape[1]):
    vaf_list.append(mdl.vaf(joint_vels[:,joint],joint_vels_hat[:,joint]))
print(vaf_list)

# compare joint vels and predicted joint vels
data_min = 1000;
data_max = 1500;
idx = 1;

plt.plot(joint_vels[data_min:data_max,idx])
plt.plot(joint_vels_hat[data_min:data_max,idx])

#%% put joint vels through opensim


