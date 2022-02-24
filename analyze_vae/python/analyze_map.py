# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:23:00 2021

@author: Joseph Sombeck
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:25:36 2021

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
import statsmodels.api as sm
import seaborn as sns
import cmasher

import pickle
import datetime
import torch
%matplotlib qt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# set global plot style
sns.set_style('ticks') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('font', size=14)          # controls default text sizes

# load in pretrained model and joint velocity data, both normalized and raw
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

path_to_hand_pos_all = glob.glob(training_data_folder + '\\*RawHandPos_*')[0]
path_to_hand_vel_all = glob.glob(training_data_folder + '\\*RawHandVel_*')[0]
path_to_trial_mask_all = glob.glob(training_data_folder + '\\*RewardTrialMask_*')[0]
path_to_joint_vel_all = glob.glob(training_data_folder + '\\*NormalizedJointVel_*')[0]

# get data, only use times when trial mask is 1
trial_mask = np.genfromtxt(path_to_data_mask,delimiter=',')[:]
joint_vels_norm = np.genfromtxt(path_to_norm_joint_vels, delimiter=',')[trial_mask==1,:]
joint_vels = np.genfromtxt(path_to_joint_vels,delimiter=',')[trial_mask==1,:]
joint_angs = np.genfromtxt(path_to_joint_angs,delimiter=',')[trial_mask==1,:]
hand_vels = np.genfromtxt(path_to_hand_vels,delimiter=',')[trial_mask==1,0:2]
muscle_vels_norm = np.genfromtxt(path_to_norm_muscle_vel,delimiter=',')[trial_mask==1,:]
hand_vels_norm = np.genfromtxt(path_to_norm_hand_vels,delimiter=',')[trial_mask==1,:]


all_trial_mask = np.genfromtxt(path_to_trial_mask_all,delimiter=',')[:]
all_hand_vels = np.genfromtxt(path_to_hand_vel_all,delimiter=',')[all_trial_mask==1,:]
all_hand_pos = np.genfromtxt(path_to_hand_pos_all,delimiter=',')[all_trial_mask==1,:]
all_joint_vel_norm = np.genfromtxt(path_to_joint_vel_all,delimiter=',')[all_trial_mask==1,:]


# load parameter file. This is currently convoluted, but works
gl.Params.load_params(gl.params,path_to_model_yaml)

# set params['cuda'] to false since my computer doesn't have a GPU ( :( )
gl.params.params['cuda']=False

kin_var_norm = joint_vels_norm

# load in vae weights
vae = vae_utils.load_vae_parameters(fpath=path_to_model_dict,input_size=kin_var_norm.shape[1]) 


# train linear decoder for this network from kin_var_norm to actual hand_vels
# either train new decoder or load one in

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



# get PD similarity matrix and correlation similarity matrix. This can take awhile (minutes)
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

path_to_PDs = glob.glob(project_folder + '\\*PD_multiunit_hash_pow2*')
if(len(path_to_PDs)>0):
    f=open(path_to_PDs[0],'rb')
    hash_PDs = pickle.load(f)[0]
    f.close()

rates = gl.rate_mult*vae_utils.vae_get_rates(vae, all_joint_vel_norm,gl.bin_size)/gl.bin_size
#%% plot PD heatmap single unit and multi-unit hash
mapping = mdl.locmap().astype(int)


for pd_data in [hand_vel_PDs, hash_PDs]:
    PD_map = vae_utils.convert_list_to_map(pd_data.reshape(1,-1),mapping)
    PD_map = PD_map*180/np.pi
    
    #1 =colors = '#e41a1c'   0 = colors = '#377eb8'   8 = colors = '#4daf4a'  26 = colors = '#65463E'
    x = PD_map[:,:,0]
    f=plt.figure()
    ax=f.gca()
    plt.imshow(x,cmap=cmasher.infinity,interpolation='none',vmin=-180,vmax=180)
    cbar=plt.colorbar()
    cbar.set_label('Preferred Direction (Deg)')
    cbar.set_ticks(np.array([-180,-90,0,90,180])) 
    
    plt.xlim([0,80])  # this manages to flip the map vertically....
    plt.ylim([0,80])
    ax.set_aspect('equal')
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])

#%% plot position vs. velocity tuning for a few neurons
all_hand_spd = np.sqrt(np.sum(np.square(all_hand_vels[:,0:-1]),axis=1)).reshape(-1,1)
neuron_idx = np.random.randint(0,high=6400,size = (15,1))

neuron_idx = [3055,4803]
speed_test = np.linspace(0,0.3,100)
dir_test = np.linspace(-np.pi,np.pi,100)

# generate test data
test_data = np.ones((len(speed_test)*len(dir_test),6)) # hand pos x, hand pos y, hand vel x, hand vel y, hand spd, constant

counter = 0
for i_s in range(len(speed_test)):
    for i_d in range(len(dir_test)):
        vel = [speed_test[i_s]*np.cos(dir_test[i_d]), speed_test[i_s]*np.sin(dir_test[i_d])]
        test_data[counter,:] = [0,0,vel[0],vel[1],speed_test[i_s],1] # hand pos x, hand pos y, hand vel x, hand vel y, hand spd, constant
        counter = counter + 1

glm_fit = []
for i_n in neuron_idx:
    # fit glm
    X = np.concatenate((all_hand_pos[:,0:-1], all_hand_vels[:,0:-1], all_hand_spd, np.ones_like(all_hand_spd)),axis=1) # (x,y) position and velocity, speed
    Y = rates[:,i_n]
    # append to glm_fit
    glm_poiss = sm.GLM(Y,X,family=sm.families.Poisson())
    res = glm_poiss.fit()
    
    
    # test glm_poiss to get values for heatmap
    test_vals = res.predict(test_data)
    
    test_vals_map = test_vals.reshape((len(speed_test),len(dir_test)))
    
    f=plt.figure(figsize=(4,4))
    ax=f.add_axes([0.15,0.15,0.75,0.75])
    plt.imshow(test_vals_map,origin='lower',cmap=cmasher.ember)
    cbar=plt.colorbar(shrink=0.80)
    cbar.set_label('Firing rate')
    #cbar.set_ticks(np.array([0,90,180,270,360])) 

    ax.set_aspect('equal')
    
    plt.xticks(ticks=[0,len(dir_test)/2,len(dir_test)]) # have to manually override tick labels
    plt.yticks(ticks=[0,len(speed_test)/2,len(speed_test)])
    ax.xaxis.set_minor_locator(MultipleLocator(len(dir_test/8)))
    ax.yaxis.set_minor_locator(MultipleLocator(len(speed_test/8)))
    
    sns.despine()
    
    ax.set_xticklabels(labels=['-180','0','180'])
    ax.set_yticklabels(labels=['0','15','30'])
    plt.xlabel('Direction (Deg)')     
    plt.ylabel('Speed (cm/s)')
    
    
    
    
#%% plot PD on same vs different electrodes with Weber data overlayed

fname = glob.glob(project_folder+'\\neigh_dist_exp_2021-09-18-124705*')[0]
f=open(fname,'rb')
pd_diff,unit_dist = pickle.load(f) # dist is in blocks, will be converted to mm
f.close()

block_size = 0.05
unit_dist = unit_dist*block_size # mm

#%%
is_neigh = unit_dist < 0.15 # 150 um
    
bin_edges = np.linspace(0,np.pi,19)

neigh_hist, bin_edges = np.histogram(pd_diff[is_neigh==1],bins=bin_edges)
neigh_hist = neigh_hist/np.sum(is_neigh==1)
non_neigh_hist, bin_edges = np.histogram(pd_diff[is_neigh==0],bins=bin_edges)
non_neigh_hist = non_neigh_hist/np.sum(is_neigh==0)
    
bin_centers = bin_edges[0:-1] + np.mean(np.diff(bin_edges))/2
    
#%%
f=plt.figure(figsize=(5,5))
ax=f.add_axes([0.2,0.15,0.75,0.75])
plt.plot(bin_centers*180/np.pi, neigh_hist*100,'.',linestyle='--',color='b',markersize=16)
plt.plot(bin_centers*180/np.pi, non_neigh_hist*100,'.',linestyle='--',color='r',markersize=16)
plt.plot(bin_centers*180/np.pi, [15,6,10.5,9,9,9,6,0,6,6,3,3,0,3,6.5,1,1,1],'s',linewidth=2,linestyle='none',color='black',markersize=10,markerfacecolor='none')
    
plt.ylabel('Percentage')
plt.xlabel('Difference in Preferred Direction (Deg)')
    
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2.5)) 
plt.xticks(ticks=np.arange(0,200,60)) 
plt.yticks(ticks=np.arange(0,18,5))
sns.despine()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    