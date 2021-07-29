# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:25:36 2021

@author: Joseph Sombeck
"""
import numpy as np
import matplotlib.pyplot as plt

import vae_utils 
import vae_model_code as mdl
import glob
import stim_exp_utils
import scipy as sp

import global_vars as gl
gl.global_vars()

#%% load in pretrained model and joint velocity data, both normalized and raw
# get project folder, file with vae_params and training parameters
project_folder = r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801'
path_to_model_dict = glob.glob(project_folder + r'\*model_params*')[0]
path_to_model_yaml = glob.glob(project_folder + r'\*.yaml')[0]
path_to_norm_joint_vels = glob.glob(project_folder + r'\*NormalizedJointVel*.txt')[0]
path_to_joint_vels = glob.glob(project_folder + r'\*RawJointVel*.txt')[0]
path_to_joint_angs = glob.glob(project_folder + '\\*JointAng*')[0]
path_to_data_mask = glob.glob(project_folder + '\\*TrialMask*')[0]


# get data, only use times when trial mask is 1
trial_mask = np.genfromtxt(path_to_data_mask,delimiter=',')[:]
joint_vels_norm = np.genfromtxt(path_to_norm_joint_vels, delimiter=',')[trial_mask==1,:]
joint_vels = np.genfromtxt(path_to_joint_vels,delimiter=',')[trial_mask==1,:]
joint_angs = np.genfromtxt(path_to_joint_angs,delimiter=',')[trial_mask==1,:]

# load parameter file. This is currently convoluted, but works
gl.Params.load_params(gl.params,path_to_model_yaml)

# set params['cuda'] to false since my computer doesn't have a GPU ( :( )
gl.params.params['cuda']=False

# load in vae weights
vae = vae_utils.load_vae_parameters(fpath=path_to_model_dict,input_size=joint_vels_norm.shape[1]) 


#%% train linear decoder for this network from rates to actual joint_vels
rates = vae_utils.vae_get_rates(vae,joint_vels_norm,gl.bin_size)
dec = vae_utils.make_linear_decoder(x=rates, y=joint_vels, drop_rate=0.99,n_iters=5,lr=0.01)
dec_weights, dec_bias = vae_utils.get_decoder_params(dec=dec)

# evaluate decoder by plotting joint velocities against actual joint vel
joint_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=rates)
vaf_list = []
for joint in range(joint_vels_norm.shape[1]):
    vaf_list.append(mdl.vaf(joint_vels[:,joint],joint_vels_hat[:,joint]))
print(vaf_list)

data_min = 1;
data_max = 250;
idx = 1;

plt.plot(joint_vels[data_min:data_max,idx])
plt.plot(joint_vels_hat[data_min:data_max,idx])

joint_vels_hat = joint_vels_hat[0:100,:]

#%% get PD similarity matrix and correlation similarity matrix. This can take awhile (minutes)
corr_sim_mat = vae_utils.get_correlation_similarity(vae,joint_vels_norm)

# load in PD mats instead of calculating...
#PD_sim_mat, hand_vel_PDs, hand_vel_params = vae_utils.get_PD_similarity(vae,joint_vels_norm,joint_angs)
#hand_vel_PDs = np.genfromtxt(r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801\PD_calc.txt')
#PD_sim_mat = np.genfromtxt(r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801\PD_sim_mat.txt')

#%% run many stimulation trials, randomly choosing stimulation channel and joint velocities
input_data = {}
input_data['amp'] = 10 # uA
input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 10 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'exp_decay'
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = joint_vels_norm
input_data['all_joint_ang'] = joint_angs
input_data['dec'] = dec
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 25
input_data['n_stim_chans'] = 1

input_data['sim_mat'] = corr_sim_mat
input_data['sim_tol'] = 0.60
stim_exp_out=stim_exp_utils.run_many_stim_trials(input_data)

stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)
# return [joint_ang_0_list, true_rates_list, no_stim_rates_list, stim_rates_list, 
#      stim_chan_list,  true_joint_vels_list, no_stim_joint_vels_list, stim_joint_vels_list, 
#      true_point_kin_list, no_stim_point_kin_list, stim_point_kin_list]

#%% compute metrics for stim experiment


# compute change in FR with and without stim compared to underlying
rates_true = stim_exp_out[1]
rates_no = stim_exp_out[2]
rates_stim = stim_exp_out[3]

delta_rates_stim = np.zeros((rates_true.shape[0],))
delta_rates_no = np.zeros_like(delta_rates_stim)

for i_trial in range(rates_true.shape[0]): # each stim trial
    delta_rates_stim[i_trial] = np.sum(np.mean(rates_stim[i_trial,stim_start_idx:stim_end_idx,:] - rates_true[i_trial,stim_start_idx:stim_end_idx,:],axis=0))
    delta_rates_no[i_trial] = np.sum(np.mean(rates_no[i_trial,stim_start_idx:stim_end_idx,:] - rates_true[i_trial,stim_start_idx:stim_end_idx,:],axis=0))

#%%
# compare change in hand/elbow vel from underlying to PD prediction



#%%# compute magnitude of hand vel comparing stim and no stim to underlying

point_kin_true = stim_exp_out[8]
point_kin_no = stim_exp_out[9]
point_kin_stim = stim_exp_out[10]


delta_vel_stim = np.zeros((len(point_kin_true),2)) # trial, hand/elbow vel mag
delta_vel_no = np.zeros_like(delta_vel_stim)

for i_trial in range(len(point_kin_true)): # each stim trial
    # hand
    delta_vel_stim[i_trial,0] = np.linalg.norm(np.mean(point_kin_stim[i_trial][1][stim_start_idx:stim_end_idx,1:3] - point_kin_true[i_trial][1][stim_start_idx:stim_end_idx,1:3],axis=0))
    delta_vel_no[i_trial,0] = np.linalg.norm(np.mean(point_kin_no[i_trial][1][stim_start_idx:stim_end_idx,1:3] - point_kin_true[i_trial][1][stim_start_idx:stim_end_idx,1:3],axis=0))
    # elbow
    delta_vel_stim[i_trial,1] = np.linalg.norm(np.mean(point_kin_stim[i_trial][4][stim_start_idx:stim_end_idx,1:3] - point_kin_true[i_trial][4][stim_start_idx:stim_end_idx,1:3],axis=0))
    delta_vel_no[i_trial,1] = np.linalg.norm(np.mean(point_kin_no[i_trial][4][stim_start_idx:stim_end_idx,1:3] - point_kin_true[i_trial][4][stim_start_idx:stim_end_idx,1:3],axis=0))
    
    
plt.hist(delta_vel_stim[:,0]-delta_vel_no[:,0])



#stim_exp_metrics = stim_exp_utils.compute_exp_metrics(stim_exp_out)


#%% run stimulation trial
# set stim params

joint_vel_samp = joint_vels_norm[0:50] # use normalized joint velocity for vae, actual joint velocity for decoder and opensim

input_data = {}
input_data['stim_chan'] = np.array([4200])
input_data['amp'] = 1 # uA
input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 100 # s
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'exp_decay'
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['joint_vel'] = joint_vel_samp
input_data['init_joint_ang'] = joint_angs[0]
input_data['dec'] = dec
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

# [rates, samp_rates, samp_rates_stim, is_act, joint_vel_stim, point_kin_data]
stim_out = stim_exp_utils.run_stim_trial(input_data)
no_stim_rates = stim_out[1]
stim_rates = stim_out[2]
is_act = stim_out[3]

point_kin = stim_out[-1] # [hand_pos,hand_vel,hand_acc,elbow_pos,elbow_vel,elbow_acc]



# visualize activation
vae_utils.visualize_activation_map(stim_rates, no_stim_rates, idx=1)
vae_utils.visualize_activation_dist(is_act, input_data['stim_chan'],input_data['block_size'])

#%%
"""
#%% plot decoder weights as a histogram -- make sure weights are not large positive and negative values
for i in range(dec_weights.shape[0]):
    plt.figure()
    plt.hist(dec_weights[i,:])

#%% evaluate decoder by plotting joint velocities against actual joint vel
joint_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=rates)
vaf_list = []
for joint in range(joint_vels_norm.shape[1]):
    vaf_list.append(mdl.vaf(joint_vels[:,joint],joint_vels_hat[:,joint]))
print(vaf_list)

data_min = 1;
data_max = 250;
idx = 1;

plt.plot(joint_vels[data_min:data_max,idx])
plt.plot(joint_vels_hat[data_min:data_max,idx])

joint_vels_hat = joint_vels_hat[0:100,:]


#%% visualize PDs
mapping = mdl.locmap().astype(int)
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mapping)

PD_hist, PD_bin_edges = np.histogram(hand_vel_PDs)

PD_bin_centers = PD_bin_edges[0:-1] + np.mean(np.diff(PD_bin_edges))/2

plt.figure()
plt.subplot(111,polar=True)
plt.bar(x=PD_bin_centers,
        height=PD_hist,
        width=2*np.pi/len(PD_bin_centers))

plt.figure()
plt.imshow(PD_map[:,:,0])



"""