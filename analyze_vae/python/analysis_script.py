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
import scipy as sp

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
dec = vae_utils.make_linear_decoder(x=rates, y=joint_vels, drop_rate=0.95,n_iters=250,lr=0.01)

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
hand_vel_PDs = np.genfromtxt(r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801\PD_calc.txt')
PD_sim_mat = np.genfromtxt(r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-05-12-210801\PD_sim_mat.txt')

#%% run experiments for multiple amplitudes
input_data = {}
input_data['amp_list'] = [0,5,20,50] # uA
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 1 
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
input_data['n_trials'] = 150
input_data['n_stim_chans'] = 1

input_data['sim_mat'] = PD_sim_mat
input_data['sim_tol'] = 0.80

amp_exp_out = stim_exp_utils.run_amp_stim_exp(input_data)

#%% compare metrics across amps and plot
# amp_exp_out: For each amp: delta_vel_stim, delta_vel_no, delta_mag_stim, delta_mag_no, delta_dir_stim, delta_dir_no, pred_delta_dir

plt.figure()

data_to_plot = []
for i_amp in range(len(input_data['amp_list'])):
    delta_mag_stim = amp_exp_out[i_amp][2]
    delta_mag_no = amp_exp_out[i_amp][3]
    data_to_plot.append(delta_mag_stim[:,0]-delta_mag_no[:,0])
     
    
plt.boxplot(data_to_plot,positions=input_data['amp_list'],widths=4)

plt.xlim((input_data['amp_list'][0]-10,input_data['amp_list'][-1]+10))
plt.xlabel('Amplitude')
plt.ylabel('Change in hand vel magnitude')

#%% direction

plt.figure()
bin_edges = np.arange(0,np.pi,np.pi/10)

colors = plt.cm.inferno([0,50,100,150,200,250])

for i_amp in range(len(input_data['amp_list'])):
    delta_dir_stim = amp_exp_out[i_amp][4]
    pred_delta_dir = amp_exp_out[i_amp][-1]

    hist_vals,edges = np.histogram(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])),bin_edges)
    plt.plot(bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2,hist_vals/delta_dir_stim.shape[0],color=colors[i_amp],marker='.',markersize=12)
    
plt.xlim((0,np.pi))
plt.xlabel('Actual - Predicted Dir (radians)')
plt.ylabel('Proportion of trials')    


#%%

# run stimulation trial
# set stim params

joint_vel_samp = joint_vels_norm[0:50] # use normalized joint velocity for vae, actual joint velocity for decoder and opensim

input_data = {}
input_data['stim_chan'] = np.array([200])
input_data['amp'] = 1 # uA
input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 10 # s
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
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
# run many stimulation trials, randomly choosing stimulation channel and joint velocities
input_data = {}
input_data['amp'] = 20 # uA
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
input_data['n_trials'] = 250
input_data['n_stim_chans'] = 1

input_data['sim_mat'] = PD_sim_mat
input_data['sim_tol'] = 0.80

# opensim's C code has a memory leak, it's small but if you run a ton of simulations you might have a problem.
stim_exp_out=stim_exp_utils.run_many_stim_trials(input_data)

stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)

#%% compute magnitude of hand vel comparing stim and no stim to underlying
# compare change in hand/elbow vel from underlying PD prediction
metrics = stim_exp_utils.compute_kin_metrics(stim_exp_out, hand_vel_PDs, stim_start_idx, stim_end_idx, make_plots=False)



#
# plot decoder weights as a histogram -- make sure weights are not large positive and negative values
for i in range(dec_weights.shape[0]):
    plt.figure()
    plt.hist(dec_weights[i,:])

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


# visualize PDs
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