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

import pickle
import datetime
import torch
%matplotlib qt
#%% load in pretrained model and joint velocity data, both normalized and raw
# get project folder, file with vae_params and training parameters
project_folder = r'D:\Lab\Data\StimModel\models\Han_20160315_RW_2021-09-07-184542'
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
    
    

# evaluate decoder by plotting joint velocities against actual joint vel
hand_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=kin_var_norm)
hand_vaf = []

for hand in range(hand_vels.shape[1]):
    hand_vaf.append(mdl.vaf(hand_vels[:,hand],hand_vels_hat[:,hand]))
print(hand_vaf)


x = vae.state_dict()
dec_weights = x['decoder.layer1.0.weight'].numpy()
plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.hist(dec_weights[i,:])

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
plt.colorbar()
#%% compare PD and hand-vel-decoder direction
#%% ONLY MAKES SENSE IF VAE DECODES HAND VELS
x = vae.state_dict()
dec_weights = x['decoder.layer1.0.weight'].numpy()

dec_dir = np.arctan2(dec_weights[1,:],dec_weights[0,:])

plt.figure()
plt.hist(vae_utils.circular_diff(dec_dir,hand_vel_PDs),20)



#%% get PD similarity vs. distance for neurons on and not-on the same electrode
neigh_pd_diff, neigh_dist, non_neigh_pd_diff, non_neigh_dist = vae_utils.get_pd_neighbor_dist(mdl.locmap(), hand_vel_PDs, max_neigh_dist=3) # in blocks


plt.figure()
plt.hist(np.abs(non_neigh_pd_diff))
plt.figure()
plt.hist(np.abs(neigh_pd_diff),density=True)

#%% run experiments for multiple amplitudes and number of electrodes
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
#input_data['amp_list'] = [0,1,2,3,4,5]
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 20 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.067 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 200
input_data['n_stim_chans_list'] = [1,4]

input_data['PD_tol'] = np.pi/8

amp_elec_exp_out, n_chans, amp, stim_exp_out = stim_exp_utils.run_elec_amp_stim_exp(input_data)

exp_fname = 'amp_elec_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_elec_exp_out,n_chans,amp,input_data],f)
f.close()

#%% load data (assuming a break)
fname = project_folder+'\\amp_elec_exp_ratemult10_2021-09-02-141425.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_elec_exp_out = temp[0]
n_chans = temp[1]
amp = temp[2]
input_data = temp[3]

#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir

plt.figure()

data_to_plot = []
pos_to_plot = []
idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_elec_exp_out[idx_look][1]
        data_to_plot.append(delta_mag_stim[:,0])
        pos_to_plot.append(input_data['amp_list'][i_amp]+i_chan)#-len(input_data['n_stim_chans_list'])/2)
        idx_look=idx_look+1
        
plt.boxplot(data_to_plot,positions=pos_to_plot,widths=1)


plt.xlabel('Amplitude')
plt.ylabel('Change in hand vel magnitude')


# histogram summary
plt.figure()
bin_edges = np.arange(0,np.pi,np.pi/10)

colors = plt.cm.inferno([0,40,80,120,160,200,240])

ls = ['solid','dashed','dotted']
m = ['.','s','v']

idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_elec_exp_out[idx_look][-2]
        pred_delta_dir = amp_elec_exp_out[idx_look][-1]
    
        hist_vals,edges = np.histogram(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])),bin_edges)
        plt.plot((bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2)*180/np.pi,hist_vals/delta_dir_stim.shape[0],color=colors[i_amp],linestyle=ls[i_chan],marker=m[i_chan],markersize=12)
        idx_look = idx_look+1
    
plt.xlim((0,180))
plt.xlabel('Actual - Predicted Dir (deg)')
plt.ylabel('Proportion of trials')    
plt.legend(['0uA','5uA','10uA','15uA','20uA','40uA','80uA'])

# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;

prop_below_thresh = np.zeros((len(input_data['n_stim_chans_list']),len(input_data['amp_list'])))
plt.figure()
ls = ['solid','dashed','dotted']
m = ['.','s','v']
colors = plt.cm.inferno([0,100,200])

idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_elec_exp_out[idx_look][-2]
        pred_delta_dir = amp_elec_exp_out[idx_look][-1]
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_below_thresh[i_chan,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_chan,:],color=colors[i_chan],linestyle=ls[i_chan],marker=m[i_chan],markersize=12)
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['1 chan','4 chan'])



#%% run experiments comparing effect of amplitude and frequency
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,20,40,80] # uA
input_data['freq_list'] = [50,100,200,500] # Hz

input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['n_pulses_list'] = [10,20,40,100]
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.067 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 1000
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

amp_freq_exp_out, freq, amp = stim_exp_utils.run_amp_freq_stim_exp(input_data)

exp_fname = 'amp_freq_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_freq_exp_out,freq,amp,input_data],f)
f.close()

#%% load data
fname = project_folder+'\\amp_freq_exp_2021-09-01-143356.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_freq_exp_out = temp[0]
freq = temp[1]
amp = temp[2]
input_data = temp[3]

#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir

plt.figure()

data_to_plot = []
pos_to_plot = []
idx_look = 0
for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_freq_exp_out[idx_look][1]
        data_to_plot.append(delta_mag_stim[:,0])
        pos_to_plot.append(input_data['amp_list'][i_amp]+i_freq-len(input_data['freq_list'])/2)
        idx_look=idx_look+1
        
plt.boxplot(data_to_plot,positions=pos_to_plot,widths=1)


plt.xlabel('Amplitude')
plt.ylabel('Stim effect mag')


# histogram summary
plt.figure()
bin_edges = np.arange(0,np.pi,np.pi/10)

colors = plt.cm.inferno([0,50,100,150,200,250])
ls = ['solid','dashed','dotted','solid']
m = ['.','s','v','^']

idx_look = 0
for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_freq_exp_out[idx_look][-2]
        pred_delta_dir = amp_freq_exp_out[idx_look][-1]
    
        hist_vals,edges = np.histogram(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])),bin_edges)
        plt.plot((bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2)*180/np.pi,hist_vals/delta_dir_stim.shape[0],color=colors[i_amp],linestyle=ls[i_freq],marker=m[i_freq],markersize=12)
        idx_look = idx_look+1
    
plt.xlim((0,180))
plt.xlabel('Actual - Predicted Dir (deg)')
plt.ylabel('Proportion of trials')    
plt.legend(['0uA','5uA','10uA','20uA','40uA','80uA'])

# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;

prop_below_thresh = np.zeros((len(input_data['freq_list']),len(input_data['amp_list'])))
idx_look = 0
plt.figure()
colors = plt.cm.inferno([0,50,100,150,200,250])

for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_freq_exp_out[idx_look][-2]
        pred_delta_dir = amp_freq_exp_out[idx_look][-1]
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_below_thresh[i_freq,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_freq,:],color=colors[i_freq])
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['50 Hz','100 Hz','200 Hz','500 Hz'])

#%% run single electrode experiment where we pick locations with high neighbor similarity

PD_sim_mat = (np.pi - np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1))))/np.pi

neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)
neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))

plt.imshow(neigh_map[:,:,0])

#%%
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,20] # uA
input_data['step_list'] = [0,1,2,4]
input_data['freq'] = 100 # Hz

input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['n_pulses'] = 20
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.067 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 1000
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

input_data['sim_score'] = neigh_sim
input_data['sim_tol'] = 0.65 #

amp_high_sim_exp_out, amp, step, stim_chan_list = stim_exp_utils.run_amp_high_sim_exp(input_data)

#%% save data
exp_fname= 'amp_highsim_exp'
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_high_sim_exp_out,amp,step,stim_chan_list,input_data],f)
f.close()

#%% load data
fname = project_folder+'\\amp_freq_exp_2021-09-01-143356.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_freq_exp_out = temp[0]
freq = temp[1]
amp = temp[2]
input_data = temp[3]



#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir
plt.figure()
bin_edges = np.arange(0,1,0.1)

colors = plt.cm.inferno([0,40,80,120,160,200,240])

ls = ['solid','dashed','dotted','solid','dashed','dotted']
m = ['.','s','v','.','s','v']

idx_look = 0
for i_step in range(len(input_data['step_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = stim_chan_list[idx_look][-2]
        pred_delta_dir = stim_chan_list[idx_look][-1]
    
        hist_vals,edges = np.histogram(neigh_sim[stim_chan_list[idx_look].astype(int).reshape(-1,1)],bin_edges)
        plt.plot((bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2),hist_vals/delta_dir_stim.shape[0],color=colors[i_step],linestyle=ls[i_step],marker=m[i_step],markersize=12)
        idx_look = idx_look +1


plt.figure()

data_to_plot = []
pos_to_plot = []
idx_look = 0
for i_step in range(len(input_data['step_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_high_sim_exp_out[idx_look][1]
        data_to_plot.append(delta_mag_stim[:,0])
        pos_to_plot.append(input_data['amp_list'][i_amp] + i_step)#-len(input_data['n_stim_chans_list'])/2)
        idx_look = idx_look +1
plt.boxplot(data_to_plot,positions=pos_to_plot,widths=1)


plt.xlabel('Amplitude')
plt.ylabel('Change in hand vel magnitude')


# histogram summary
plt.figure()
bin_edges = np.arange(0,np.pi,np.pi/10)

colors = plt.cm.inferno([0,40,80,120,160,200,240])

ls = ['solid','dashed','dotted','solid','dashed','dotted']
m = ['.','s','v','.','s','v']
idx_look = 0
for i_step in range(len(input_data['step_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_high_sim_exp_out[idx_look][-2]
        pred_delta_dir = amp_high_sim_exp_out[idx_look][-1]
    
        hist_vals,edges = np.histogram(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])),bin_edges)
        plt.plot((bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2)*180/np.pi,hist_vals/delta_dir_stim.shape[0],color=colors[i_amp],linestyle=ls[i_step],marker=m[i_step],markersize=12)
        idx_look = idx_look +1
        
    
plt.xlim((0,180))
plt.xlabel('Actual - Predicted Dir (deg)')
plt.ylabel('Proportion of trials')    
plt.legend(['0uA','5uA','10uA','15uA','20uA','40uA','80uA'])

# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;

prop_below_thresh = np.zeros((len(input_data['step_list']),len(input_data['amp_list'])))
plt.figure()
ls = ['solid','dashed','dotted','solid','dashed','solid','dashed','solid','dashed']
m = ['.','.','.','.','.','.','.','.','.',]
colors = plt.cm.inferno([0,40,80,120,160,200,240])

idx_look = 0
for i_step in range(len(input_data['step_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_high_sim_exp_out[idx_look][-2]
        pred_delta_dir = amp_high_sim_exp_out[idx_look][-1]
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_below_thresh[i_step,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        idx_look = idx_look +1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_step,:],color=colors[i_step],linestyle=ls[i_step],marker=m[i_step],markersize=12)
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['0 step','1 step','2 step','4 step','8 step','16 step'])


#%%

# 
good_locs = np.array([[23,46,14,27,11,10],[48,32,12,16,39,39]])
good_idx = vae_utils.convert_loc_to_idx(good_locs,mdl.locmap().astype(int))

print(neigh_sim[good_idx.astype(int)])
#%%
# run stimulation trial
# set stim params

kin_var_samp = kin_var_norm[0:50] # use normalized joint velocity for vae, actual joint velocity for decoder and opensim

input_data = {}

input_data['stim_chan'] = np.array([200])
input_data['amp'] = 1 # uA
input_data['freq'] = 100 # Hz
input_data['n_pulses'] = 10 # s
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['trans_func'] = 'none'
input_data['vae'] = vae

input_data['joint_vel'] = kin_var_samp
input_data['init_joint_ang'] = joint_angs[0]
input_data['dec'] = dec
input_data['kin'] = 'hand'

input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.1 # mm

# [rates, samp_rates, samp_rates_stim, is_act, joint_vel_stim, point_kin_data]
stim_out = stim_exp_utils.run_stim_trial(input_data)
no_stim_rates = stim_out[1]
stim_rates = stim_out[2]
is_act = stim_out[3]

point_kin = stim_out[-1] # [hand_pos,hand_vel,hand_acc,elbow_pos,elbow_vel,elbow_acc]

no_stim_rates, stim_rates = vae_utils.visualize_activation_map(stim_rates, no_stim_rates, idx=1)
vae_utils.visualize_activation_dist(is_act, input_data['stim_chan'],input_data['block_size'])

#%%
n_samp = 10000
rate = 20+np.zeros((n_samp,1))
samp_rate = vae_utils.sample_rates(rate)

#%%

"""
# compare hand and joint decoders...

# step 1 - compare hand vel decoded from normal rates
idx_start = 0
idx_end = 500
joint_vel_samp = joint_vels_norm[idx_start:idx_end]
init_joint_ang = joint_angs[idx_start]

# get rates
rates = vae_utils.vae_get_rates(vae,joint_vel_samp,gl.bin_size)
# get hand vels from hand vel dec
hand_vels_hat = vae_utils.linear_dec_forward(dec=hand_dec,x=rates)
# get hand vels from joint vel dec
joint_vels_hat = vae_utils.linear_dec_forward(dec=dec,x=rates)
int_joint_ang = vae_utils.integrate_vel(init_joint_ang,joint_vels_hat,gl.bin_size)
point_kin_data = osim_utils.get_pointkin(int_joint_ang)

joint_dec_hand_vels = point_kin_data[1][:,1:3]

plt.figure()
plt.plot(hand_vels_hat[:,0])
#plt.plot(joint_dec_hand_vels[:,0])

#




#
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