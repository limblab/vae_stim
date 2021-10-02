# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:38:49 2021

@author: Joseph Sombeck
"""

#%% load in useful variables, imports, etx.
import global_vars as gl # import global variables first. I'm not ecstatic about this approach but oh well
gl.global_vars()

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

import vae_utils 
import vae_model_code as mdl
import glob
import stim_exp_utils
import osim_utils
import scipy as sp

from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
 
import pickle
import datetime
import torch
%matplotlib qt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import random


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
    
    
#%% run experiments for multiple amplitudes and number of electrodes
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
#input_data['amp_list'] = [0,5,10,15]
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 200 # Hz
input_data['n_pulses'] = 40 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 1
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 500
input_data['n_stim_chans_list'] = [1,2,4,8]

input_data['PD_tol'] = np.pi/8

amp_elec_exp_out, n_chans, amp, stim_exp_out, stim_chan_list = stim_exp_utils.run_elec_amp_stim_exp(input_data)

exp_fname = 'amp_elec_single_map_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_elec_exp_out,n_chans,amp,input_data, stim_chan_list],f)
f.close()

#%% load data (assuming a break)
fname = project_folder+'\\amp_elec_single_map_exp_ratemult10_2021-09-30-111320.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_elec_exp_out = temp[0]
n_chans = temp[1]
amp = temp[2]
input_data = temp[3]
stim_chan_list = temp[4]

#%% magnitude of effect, error bars across trials of the same location

colors = plt.cm.inferno([0,70,140,210])

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mean_mag = np.zeros((len(input_data['n_stim_chans_list']),len(input_data['amp_list'])))
std_mag = np.zeros_like(mean_mag)
offset = [-1,-0.5,0,0.5,1]*1

idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_elec_exp_out[idx_look][1]
        data = delta_mag_stim[:,0]/max_hand_vel_mag*100
        mean_mag[i_chan,i_amp] = np.mean(data)
        std_mag[i_chan,i_amp] = np.std(data)
        idx_look=idx_look+1
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_chan],mean_mag[i_chan,:], std_mag[i_chan,:], capsize=5,elinewidth=2, \
                     color=colors[i_chan],linewidth=2,marker='.',markersize=16)

plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')


#%% angular error -- current through each electrode

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

mean_err = np.zeros((len(input_data['n_stim_chans_list']),len(input_data['amp_list'])))
std_err = np.zeros_like(mean_mag)
offset = [-1,-0.5,0,0.5,1]*1
idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_elec_exp_out[idx_look][-2]
        pred_delta_dir = amp_elec_exp_out[idx_look][-1]
        error_all = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))*180/np.pi
        mean_err[i_chan,i_amp] = np.mean(error_all)
        std_err[i_chan,i_amp] = np.std(error_all)
        idx_look=idx_look+1
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_chan],mean_err[i_chan,:], std_err[i_chan,:], capsize=5,elinewidth=2, \
                     color=colors[i_chan],linewidth=2,marker='.',markersize=16)


plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,85])
plt.ylim([0,180])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,200,40))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(10))

#%% plot PD map with stimulated locations more prominent than non-stimulated
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mdl.locmap().astype(int))[:,:,0]
alpha_list = np.zeros_like(PD_map) + 0.5
is_stim = np.ones_like(PD_map)

loc_map = mdl.locmap()

idx_look = 0
for i_chan in [0]:#range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])): 
        if(i_amp>0): # ignore 0uA condition
            stim_chans = stim_chan_list[idx_look].reshape(-1,)
            stim_chans_loc = loc_map[stim_chans.astype(int)].astype(int)
            for i in range(len(stim_chans_loc)):
                alpha_list[stim_chans_loc[i,0],stim_chans_loc[i,1]] = 1.0
        idx_look=idx_look+1
 
    
fig = plt.figure(figsize=(6,5))
ax = fig.gca()   
im=plt.imshow(PD_map, cmap='twilight')
#cbar = fig.colorbar(plt.cm.ScalarMappable(norm=im.norm,cmap=im.cmap))
cbar = plt.colorbar()
cbar.set_label('PD (deg)')
cbar.set_ticks(np.array([-180,-90,0,90,180]))  
    
   




#%% compute neigh sim
    
PD_sim_mat = np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1)))/np.pi*180
neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)
    

#%% neighborhood score vs. angular error for each location

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

mean_err = np.zeros((len(input_data['n_stim_chans_list']),len(input_data['amp_list'])))
std_err = np.zeros_like(mean_mag)

idx_look = [17]

for i_look in range(len(idx_look)):
    delta_dir_stim = amp_elec_exp_out[idx_look[i_look]][-2]
    pred_delta_dir = amp_elec_exp_out[idx_look[i_look]][-1]

    error_all = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))*180/np.pi
    neigh_scores = np.mean(neigh_sim[stim_chan_list[idx_look[i_look]].astype(int)],axis=1)
    
    ax.plot(neigh_scores, error_all, \
                     color=colors[i_look],linewidth=0,marker='.',markersize=16)



#%% stimulate the same location(s) many times and measure effect -- measure consistency across many locations
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

#input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
input_data['amp_list'] = [0,5,10,15,20,40,80]
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 200 # Hz
input_data['n_pulses'] = 40 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 1
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 20
input_data['n_stim_chans'] = 2 # can't be a list
input_data['n_sets'] = 500

input_data['PD_tol'] = np.pi/8

single_loc_exp, loc_idx, amp, stim_exp_out = stim_exp_utils.run_single_location_exp(input_data)

exp_fname = 'multi_loc_repetition_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([single_loc_exp,loc_idx,amp,input_data],f)
f.close()

#%% load data (assuming a break)
fname = project_folder+'\\single_loc_exp_ratemult10_nchans1_2021-09-29-151346.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

single_loc_exp = temp[0]
loc_idx = temp[1]
amp = temp[2]
input_data = temp[3]

#%% measure standard deviation of angular error, plot as histogram for provided amplitude(s)
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\multi_loc_repetition_exp*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data = temp[3]

# get std values across folders
std_ang_err = np.zeros((len(fnames),input_data['n_sets'],len(input_data['amp_list']))) # num electrodes, n_sets, n_amps
mean_ang_err = np.zeros_like(std_ang_err)
mean_mag = np.zeros_like(std_ang_err)

for i_folder in range(len(fnames)):
    idx_look = 0
    f=open(fnames[i_folder],'rb')
    temp = pickle.load(f)
    f.close()
    
    single_loc_exp = temp[0]
    loc_idx = temp[1]
    amp = temp[2]
    input_data = temp[3]
    for i_set in range(input_data['n_sets']):
        for i_amp in range(len(input_data['amp_list'])):
            # angular error data
            delta_dir_stim = single_loc_exp[idx_look][-2]
            pred_delta_dir = single_loc_exp[idx_look][-1]
            error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
            mean_ang_err[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
            std_ang_err[i_folder,i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
        
            
            # magnitude data
            delta_mag_stim = single_loc_exp[idx_look][1]
            data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
            mean_mag[i_folder,i_set,i_amp] = np.mean(data)
            
            idx_look = idx_look + 1

#%% plot errobars of std_err for each amplitude and num-elec
colors = plt.cm.inferno([0,90,180])
fig = plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])
bin_edges = np.arange(0,100,5)
offset = np.array([-0.75,-0.25,0.25,0.75])*0.5

for i_elec in range(len(fnames)):
    mean_std =np.mean(std_ang_err[i_elec,:,:],axis=0)
    std_std = np.std(std_ang_err[i_elec,:,:],axis=0)
    plt.errorbar(np.array(input_data['amp_list'])+offset[i_elec],mean_std,std_std, \
                 color=colors[i_elec],linewidth=2,marker='.',markersize=16)

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Standard dev. of angular errors (deg)')

plt.xlim([-5,85])
plt.ylim([0,160])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,180,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))


#%% magnitude of effect, error bars across trials of the same location
colors = plt.cm.inferno([0,90,180])
offset = [-1,-0.5,0,0.5,1]*1

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

for i_elec in range(mean_mag.shape[0]):
    mean_to_plot = np.mean(mean_mag[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_mag[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_elec],mean_to_plot, std_to_plot, capsize=5,elinewidth=2, \
                     color=colors[i_elec],linewidth=2,marker='.',markersize=16)

plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')
#%% angular error
colors = plt.cm.inferno([0,90,180])

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

offset = [-1,-0.5,0,0.5,1]*1

for i_elec in range(mean_ang_err.shape[0]):
    mean_to_plot = np.mean(mean_ang_err[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_ang_err[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_elec],mean_to_plot, std_to_plot, capsize=5,elinewidth=2, \
                     color=colors[i_elec],linewidth=2,marker='.',markersize=16)


plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,85])
plt.ylim([0,160])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,180,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
    

"""
# plot histograms (1 fig for each num-elec) of std_err. use subset of amps
colors = plt.cm.inferno([0,40,80,120,160,200,240])
plt.figure()
bin_edges = np.arange(0,np.ceil(np.max(std_ang_err)),1)
offset = [-0.75,-0.25,0.25,0.75]*1

fig,ax = plt.subplots(4,1,sharex=True,sharey=True)

for i_elec in range(len(fnames)):
    color_idx = 0
    for i_amp in [0,1,2,3,4,6]:
        ax[i_elec].hist(std_ang_err[i_elec,:,i_amp],bins=bin_edges,histtype='step',cumulative=True,linewidth=2,color=colors[color_idx])
        
        color_idx=color_idx+1
"""


















