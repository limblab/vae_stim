# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:35:27 2021

@author: Joseph Sombeck
"""


#%% load in useful variables, imports, etx.
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
    
#%% compute neigh sim
    
PD_sim_mat = np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1)))/np.pi*180
neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)

neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))

plt.figure()
plt.imshow(neigh_map[:,:,0])

#%%
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,20,40,80] # uA
input_data['step_list'] = [0,1,2,4,8]
input_data['freq'] = 200 # Hz

input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['n_pulses'] = 40
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 1
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 50
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

input_data['sim_score'] = neigh_sim
input_data['sim_tol'] = np.percentile(neigh_sim,15) # only use similarities below this

amp_high_sim_exp_out, amp, step, stim_chan_list = stim_exp_utils.run_amp_high_sim_exp(input_data)

exp_fname= 'amp_highsim_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_high_sim_exp_out,amp,step,stim_chan_list,neigh_sim,input_data],f)
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


#%% histogram summary
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
