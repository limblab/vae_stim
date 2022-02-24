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

#%% joint vel vaf

kin_var_hat,rates = vae_utils.vae_forward(vae,kin_var_norm)

kin_vaf = []
kin_r2 = []
for hand in range(kin_var_norm.shape[1]):
    kin_vaf.append(mdl.vaf(kin_var_norm[:,hand],kin_var_hat[:,hand]))
    kin_r2.append(np.corrcoef(kin_var_norm[:,hand],kin_var_hat[:,hand])[0,1]**2)
print(kin_vaf)
print(kin_r2)
    
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
    hand_r2 = []
    for hand in range(hand_vels.shape[1]):
        hand_vaf.append(mdl.vaf(hand_vels[:,hand],hand_vels_hat[:,hand]))
        hand_r2.append(np.corrcoef(hand_vels[:,hand],hand_vels_hat[:,hand])[0,1]**2)
    print(hand_vaf)
    print(hand_r2)


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

#%% get multi-neuron hash PDs
    
hash_PDs, hash_params, rates, hash_rates = vae_utils.get_hash_PDs(vae, kin_var_norm, hand_vels, mdl.locmap()*0.05) # locations in mm


#%% compare hash PDs to single-unit PDs
PD_diff = vae_utils.circular_diff(hash_PDs,hand_vel_PDs)*180/np.pi
plt.figure()
plt.hist(PD_diff,40)

#%% visualize PDs
mapping = mdl.locmap().astype(int)
PD_bin_edges = np.linspace(-np.pi,np.pi,16)
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mapping)
PD_map = PD_map*180/np.pi
PD_hist, PD_bin_edges = np.histogram(hand_vel_PDs,bins=PD_bin_edges)

PD_bin_centers = PD_bin_edges[0:-1] + np.mean(np.diff(PD_bin_edges))/2
PD_hist= PD_hist/np.sum(PD_hist)

plt.figure()
ax=plt.subplot(111,polar=True)
plt.bar(x=PD_bin_centers,
        height=PD_hist,
        width=2*np.pi/len(PD_bin_centers),
        edgecolor='none',linewidth=0,color='#377eb8')
        

ax.set_ylim([0,0.15])
ax.set_yticks([0,0.05,0.1,0.15])

#1 =colors = '#e41a1c'   0 = colors = '#377eb8'   8 = colors = '#4daf4a'  26 = colors = '#65463E'
x = PD_map[:,:,0]
x[x<0] = x[x<0] + 360
plt.figure()
plt.imshow(x,cmap=cmasher.infinity,interpolation='none',vmin=0,vmax=360)
cbar=plt.colorbar()
cbar.set_label('Preferred Direction (Deg)')
cbar.set_ticks(np.array([0,90,180,270,360])) 
#cbar.ax.set_yticklabels(['360','270','180','90','0'])

plt.xlim([0,80])  # this manages to flip the map vertically....
plt.ylim([0,80])

#%% stimulate the same location(s) many times and measure effect
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
#input_data['amp_list'] = [0,10,20]
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
input_data['n_trials'] = 2
input_data['n_stim_chans'] = 1 # please don't change to more, I'm not sure if it works....
input_data['n_sets'] = 6

input_data['PD_tol'] = np.pi/8

single_loc_exp, loc_idx, amp, stim_exp_out, stim_chan_list= stim_exp_utils.run_single_location_exp(input_data)

exp_fname = 'single_loc_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([single_loc_exp,loc_idx,amp,input_data],f)
f.close()

#%% magnitude of effect, error bars across trials of the same location

plt.figure(figsize=(5,5))
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

mean_mag = np.zeros((input_data['n_sets'],len(input_data['amp_list'])))
std_mag = np.zeros_like(mean_mag)
offset = [-1,-0.5,0,0.5,1]*1

idx_look = 0
for i_set in range(input_data['n_sets']):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = single_loc_exp[idx_look][1]
        data = delta_mag_stim[:,0]/max_hand_vel_mag*100
        mean_mag[i_set,i_amp] = np.mean(data)
        std_mag[i_set,i_amp] = np.std(data)
        idx_look=idx_look+1
    plt.errorbar(np.array(input_data['amp_list'])+offset[i_set],mean_mag[i_set,:], std_mag[i_set,:], capsize=5,elinewidth=2)#, \
                     #color=colors[i_set], linestyle=ls, marker=m, markersize=ms,linewidth=2)

plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,10))
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')
#%% angular error

# histogram summary
plt.figure(figsize=(6,5))

mean_err = np.zeros((input_data['n_sets'],len(input_data['amp_list'])))
std_err = np.zeros_like(mean_mag)
offset = [-1,-0.5,0,0.5,1]*1
idx_look = 0
for i_set in range(input_data['n_sets']):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = single_loc_exp[idx_look][-2]
        pred_delta_dir = single_loc_exp[idx_look][-1]
        error_all = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))*180/np.pi
        mean_err[i_set,i_amp] = np.mean(error_all)
        std_err[i_set,i_amp] = np.std(error_all)
        idx_look=idx_look+1
    plt.errorbar(np.array(input_data['amp_list'])+offset[i_set],mean_err[i_set,:], std_err[i_set,:], capsize=5,elinewidth=2)#, \
                     #color=colors[i_set], linestyle=ls, marker=m, markersize=ms,linewidth=2)
plt.xlim([-5,85])
plt.ylim([0,180])
sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')
#%% run experiments for multiple amplitudes and number of electrodes
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
input_data['n_trials'] = 1000
input_data['n_stim_chans_list'] = [1]

input_data['PD_tol'] = np.pi/8

amp_elec_exp_out, n_chans, amp, stim_exp_out, stim_chan_list = stim_exp_utils.run_elec_amp_stim_exp(input_data)

exp_fname = 'amp_elec_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_elec_exp_out,n_chans,amp,input_data, stim_chan_list],f)
f.close()

#%% load data (assuming a break)
fname = project_folder+'\\amp_elec_exp_ratemult10_2021-09-18-124701.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_elec_exp_out = temp[0]
n_chans = temp[1]
amp = temp[2]
input_data = temp[3]
stim_chan_list = temp[4]

#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir

plt.figure()
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

data_to_plot = []
pos_to_plot = []
idx_look = 0
for i_chan in range(len(input_data['n_stim_chans_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_elec_exp_out[idx_look][1]
        data_to_plot.append(delta_mag_stim[:,0]/max_hand_vel_mag*100)
        pos_to_plot.append(input_data['amp_list'][i_amp]+i_chan)#-len(input_data['n_stim_chans_list'])/2)
        idx_look=idx_look+1
        
plt.boxplot(data_to_plot,positions=pos_to_plot,widths=1)


plt.xlabel('Amplitude')
plt.ylabel('Magnitude (% max speed)')


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

input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
input_data['freq_list'] = [200] # Hz

input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['n_pulses_list'] = [40]
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
input_data['n_trials'] = 1000
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

amp_freq_exp_out, freq, amp = stim_exp_utils.run_amp_freq_stim_exp(input_data)

exp_fname = 'amp_freq_exp'
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_freq_exp_out,freq,amp,input_data],f)
f.close()

#%% load data
fname = project_folder+'\\amp_freq_exp_ratemult10_nchans4_2021-09-16-131217.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_freq_exp_out = temp[0]
freq = temp[1]
amp = temp[2]
input_data = temp[3]

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir

plt.figure(figsize=(5,5))
offset = [-0.2,0,0.2]
idx_look = 0
colors = plt.cm.inferno([0,80,160])

for i_freq in range(len(input_data['freq_list'])):
    data_to_plot = []
    pos_to_plot = []
    err_to_plot = []
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_freq_exp_out[idx_look][1]/max_hand_vel_mag*100
        mean_mag_stim = np.mean(delta_mag_stim)
        std_mag_stim = np.std(delta_mag_stim)
        data_to_plot.append(mean_mag_stim)
        err_to_plot.append(std_mag_stim)
        pos_to_plot.append(input_data['amp_list'][i_amp] + offset[i_freq])
        idx_look=idx_look+1
        
    plt.errorbar(pos_to_plot,data_to_plot,err_to_plot,color=colors[i_freq,:])

#plt.legend(['50Hz','100Hz','200Hz'],loc='upper left')
plt.legend(['100 Hz'],loc='upper left')
plt.xlabel('Amplitude')
plt.ylabel('Magnitude (% high speed)')
sns.despine()

#%% histogram summary
plt.figure(figsize=(6,5))
bin_edges = np.arange(0,np.pi,np.pi/10)

colors = plt.cm.inferno([0,40,80,120,160,200,240])

ls = ['solid','dashed','dotted']
m = ['.','s','v']

idx_look = 0
for i_freq in range(1):#range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        if(i_amp in [0,1,2,4,6]):
            delta_dir_stim = amp_freq_exp_out[idx_look][-2]
            pred_delta_dir = amp_freq_exp_out[idx_look][-1]
        
            hist_vals,edges = np.histogram(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])),bin_edges)
            plt.plot((bin_edges[0:-1]+np.mean(np.diff(bin_edges))/2)*180/np.pi,hist_vals/delta_dir_stim.shape[0],color=colors[i_amp],linestyle=ls[i_freq],marker=m[i_freq],markersize=12)
        idx_look = idx_look+1
    
plt.xlim((0,180))
#plt.ylim([0,0.15])
plt.xlabel('Actual - Predicted Dir (deg)')
plt.ylabel('Proportion of trials')    
plt.legend(['0uA','5uA','10uA','20uA','80uA'],loc='lower left')
sns.despine()
#%% histogram summary -- only look at angle
# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;

prop_below_thresh = np.zeros((len(input_data['freq_list']),len(input_data['amp_list'])))
idx_look = 0
plt.figure(figsize=(6,5))
colors = plt.cm.inferno([0,40,80,120,160,200,240])

for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_freq_exp_out[idx_look][-2]
        pred_delta_dir = amp_freq_exp_out[idx_look][-1]
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_below_thresh[i_freq,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_freq,:],color=colors[i_freq])
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion predicted')            
#plt.legend(['50 Hz','100 Hz','200 Hz','500 Hz'])
plt.legend(['100 Hz'])
sns.despine()
#%% histogram summary -- require some minimum length
# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;
idx_base = [0,len(input_data['amp_list']),2*len(input_data['amp_list'])]
delta_mag_base = np.array([])
for i in idx_base:
    delta_mag_base_temp = amp_freq_exp_out[i][1][:,0]/max_hand_vel_mag*100
    delta_mag_base = np.append(delta_mag_base,delta_mag_base_temp)

min_mag = np.percentile(delta_mag_base,95)

prop_below_thresh = np.zeros((len(input_data['freq_list']),len(input_data['amp_list'])))
prop_above_min_mag = np.zeros_like(prop_below_thresh)
idx_look = 0
plt.figure(figsize=(6,5))
colors = plt.cm.inferno([0,80,160])


for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_freq_exp_out[idx_look][-2]
        pred_delta_dir = amp_freq_exp_out[idx_look][-1]
        delta_mag_stim = amp_freq_exp_out[idx_look][1][:,0]/max_hand_vel_mag*100        
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_above_min_mag[i_freq,i_amp] = np.sum(delta_mag_stim >= min_mag)/len(delta_mag_stim) # 1-prop no effect
        ang_diff[delta_mag_stim < min_mag] = 1000
        
        prop_below_thresh[i_freq,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        #prop_below_thresh[i_freq,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff) # proportion predictable
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_above_min_mag[i_freq,:]+prop_below_thresh[i_freq,:],color=colors[i_freq],linestyle='-')
    #plt.plot(input_data['amp_list'],0.5*(prop_above_min_mag[i_freq,:]),color=colors[i_freq],linestyle='--')
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['50 Hz','100 Hz','200 Hz','500 Hz'])

#%% histogram summary -- project onto PD, require some minimum length
# proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;
idx_base = 0
delta_mag_base = amp_freq_exp_out[idx_base][1][:,0]/max_hand_vel_mag*100
delta_dir_stim = amp_freq_exp_out[idx_base][-2]
pred_delta_dir = amp_freq_exp_out[idx_base][-1]
proj_val = delta_mag_base*np.cos(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])))
min_proj = np.percentile(proj_val,95)

prop_below_thresh = np.zeros((len(input_data['freq_list']),len(input_data['amp_list'])))

idx_look = 0
plt.figure()
colors = plt.cm.inferno([0,40,80,120,160,200,240])

for i_freq in range(len(input_data['freq_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_freq_exp_out[idx_look][1][:,0]/max_hand_vel_mag*100
        delta_dir_stim = amp_freq_exp_out[idx_look][-2]
        pred_delta_dir = amp_freq_exp_out[idx_look][-1]
        
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        proj_val = delta_mag_stim*np.cos(ang_diff)

        
        prop_below_thresh[i_freq,i_amp] = np.sum(proj_val > min_proj)/len(ang_diff)
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_freq,:],color=colors[i_freq])
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['50 Hz','100 Hz','200 Hz','500 Hz'])

#%% plot magnitude vs angular error for each

idx_look = 16
print(freq[idx_look],amp[idx_look])
delta_dir_stim = amp_freq_exp_out[idx_look][-2]
pred_delta_dir = amp_freq_exp_out[idx_look][-1]
delta_mag_stim = amp_freq_exp_out[idx_look][1]/max_hand_vel_mag*100
ang_err = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        
plt.figure()
plt.plot(ang_err,delta_mag_stim,'.')

#%% something about a detection threshold
# compare delta-vel-mag for same trials. Use 0 amp as base. 
idx_look = 0
delta_mag_stim_0 = amp_freq_exp_out[0][1]
plt.figure()
ax = plt.axes()
colors = plt.cm.inferno([0,80,160])

for i_freq in range(len(input_data['freq_list'])):
    pos_to_plot = []
    data_to_plot = []
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_freq_exp_out[idx_look][1]
        mag_diff = (delta_mag_stim[:,0]-delta_mag_stim_0[:,0])/max_hand_vel_mag*100
        data_to_plot.append(np.sum(mag_diff > 0)/len(mag_diff))
        pos_to_plot.append(input_data['amp_list'][i_amp])#-len(input_data['n_stim_chans_list'])/2)
        idx_look=idx_look+1

    plt.plot(pos_to_plot[1:],data_to_plot[1:],color=colors[i_freq])


plt.xlim([0,100])
plt.ylim([0.4,1.0])
plt.xticks(ticks=np.arange(0,110,10))
plt.yticks(ticks=np.arange(0.4,1.1,0.1))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

sns.despine()
            

#%% run single electrode experiment where we pick locations with high neighbor similarity

PD_sim_mat = (np.pi - np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1))))/np.pi

neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)
neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))

plt.figure()
plt.imshow(neigh_map[:,:,0])

#%%
input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,20,40] # uA
input_data['step_list'] = [0,1,2,4,8]
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
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 500
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

input_data['sim_score'] = neigh_sim
input_data['sim_tol'] = np.percentile(neigh_sim,85) #

amp_high_sim_exp_out, amp, step, stim_chan_list = stim_exp_utils.run_amp_high_sim_exp(input_data)

#%% save data
exp_fname= 'amp_highsim_exp'
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


#%% get PD similarity vs. distance for neurons on and not-on the same electrode
dists, pd_diffs = vae_utils.get_pd_dist(mdl.locmap(), hand_vel_PDs) # in blocks



#%%
non_neigh_pd_diff = pd_diffs[dists>3]
neigh_pd_diff = pd_diffs[dists<=1]
plt.figure()
plt.hist(non_neigh_pd_diff)
plt.figure()
plt.hist(neigh_pd_diff)

#%% run experiments for multiple amplitudes at different block sizes

input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
input_data['block_size_list'] = [0.03,0.05,0.067,0.08,0.1,0.2] # mm


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

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 500
input_data['n_stim_chans'] = 4

input_data['PD_tol'] = np.pi/8

amp_blocksize_exp_out, block_size, amp, stim_exp_out = stim_exp_utils.run_blocksize_amp_stim_exp(input_data)

exp_fname = 'amp_blocksize_exp'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_blocksize_exp_out,n_chans,amp,input_data],f)
f.close()

#%% compare metrics across amps and plot
# amp_elec_exp_out: For each condition: delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir

plt.figure()
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

data_to_plot = []
pos_to_plot = []
idx_look = 0
for i_size in range(len(input_data['block_size_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = amp_blocksize_exp_out[idx_look][1]
        data_to_plot.append(delta_mag_stim[:,0]/max_hand_vel_mag*100)
        pos_to_plot.append(input_data['amp_list'][i_amp]+i_size)#-len(input_data['n_stim_chans_list'])/2)
        idx_look=idx_look+1
        
plt.boxplot(data_to_plot,positions=pos_to_plot,widths=1)


plt.xlabel('Amplitude')
plt.ylabel('Magnitude (% max speed)')

#%% proportion of trials within some bound for different amplitudes (each line is a number of channels)
ang_thresh = np.pi/2;

prop_below_thresh = np.zeros((len(input_data['block_size_list']),len(input_data['amp_list'])))
plt.figure()
ls = ['solid','dashed','dotted']
m = ['.','s','v']
colors = plt.cm.inferno([0,40,80,120,160,200,240])

idx_look = 0
for i_size in range(len(input_data['block_size_list'])):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = amp_blocksize_exp_out[idx_look][-2]
        pred_delta_dir = amp_blocksize_exp_out[idx_look][-1]
        ang_diff = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))
        prop_below_thresh[i_size,i_amp] = np.sum(ang_diff<ang_thresh)/len(ang_diff)
        idx_look = idx_look+1
    plt.plot(input_data['amp_list'],prop_below_thresh[i_size,:],color=colors[i_size],linestyle=ls[0],marker=m[0],markersize=12)
    
plt.xlabel('Amplitude')
plt.ylabel('Proportion of trials within 90 deg of prediction')            
plt.legend(['30um','50um','67um','80um','100um','200um'])



#%%

# 
good_locs = np.array([[23,46,14,27,11,10],[48,32,12,16,39,39]])
good_idx = vae_utils.convert_loc_to_idx(good_locs,mdl.locmap().astype(int))

print(neigh_sim[good_idx.astype(int)])
#%%
# run stimulation trial
# set stim params

kin_var_samp = kin_var_norm[20:50] # use normalized joint velocity for vae, actual joint velocity for decoder and opensim

#kin_var_samp = kin_var_norm[20,:].reshape((1,-1))
#kin_var_use = np.tile(kin_var_samp,(30,1))

input_data = {}

input_data['stim_chan'] = np.array([2525]) # 2525 = good, 3262 = bad
input_data['amp'] = 10 # uA
input_data['freq'] = 200 # Hz
input_data['n_pulses'] = 40 # s
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 0

input_data['trans_func'] = 'none'
input_data['vae'] = vae

input_data['joint_vel'] = kin_var_samp
input_data['init_joint_ang'] = joint_angs[0]
input_data['dec'] = dec
input_data['kin'] = 'hand'
input_data['vae_dec'] = vae_dec

input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

# [rates, samp_rates_stim, is_act, hand_vel_true, hand_vel_stim]
stim_out = stim_exp_utils.run_stim_trial(input_data)
input_data['amp']=0
no_stim_out = stim_exp_utils.run_stim_trial(input_data)

no_stim_rates = no_stim_out[1]
stim_rates = stim_out[1]
is_act = stim_out[2]
is_act_once = np.any(is_act,axis=1)

point_kin_no = 100*no_stim_out[-1] # [hand_pos,hand_vel,hand_acc,elbow_pos,elbow_vel,elbow_acc]
point_kin_stim = 100*stim_out[-1]
no_stim_rates_map, stim_rates_map = vae_utils.visualize_activation_map(stim_rates, no_stim_rates, idx=1)

#vae_utils.visualize_activation_dist(is_act, input_data['stim_chan'],input_data['block_size'])

#%% set global plot style
sns.set_style('ticks') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('font', size=14)          # controls default text sizes

plt.figure(figsize=(5,5))
ax=plt.axes()
idx = 1

plt.plot(point_kin_no[idx-1:idx+6,0],point_kin_no[idx-1:idx+6,1],color='black', marker='.',markersize=12, linewidth=2)  # no stim 
plt.plot(point_kin_stim[idx-1:idx+6,0],point_kin_stim[idx-1:idx+6,1],color='tab:red', marker='.', markersize=12, linewidth=2) # stim

sns.despine()

#plt.xlim([-12,4])
#plt.ylim([-4,12])
#plt.xticks(ticks=np.arange(-12,6,4))
#plt.yticks(ticks=np.arange(-4,14,4))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.xlabel('X Hand-vel (cm/s)')     
plt.ylabel('Y Hand-vel (cm/s)')
            
#%% use rates to get stim effect vector
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,95)

kin_var_no = vae_utils.vae_decoder(vae=input_data['vae'],samples=no_stim_rates,bin_size=gl.bin_size)
kin_var_stim = vae_utils.vae_decoder(vae=input_data['vae'],samples=stim_rates,bin_size=gl.bin_size)
        
hand_vel_no = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_no)
hand_vel_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_stim)

stim_eff = np.mean(hand_vel_stim[1:5,:] - hand_vel_no[1:5,:],axis=0) # magnitude is meaningless here

#%% plot contribution of each neuron to stim-effect vector

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

stim_eff = np.zeros((6400,2))
r=[]
theta=[]
theta_PD=[]
x_data = []
y_data = []
fig = plt.figure(figsize=(4,4))
ax=fig.gca()
lims = 0.6
# plot circle 
ax.add_patch(plt.Circle((0,0),lims,color='black',fill=False))
plt.plot([-lims,lims],[0,0],color='grey',linewidth=1)
plt.plot([0,0],[-lims,lims],color='grey',linewidth=1)

for i in range(6400):
    if(is_act_once[i]==True):
        no_rates_use = np.zeros_like(no_stim_rates)
        no_rates_use[:,i] = no_stim_rates[:,i]
        
        stim_rates_use = np.zeros_like(stim_rates)
        stim_rates_use[:,i] = stim_rates[:,i]
                
        kin_var_no = vae_utils.vae_decoder(vae=input_data['vae'],samples=no_rates_use,bin_size=gl.bin_size)
        kin_var_stim = vae_utils.vae_decoder(vae=input_data['vae'],samples=stim_rates_use,bin_size=gl.bin_size)
        
        hand_vel_no = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_no)
        hand_vel_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_stim)
    
        stim_eff[i,:] = np.mean(hand_vel_stim[1:5,:] - hand_vel_no[1:5,:],axis=0)
    
        x_data.append(stim_eff[i,0])
        y_data.append(stim_eff[i,1])
        theta.append(np.arctan2(stim_eff[i,1],stim_eff[i,0]))
        r.append(100*np.sqrt(np.sum(np.square(stim_eff[i,:])))/max_hand_vel_mag)
        theta_PD.append(hand_vel_PDs[i])
            
        #ax.plot(theta,r,color='black',alpha=random.uniform(0.25,1))
        
        plt.plot([0,r[-1]*np.cos(theta[-1])],[0,r[-1]*np.sin(theta[-1])], \
                  linewidth=0.5, color='black')
        #plt.arrow(0,0,r[-1]*np.cos(theta[-1]),r[-1]*np.sin(theta[-1]), \
        #          width=0.005,head_width=0.02,head_length=0.035,edgecolor='none',facecolor='black')
 
    
ax.set_xticks(ticks=[])
ax.set_yticks(ticks=[])
plt.xlim([-lims,lims])
plt.ylim([-lims,lims])
ax.set_aspect('equal')



bin_edges = np.linspace(-np.pi,np.pi,20)

r_plot, bin_edges = np.histogram(theta,bin_edges,weights=r)

bin_centers = bin_edges[0:-1] + np.mean(np.diff(bin_edges))/2

plt.figure(figsize=(4,4))
ax=plt.subplot(111,polar=True)
plt.bar(x=bin_centers,
        height=r_plot,
        width=2*np.pi/len(bin_centers),
        edgecolor='none',linewidth=0,color='#377eb8')
        
        
ax.set_yticklabels(labels=[])
ax.set_xticklabels(labels=[])
ax.set_aspect('equal')
ax.set_rlim([0,4])
#ax.set_ylim([0,8])
#ax.set_yticks([0,0.05,0.1,0.15])
#%        
        
#%%



"""
#%% retrain vae decoder and plot weights vs. PD

rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)

dec_weights = vae.decoder.state_dict()['layer1.0.weight'].numpy()
vae_dec_new = vae_utils.make_linear_decoder(x=rates, y=kin_var_norm, drop_rate=0.995,n_iters=2000,lr=0.001,init_weights=dec_weights)   
hand_dec_weights = dec.state_dict()['layer1.0.weight'].numpy()
whole_decoder_weights = np.matmul(np.transpose(vae_dec_new.state_dict()['layer1.0.weight'].numpy()),np.transpose(hand_dec_weights))
whole_decoder_dir = np.arctan2(whole_decoder_weights[:,1],whole_decoder_weights[:,0])

PD_err = vae_utils.circular_diff(whole_decoder_dir, hand_vel_PDs)
    
plt.figure()
plt.hist(np.abs(PD_err)*180/np.pi,50)   

weight_mag=np.sqrt(np.sum(np.square(whole_decoder_weights),axis=1))
plt.figure()
plt.plot(whole_decoder_dir, np.sqrt(np.sum(np.square(whole_decoder_weights),axis=1)),'.')

# save retrained vae decoder
f=open(project_folder + '\\retrained_vae_dec.pkl','wb')
pickle.dump(vae_dec_new,f)
f.close()
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