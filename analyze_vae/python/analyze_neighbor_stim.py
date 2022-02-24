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
import cmasher

import pickle
import datetime
import torch
%matplotlib qt

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import random

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

path_to_PDs = glob.glob(project_folder + '\\*PD_multiunit_hash_pow2*')
if(len(path_to_PDs)>0):
    f=open(path_to_PDs[0],'rb')
    hand_vel_PDs = pickle.load(f)[0]
    f.close()
    
else:
    hand_vel_PDs, hand_vel_params = vae_utils.get_PD_similarity(vae,kin_var_norm,hand_vels)
    
    f=open(project_folder + '\\PD_calc.pkl','wb')
    pickle.dump(hand_vel_PDs,f)
    f.close()    
    
    
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)
    
#%% compute neigh sim
    
PD_sim_mat = np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1)))/np.pi*180
neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)

neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mdl.locmap().astype(int))

#plt.figure()
#plt.imshow(neigh_map[:,:,0],interpolation='none')
#plt.xlim([37,53])
#plt.ylim([27,43])

fig=plt.figure()
ax=fig.gca()
im=plt.imshow(PD_map[:,:,0]*180/np.pi,cmap=cmasher.infinity,vmin=-180,vmax=180)
cbar = fig.colorbar(im)
cbar.set_label('Preferred Direction (Deg)')
cbar.set_ticks(np.array([-180,-90,0,90,180]))

plt.xlim([52,77])
plt.ylim([46,71])

ax.set_aspect('equal')
ax.set_xticks(ticks=[])
ax.set_yticks(ticks=[])

#%%

input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec

input_data['amp_list'] = [0,5,10,15,20]
input_data['step_list'] = [0,1,2,3,4]
input_data['freq'] = 200 # Hz

input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['n_pulses'] = 40
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 0
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()

input_data['block_size'] = 0.05 # mm
input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 20
input_data['n_stim_chans'] = 1
input_data['n_sets'] = 100

input_data['PD_tol'] = np.pi/8

input_data['sim_score'] = neigh_sim
input_data['sim_tol'] = np.percentile(neigh_sim,10) # only use similarities below this

amp_high_sim_exp_out, amp, step, stim_chan_list, set_list = stim_exp_utils.run_amp_high_sim_repeat_exp(input_data)

exp_fname= 'amp_highsim_1chan_exp'

# save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+'_multiunitPDs' + dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_high_sim_exp_out,amp,step,stim_chan_list,neigh_sim,input_data,set_list],f)
f.close()

#%% measure and organize magnitude and angular error across sets -- LOAD DATA for 1 chan stim

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\amp_highsim_1chan_exp_*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data = temp[5]
stim_chans_all = np.zeros((len(fnames)*input_data['n_sets'],len(input_data['step_list'])))

# get values across folders
std_ang_err = np.zeros((len(fnames)*input_data['n_sets'],len(input_data['amp_list']),len(input_data['step_list']))) # num electrodes, n_sets, n_amps, n_steps
mean_ang_err = np.zeros_like(std_ang_err)
mean_mag = np.zeros_like(std_ang_err)

for i_folder in range(len(fnames)):
    idx_look = 0
    f=open(fnames[i_folder],'rb')
    temp = pickle.load(f)
    f.close()
    
    amp_high_sim_exp_out = temp[0]
    freq = temp[1]
    amp = temp[2]
    input_data = temp[5]
    set_list = temp[6]
    
    for i_set in range(input_data['n_sets']):
        for i_step in range(len(input_data['step_list'])):
            for i_amp in range(len(input_data['amp_list'])):
                # angular error data
                delta_dir_stim = amp_high_sim_exp_out[idx_look][-2]
                pred_delta_dir = amp_high_sim_exp_out[idx_look][-1]
                error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
                mean_ang_err[i_folder*input_data['n_sets'] + i_set,i_amp,i_step] = abs(error_all)*180/np.pi
                std_ang_err[i_folder*input_data['n_sets'] + i_set,i_amp,i_step] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
            
                
                # magnitude data
                delta_mag_stim = amp_high_sim_exp_out[idx_look][1]
                data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
                mean_mag[i_folder*input_data['n_sets'] + i_set,i_amp,i_step] = np.mean(data)
                
                # stim chans
                if(i_amp==0): # chans are the same for all amps
                    stim_chans_all[i_folder*input_data['n_sets'] + i_set,i_step] = temp[3][idx_look][0]
                
                idx_look = idx_look + 1

#%% measure and organize magnitude and angular error across sets -- LOAD DATA for 4 chan stim

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\amp_highsim_4chans_exp_*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data_4 = temp[5]
stim_chans_all_4 = np.zeros((len(fnames)*input_data_4['n_sets'],4))

# get values across folders
std_ang_err_4 = np.zeros((len(fnames)*input_data_4['n_sets'],len(input_data_4['amp_list']))) # num electrodes, n_sets, n_amps, n_steps
mean_ang_err_4 = np.zeros_like(std_ang_err_4)
mean_mag_4 = np.zeros_like(std_ang_err_4)

for i_folder in range(len(fnames)):
    idx_look = 0
    f=open(fnames[i_folder],'rb')
    temp = pickle.load(f)
    f.close()
    
    amp_high_sim_exp_out = temp[0]
    freq = temp[1]
    amp = temp[2]
    input_data_4 = temp[5]
    set_list = temp[6]
    
    for i_set in range(input_data_4['n_sets']):
        for i_amp in range(len(input_data_4['amp_list'])):
            # angular error data
            delta_dir_stim = amp_high_sim_exp_out[idx_look][-2]
            pred_delta_dir = amp_high_sim_exp_out[idx_look][-1]
            error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
            mean_ang_err_4[i_folder*input_data_4['n_sets'] + i_set,i_amp] = abs(error_all)*180/np.pi
            std_ang_err_4[i_folder*input_data_4['n_sets'] + i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
        
            
            # magnitude data
            delta_mag_stim = amp_high_sim_exp_out[idx_look][1]
            data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
            mean_mag_4[i_folder*input_data_4['n_sets'] + i_set,i_amp] = np.mean(data)
            
            # stim chans
            if(i_amp==0): # chans are the same for all amps
                stim_chans_all_4[i_folder*input_data_4['n_sets'] + i_set,:] = temp[3][idx_look][0]
            
            idx_look = idx_look + 1

#%% magnitude of effect, error bars across different locations
#colors = plt.cm.inferno([0,50,100,150,200,250])

colors = ['#9ecae1','#6baed6','#3182bd','#08519c','black']
colors.reverse()
offset = np.array([-1,-0.6,-0.2,0.6,1.0])*0.75
fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])


# plot single electrode result
for i_step in range(mean_mag.shape[2]):
    mean_to_plot = np.mean(mean_mag[:,:,i_step],axis=0) # mean over sets
    std_to_plot = np.std(mean_mag[:,:,i_step],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_step],mean_to_plot, std_to_plot, capsize=5,elinewidth=1.25, \
                     color=colors[i_step],linewidth=1.5,marker='s',markersize=8)

# plot 4 chan result
mean_to_plot = np.mean(mean_mag_4,axis=0)
std_to_plot = np.std(mean_mag_4,axis=0)
ax.errorbar(np.array(input_data['amp_list'])+0.2,mean_to_plot, std_to_plot, capsize=5,elinewidth=1.25, \
                     color='black',linewidth=1.5,marker='.',markersize=20,linestyle='--')
        
        
plt.xlim([-5,25])
plt.ylim([0,4.25])
sns.despine()
plt.xticks(ticks=np.arange(0,25,10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')

# angular error, error bars across different locations
fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

for i_step in range(mean_mag.shape[2]):
    mean_to_plot = np.mean(mean_ang_err[:,:,i_step],axis=0) # mean over sets
    std_to_plot = np.std(mean_ang_err[:,:,i_step],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_step],mean_to_plot, std_to_plot, capsize=5,elinewidth=1.25, \
                     color=colors[i_step],linewidth=1.5,marker='s',markersize=8)

# plot 4 chan result
mean_to_plot = np.mean(mean_ang_err_4,axis=0)
std_to_plot = np.std(mean_ang_err_4,axis=0)
ax.errorbar(np.array(input_data['amp_list'])+0.2,mean_to_plot, std_to_plot, capsize=5,elinewidth=1.25, \
                     color='black',linewidth=1.5,marker='.',markersize=20,linestyle='--')

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,25])
plt.ylim([0,160])
sns.despine()
plt.xticks(ticks=np.arange(0,25,10))
plt.yticks(ticks=np.arange(0,180,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))

#%% magnitude statistics
mag = np.array([])
total_curr = np.array([])
dist = np.array([])

dist_list = [0,50,100,150,200]

for i_amp in range(1,mean_mag.shape[1]): # don't include 0 uA condition
    for i_dist in range(mean_mag.shape[2]):
        mag = np.append(mag, mean_mag[:,i_amp,i_dist])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp],mean_mag.shape[0]))
        dist = np.append(dist, np.tile(dist_list[i_dist], mean_mag.shape[0]))

df = pd.DataFrame({'mag' : mag, 'total_curr' : total_curr, 'dist' : dist})

# perform two-way anova
model_anova = ols('mag ~ total_curr + dist', data=df).fit()
# model_anova.summary() to get overal model statistic

mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = mag
X = np.transpose(np.vstack((total_curr,dist)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)

#%% ang-err statistics
err = np.array([])
total_curr = np.array([])
dist = np.array([])

dist_list = [0,50,100,150,200]

for i_amp in range(1,mean_ang_err.shape[1]): # don't include 0 uA condition
    for i_dist in range(mean_ang_err.shape[2]):
        err = np.append(err, mean_ang_err[:,i_amp,i_dist])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp],mean_ang_err.shape[0]))
        dist = np.append(dist, np.tile(dist_list[i_dist], mean_ang_err.shape[0]))

df = pd.DataFrame({'err' : err, 'total_curr' : total_curr, 'dist' : dist})

# perform two-way anova
model_anova = ols('err ~ total_curr + dist', data=df).fit()
# model_anova.summary() to get overal model statistic

mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = err
X = np.transpose(np.vstack((total_curr,dist)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)

#%% percentage distribution overlap for two conditions

amp_idx = 1
step_idx = 0
data1 = mean_ang_err[:,0,step_idx] # null condition
data2 = mean_ang_err[:,amp_idx,step_idx]
perc_over, data1_hist,data2_hist, bin_edges = vae_utils.compute_histogram_overlap(data1, data2 ,bin_size=5)


print(perc_over)

t_stat,p = sp.stats.ranksums(data1,data2)
print(p)


#%% neighborhood similary as a function of steps, but using the actual stim channels not just random ones

# stim_chans_all : # n_sets, n_steps, stim_idx
mean_sim = np.zeros((stim_chans_all.shape[1],))
std_sim = np.zeros_like(mean_sim)

# plot single channel result
for i_step in range(stim_chans_all.shape[1]):
    sim_data = neigh_sim[stim_chans_all[:,i_step].astype(int)].reshape((-1,))
    mean_sim[i_step] = np.mean(sim_data)
    std_sim[i_step] = np.std(sim_data)
    
fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.65,0.5])
offset = [6,0,0,0,0,0]

for i_step in range(stim_chans_all.shape[1]):
    plt.errorbar(input_data['step_list'][i_step]*50 + offset[i_step],mean_sim[i_step],std_sim[i_step], capsize=5,elinewidth=2, \
                         color=colors[i_step],linewidth=2,marker='s',markersize=8)
   
# plot 4 channel result
sim_data = neigh_sim[stim_chans_all_4.astype(int)].reshape((-1,))
mean_sim_4 = np.mean(sim_data)
std_sim_4 = np.std(sim_data)
    
for i_step in range(stim_chans_all.shape[1]):
    plt.errorbar(input_data['step_list'][0]*50-6,mean_sim_4, std_sim_4, capsize=5,elinewidth=2, \
                         color='black',linewidth=2,marker='.',markersize=20,linestyle='--')    
    
plt.xlabel('Distance moved (' + u"\u03bcm" + ')')
plt.ylabel('Neighborhood PD error (deg)')

plt.ylim([0,90])
sns.despine()
plt.xticks(ticks=np.arange(0,250,50))
plt.yticks(ticks=np.arange(0,100,20))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(5))






