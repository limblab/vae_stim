# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:22:40 2021

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

# load a vae_decoder if we retrained it
path_to_dec = glob.glob(project_folder + '\\*retrained_vae_dec*')
if(len(path_to_dec)>0):
    f=open(path_to_dec[0],'rb')
    vae_dec = pickle.load(f)
    f.close()
    print("loaded retrained vae dec; use input_data['vae_dec']=vae_dec")

# load constrained decoder
path_to_dec = glob.glob(project_folder + '\\*constrained_straight_to_hand*')
if(len(path_to_dec)>0):
    f=open(path_to_dec[0],'rb')
    constrained_dec = pickle.load(f)
    f.close()
    print("loaded constrained dec; use input_data['dec']=constrained_dec; input_data['kin']='straight_to_hand_constrained'")

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
    
# get PD hash matrix if it's present
path_to_PDs = glob.glob(project_folder + '\\*PD_multiunit_hash_pow2*')
if(len(path_to_PDs)>0):
    f=open(path_to_PDs[0],'rb')
    hash_PDs = pickle.load(f)[0]
    #hash_params = temp[1]
    #rates = temp[2]
    #hash_rates = temp[3]
    #res = temp[4]
    f.close()
    
#%%
#%% load data (assuming a break)
#%% measure standard deviation of angular error, plot as histogram for provided amplitude(s)

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\multi_loc_blocksize*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data = temp[3]

# get std values across folders
std_ang_err_single = np.zeros((len(fnames),input_data['n_sets'],len(input_data['amp_list']))) # num electrodes, n_sets, n_amps
mean_ang_err_single = np.zeros_like(std_ang_err_single)
mean_mag_single = np.zeros_like(std_ang_err_single)

ang_all = np.zeros_like(mean_ang_err_single)
pred_ang_all = np.zeros_like(mean_ang_err_single)
old_pred_ang_all = np.zeros_like(pred_ang_all)

for i_folder in range(len(fnames)):
    idx_look = 0
    f=open(fnames[i_folder],'rb')
    temp = pickle.load(f)
    f.close()
    
    single_loc_exp = temp[0]
    loc_idx = temp[1]
    amp = temp[2]
    input_data = temp[3]
    stim_chan_list = temp[4]
    
    for i_set in range(input_data['n_sets']):
           
        for i_amp in range(len(input_data['amp_list'])):
            # angular error data
            delta_dir_stim = single_loc_exp[idx_look][-2]
            pred_delta_dir = single_loc_exp[idx_look][-1]
            
            ang_all[i_folder,i_set,i_amp] = sp.stats.circmean(delta_dir_stim[:,0])
            pred_ang_all[i_folder,i_set,i_amp]=sp.stats.circmean(pred_delta_dir,low=-np.pi,high=np.pi)
                        
            error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_ang_all[i_folder,i_set,i_amp])

            mean_ang_err_single[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
            std_ang_err_single[i_folder,i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi

            # magnitude data
            delta_mag_stim = single_loc_exp[idx_look][1]
            data = delta_mag_stim[:,0]/max_hand_vel_mag*100
            mean_mag_single[i_folder,i_set,i_amp] = np.mean(data)
            
            idx_look = idx_look + 1
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

#%% magnitude of effect, error bars across trials of the same location
offset = np.array([-0.75,-0.25,0.25,0.75])*1.

colors = ['black', '#08519c', '#3182bd', '#6baed6']

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])
alpha_list= [1.0,1.0,1.0,1.0,1.0]
ls_list = ['-','-','-','-']
marker_list = ['.','.','.','.']
ms_list = [16,16,16,16]
multiplier = [28/40, 1, 52/40, 64/40]
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

counter = 0
for i_elec in range(mean_mag_single.shape[0]):
    # single PD
    mean_to_plot = np.mean(mean_mag_single[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_mag_single[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_to_plot*multiplier[i_elec], std_to_plot, capsize=5,elinewidth=1, \
                     color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    counter=counter+1
    # multi PD
    #mean_std =np.mean(mean_mag_multi[i_elec,:,:],axis=0)
    #std_std = np.std(mean_mag_multi[i_elec,:,:],axis=0)
    #plt.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_std,std_std, capsize=5,elinewidth=1, \
    #             color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    
plt.xlim([-5,85])
plt.xticks(ticks=np.arange(0,85,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
plt.ylim([0,4])

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')
# angular error

offset = np.array([-0.75,-0.25,0.25,0.75])*1.

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])


counter = 0
for i_elec in range(mean_ang_err_single.shape[0]):
    # single PD
    mean_to_plot = np.mean(mean_ang_err_single[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_ang_err_single[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[i_elec],mean_to_plot, std_to_plot, capsize=5,elinewidth=1, \
                     color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    counter = counter + 1
    # multi PD
    #mean_std =np.mean(mean_ang_err_multi[i_elec,:,:],axis=0)
    #std_std = np.std(mean_ang_err_single[i_elec,:,:],axis=0)
    #plt.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_std,std_std, capsize=5,elinewidth=1, \
    #             color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec], \
    #             markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,85])
plt.ylim([0,160])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,180,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))

#plt.legend(['2800 (' + u"\u03bcm" + ')', '4000 (' + u"\u03bcm" + ')', '5200 (' + u"\u03bcm" + ')', '6400 (' + u"\u03bcm" + ')'],loc="upper left")
#%% magnitude statistics
mag = np.array([])
total_curr = np.array([])
map_size = np.array([])

multiplier = [28/40, 1, 52/40, 64/40]
map_size_list = [2800,4000,5200,6400]
for i_amp in range(1,mean_mag_single.shape[-1]): # don't include 0 uA condition
    for i_size in range(mean_mag_single.shape[0]):
        mag = np.append(mag, mean_mag_single[i_size,:,i_amp]*multiplier[i_size])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp]*4,mean_mag_single.shape[1]))
        map_size = np.append(map_size, np.tile(map_size_list[i_size], mean_mag_single.shape[1]))

df = pd.DataFrame({'mag' : mag, 'total_curr' : total_curr, 'map_size' : map_size})

# perform two-way anova
model_anova = ols('mag ~ total_curr + map_size', data=df).fit()
# model_anova.summary() to get overal model statistic
mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = mag
X = np.transpose(np.vstack((total_curr,map_size)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)


#%% angular error statistics
err = np.array([])
total_curr = np.array([])
map_size = np.array([])

map_size_list = [2800,4000,5200,6400]
for i_amp in range(1,5): #mean_mag_single.shape[-1]): # don't include 0 uA condition
    for i_size in range(mean_mag_single.shape[0]):
        err = np.append(err, mean_ang_err_single[i_size,:,i_amp])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp]*4,mean_mag_single.shape[1]))
        map_size = np.append(map_size, np.tile(map_size_list[i_size], mean_mag_single.shape[1]))

df = pd.DataFrame({'err' : err, 'total_curr' : total_curr, 'map_size' : map_size})

# perform two-way anova
model_anova = ols('err ~ total_curr + map_size', data=df).fit()
# model_anova.summary() to get overal model statistic
mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = err
X = np.transpose(np.vstack((total_curr,map_size)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)


















