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

#%% make a decoder based on the PDs of neurons
rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)
PDs_as_weights = np.vstack((np.cos(hand_vel_PDs), np.sin(hand_vel_PDs)))/6400

# normalize PDs_as_weights based on PD distribution
PD_distribution, bin_edges = np.histogram(hand_vel_PDs,20)
bin_idx = np.digitize(hand_vel_PDs, bin_edges) - 1
bin_idx[bin_idx < 0] = 0
bin_idx[bin_idx >= len(PD_distribution)] = len(PD_distribution)-1
n_bin = PD_distribution[bin_idx]

PDs_as_weights = PDs_as_weights/n_bin

straight_to_hand_dec = vae_utils.make_linear_decoder(x=rates, y=hand_vels, drop_rate=0.,n_iters=1,lr=0.01)
straight_to_hand_dec.state_dict()['layer1.0.weight'][:] = torch.Tensor(PDs_as_weights)
    
#%% stimulate the same location(s) many times and measure effect -- measure consistency across many locations

input_data = {}
input_data['kin'] = 'hand';
input_data['dec'] = dec
input_data['vae_dec'] = vae_dec
#input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
input_data['amp_list'] = [0,5,10,20,80]
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 200 # Hz
input_data['n_pulses'] = 40 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 0
input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 20
input_data['n_stim_chans'] = 4 # can't be a list
input_data['n_sets'] = 200

input_data['PD_tol'] = np.pi/8

single_loc_exp, loc_idx, amp, stim_exp_out, stim_chan_list = stim_exp_utils.run_single_location_exp(input_data)

exp_fname = 'multi_loc_PDTEST' # 35, 50, 65, 80

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+'_multiUnitPDs' + dname+'.pkl'
f=open(fname,'wb')
pickle.dump([single_loc_exp,loc_idx,amp,input_data,stim_chan_list],f)
f.close()

#%% save pred angle

x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\predicted_angle_elec_space2.pkl'
f=open(fname,'wb')
pickle.dump([pred_ang_all],f)
f.close()

#%% load data (assuming a break)
#%% measure standard deviation of angular error, plot as histogram for provided amplitude(s)
colors = ['#680C0E','#E52125']


act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\multi_loc_PDTEST*')

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

# prep data for multi-elec predict
#chan_loc = mdl.locmap()

#pred_name = glob.glob(project_folder+'\\predicted_angle_elec_space8.pkl')
#f=open(pred_name[0],'rb')
#pred_ang_all = pickle.load(f)[0]
#f.close()


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
        if(np.mod(i_set,20)==0):
            print(i_set)
            
        for i_amp in range(len(input_data['amp_list'])):
            # angular error data
            delta_dir_stim = single_loc_exp[idx_look][-2]
            pred_delta_dir = single_loc_exp[idx_look][-1]
            
            ang_all[i_folder,i_set,i_amp] = sp.stats.circmean(delta_dir_stim[:,0])
            pred_ang_all[i_folder,i_set,i_amp]=sp.stats.circmean(pred_delta_dir,low=-np.pi,high=np.pi)
            
            #pred_ang_all[i_folder,i_set,i_amp], grid_idx, dist_all, PD_all = vae_utils.compute_multielec_pred(chan_loc[stim_chan_list[idx_look][0].astype(int),:],mdl.locmap(),amp[idx_look],hand_vel_PDs,elec_space=2) # dist is in blocks, each block is 50 um
            
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

"""
# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\multi_loc_repetition_exp_multiUnit*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data = temp[3]

# get std values across folders
std_ang_err_multi = np.zeros((len(fnames),input_data['n_sets'],len(input_data['amp_list']))) # num electrodes, n_sets, n_amps
mean_ang_err_multi = np.zeros_like(std_ang_err_multi)
mean_mag_multi = np.zeros_like(std_ang_err_multi)

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
            mean_ang_err_multi[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
            std_ang_err_multi[i_folder,i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
        
            
            # magnitude data
            delta_mag_stim = single_loc_exp[idx_look][1]
            data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
            mean_mag_multi[i_folder,i_set,i_amp] = np.mean(data)
            
            idx_look = idx_look + 1
"""       
#%% plot errobars of std_err for each amplitude and num-elec, and PD style
offset = np.array([-0.75,-0.25,0.25,0.75])*1.

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])
alpha_list= [1.0,1.0]
ls_list = ['-','--']
marker_list = ['s','.']
ms_list = [7,16]

counter = 0
for i_elec in range(len(fnames)):
    # single PD
    mean_std =np.mean(std_ang_err_single[i_elec,:,:],axis=0)
    std_std = np.std(std_ang_err_single[i_elec,:,:],axis=0)
    plt.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_std,std_std, capsize=5,elinewidth=1, \
                 color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    counter = counter+1
    # multi PD
    #mean_std =np.mean(std_ang_err_multi[i_elec,:,:],axis=0)
    #std_std = np.std(std_ang_err_multi[i_elec,:,:],axis=0)
    #plt.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_std,std_std, capsize=5,elinewidth=1, \
    #             color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    
    
plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Standard dev. of direction (deg)')

plt.xlim([-5,85])
plt.ylim([0,150])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,160,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))


#%% magnitude of effect, error bars across trials of the same location
offset = np.array([-0.75,-0.25,0.25,0.75])*1.

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])
alpha_list= [1.0,1.0]
ls_list = ['-','--']
marker_list = ['s','.']
ms_list = [7,16]

act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

counter = 0
n_stim_chans = [1,4]
for i_elec in range(mean_mag_single.shape[0]):
    # single PD
    mean_to_plot = np.mean(mean_mag_single[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_mag_single[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_to_plot, std_to_plot, capsize=5,elinewidth=1, \
                     color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    
    counter=counter+1
    # multi PD
    #mean_std =np.mean(mean_mag_multi[i_elec,:,:],axis=0)
    #std_std = np.std(mean_mag_multi[i_elec,:,:],axis=0)
    #plt.errorbar(np.array(input_data['amp_list'])+offset[counter],mean_std,std_std, capsize=5,elinewidth=1, \
    #             color=colors[counter],linewidth=1.5,linestyle=ls_list[i_elec],marker=marker_list[i_elec],markersize=ms_list[i_elec],markeredgewidth=2,alpha=alpha_list[i_elec])
    
#plt.xlim([-5,85])
#plt.xticks(ticks=np.arange(0,85,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
plt.ylim([0,7.5])

sns.despine()

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')

#%% magnitude statistics: increasing total current increases size, increasing number of elecs increases size at same total current
# out: mag, in: total_curr, n_elecs
mag = np.array([])
total_curr = np.array([])
n_elecs = np.array([])
n_stim_chans = [1,4] # not stored apparently, but this is how the data is loaded in

for i_amp in range(1,mean_mag_single.shape[-1]): # don't include 0 uA condition
    for i_elec in range(mean_mag_single.shape[0]):
        mag = np.append(mag, mean_mag_single[i_elec,:,i_amp])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp]*n_stim_chans[i_elec],mean_mag_single.shape[1]))
        n_elecs = np.append(n_elecs, np.tile(i_elec-1, mean_mag_single.shape[1]))

df = pd.DataFrame({'mag' : mag, 'total_curr' : total_curr, 'n_elecs' : n_elecs})

# perform two-way anova
model_anova = ols('mag ~ total_curr*n_elecs', data=df).fit()
# model_anova.summary() to get overal model statistic
mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = mag
X = np.transpose(np.vstack((total_curr,n_elecs)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)

#%% angular error
#colors = plt.cm.inferno([0,150])
offset = np.array([-0.75,-0.25,0.25,0.75])*1.

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])
alpha_list= [1.0,1.0]
ls_list = ['-','--']
marker_list = ['s','.']
ms_list = [7,16]

counter = 0
n_stim_chans = [1,4] # not stored apparently, but this is how the data is loaded in
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
plt.ylim([0,150])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,160,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
   
#%% angular error statistics: increasing total current increases angular error, increasing number of elecs decreases error
# out: mag, in: total_curr, n_elecs
ang_err = np.array([])
total_curr = np.array([])
n_elecs = np.array([])
n_stim_chans = [1,4] # not stored apparently, but this is how the data is loaded in

for i_amp in range(1,mean_ang_err_single.shape[-1]): # don't include 0 uA condition
    for i_elec in range(mean_ang_err_single.shape[0]):
        ang_err = np.append(ang_err, mean_ang_err_single[i_elec,:,i_amp])
        total_curr = np.append(total_curr, np.tile(input_data['amp_list'][i_amp]*n_stim_chans[i_elec],mean_mag_single.shape[1]))
        n_elecs = np.append(n_elecs, np.tile(i_elec-1, mean_mag_single.shape[1]))

df = pd.DataFrame({'err' : ang_err, 'total_curr' : total_curr, 'n_elecs' : n_elecs})

# perform two-way anova
model_anova = ols('err ~ total_curr+n_elecs', data=df).fit()
# model_anova.summary() to get overal model statistic

mdl_out = sm.stats.anova_lm(model_anova, typ=2)
print(mdl_out)
# perform linear regression to get coefficients

Y = ang_err
X = np.transpose(np.vstack((total_curr,n_elecs)))
X = sm.add_constant(X)
model_lm = sm.OLS(Y,X)
results = model_lm.fit()
print(results.params)
print(results.tvalues)


#%% compare 0 uA and 5 uA angular errors

effect_diff = mean_ang_err_single[0,:,1] - mean_ang_err_single[0,:,0]    

sp.stats.ttest_1samp(effect_diff,0)

#%% get what a large effect is by computing the mean of samples from a uniform distribution
n_samp_per = 100
n_runs = 100

data_samp = np.random.uniform(low=0,high=180.0,size=(n_samp_per,n_runs))
means_run = np.mean(data_samp,axis=0)

print(np.std(means_run))

 
#%% compute histogram overlap of two distributions

amp_idx = 6
data1 = mean_ang_err_single[0,:,amp_idx]
data2 = mean_ang_err_single[1,:,amp_idx]
perc_over, data1_hist,data2_hist, bin_edges = vae_utils.compute_histogram_overlap(data1, data2 ,bin_size=5)


print(perc_over)

t_stat,p = sp.stats.ranksums(data1,data2)
print(p)

#%% bootstrap multi-unit hash data to get a sense of the variance of the within map estimate
n_boot = 1000
mean_boot = np.zeros((n_boot,mean_ang_err_multi.shape[-1]))

data = mean_ang_err_multi[-1,:,:]
for i_boot in range(n_boot):
    idx_boot = np.random.choice(np.arange(0,data.shape[0]),size=data.shape[0],replace=True)
    data_boot = data[idx_boot,:]
    mean_boot[i_boot,:] = np.mean(data_boot,axis=0)
    
std_boot = np.std(mean_boot,axis=0)
#%% neighborhood correlation analysis

PD_sim_mat = np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1)))/np.pi*180
neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)


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

#%%
act_hand_vel_mag = np.linalg.norm(hand_vels,ord=2,axis=1)
max_hand_vel_mag = np.percentile(act_hand_vel_mag,100)

# get values first, load in relevant files
fnames = glob.glob(project_folder+'\\multi_loc_repetition_few_conditions_exp_ratemult10_nchans4_2021-10-05-114936*')

# get input data
f=open(fnames[0],'rb')
temp = pickle.load(f)
f.close()

input_data = temp[3]

# get std values across folders
std_ang_err = np.zeros((len(fnames),input_data['n_sets'],len(input_data['amp_list']))) # num electrodes, n_sets, n_amps
mean_ang_err = np.zeros_like(std_ang_err)
mean_mag = np.zeros_like(std_ang_err)

mean_ang_err_hash = np.zeros_like(std_ang_err)

for i_folder in range(len(fnames)):
    idx_look = 0
    f=open(fnames[i_folder],'rb')
    temp = pickle.load(f)
    f.close()
    
    single_loc_exp = temp[0]
    loc_idx = temp[1]
    amp = temp[2]
    input_data = temp[3]
    if(len(temp) > 3):
        stim_chan_list = temp[4]
        
    for i_set in range(input_data['n_sets']):
        for i_amp in range(len(input_data['amp_list'])):
            # angular error data -- single unit PD
            delta_dir_stim = single_loc_exp[idx_look][-2]
            pred_delta_dir = single_loc_exp[idx_look][-1]
            error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
            mean_ang_err[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
            std_ang_err[i_folder,i_set,i_amp] = sp.stats.circstd(delta_dir_stim[:,0])*180/np.pi
        
            # angular error data -- multi unit PD
            delta_dir_stim = single_loc_exp[idx_look][-2]
            
            stim_chan_pds = hash_PDs[stim_chan_list[idx_look].astype(int)]
            pred_delta_dir = sp.stats.circmean(stim_chan_pds,axis=1,high=np.pi,low=-np.pi)
            error_all = vae_utils.circular_diff(sp.stats.circmean(delta_dir_stim[:,0]),pred_delta_dir[0])
            mean_ang_err_hash[i_folder,i_set,i_amp] = abs(error_all)*180/np.pi
            
            # magnitude data
            delta_mag_stim = single_loc_exp[idx_look][1]
            data = delta_mag_stim[0:10,0]/max_hand_vel_mag*100
            mean_mag[i_folder,i_set,i_amp] = np.mean(data)
            
            idx_look = idx_look + 1



#%% angular error
colors = plt.cm.inferno([0,90,180])

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

offset = [-1,-0.5,0,0.5,1]*1

for i_elec in range(mean_ang_err.shape[0]):
    mean_to_plot = np.mean(mean_ang_err[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_ang_err[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list']),mean_to_plot, std_to_plot, capsize=5,elinewidth=2, \
                     color='r',linewidth=2,marker='.',markersize=16)
    
    mean_to_plot = np.mean(mean_ang_err_hash[i_elec,:,:],axis=0) # mean over sets
    std_to_plot = np.std(mean_ang_err_hash[i_elec,:,:],axis=0)
    ax.errorbar(np.array(input_data['amp_list'])+1,mean_to_plot, std_to_plot, capsize=5,elinewidth=2, \
                     color='b',linewidth=2,marker='.',markersize=16)

plt.xlabel('Current per electrode (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,25])
plt.ylim([0,160])
sns.despine()
plt.xticks(ticks=np.arange(0,25,20))
plt.yticks(ticks=np.arange(0,180,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))

plt.legend(['single_unit','hash'])



#%%
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






# run experiments for multiple amplitudes and number of electrodes
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

#save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([amp_elec_exp_out,n_chans,amp,input_data, stim_chan_list],f)
f.close()

# load data (assuming a break)
fname = project_folder+'\\amp_elec_single_map_exp_ratemult10_2021-09-30-111320.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

amp_elec_exp_out = temp[0]
n_chans = temp[1]
amp = temp[2]
input_data = temp[3]
stim_chan_list = temp[4]

# magnitude of effect, error bars across trials of the same location

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


# angular error -- current through each electrode

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

# compute neigh sim
    
PD_sim_mat = np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1)))/np.pi*180
neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)
    

# neighborhood score vs. angular error for each location

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



"""


#%%







