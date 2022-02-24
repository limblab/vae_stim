# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:53:51 2021

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

    
#%% visualize whole map (PD and neigh sim) to pick locations to stim 
    
neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mdl.locmap().astype(int))

fig,ax = plt.subplots(1,2,sharex=True,sharey=True)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right',size='5%',pad=0.05)
im=ax[0].imshow(neigh_map[:,:,0])
fig.colorbar(im,cax=cax,orientation='vertical')

divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right',size='5%',pad=0.05)
im=ax[1].imshow(PD_map[:,:,0],cmap=cmasher.infinity)
fig.colorbar(im,cax=cax,orientation='vertical')


#%% get idx from locations
loc_good = np.array([45,31]).reshape(-1,1)
loc_bad = np.array([62,40]).reshape(-1,1)

idx_good = vae_utils.convert_loc_to_idx(loc_good,mdl.locmap())[0].astype(int)
idx_bad = vae_utils.convert_loc_to_idx(loc_bad,mdl.locmap())[0].astype(int)

print(hand_vel_PDs[idx_good]*180/np.pi)
print(hand_vel_PDs[idx_bad]*180/np.pi)
#%% make a linear decoder where all weights the same magnitude but in the directio of the PD
rates = vae_utils.vae_get_rates(vae, kin_var_norm,gl.bin_size)
PDs_as_weights = np.vstack((np.cos(hand_vel_PDs), np.sin(hand_vel_PDs)))/6400
# make a linear decoder, then override the weights
straight_to_hand_dec = vae_utils.make_linear_decoder(x=rates, y=hand_vels, drop_rate=0.,n_iters=1,lr=0.01)
straight_to_hand_dec.state_dict()['layer1.0.weight'][:] = torch.Tensor(PDs_as_weights)

#%% stimulate the same location(s) many times and measure effect
#kin_var_use = np.random.normal(0,1,(1,7))
#kin_var_use = np.tile(kin_var_use,(kin_var_norm.shape[0],1))

input_data = {}
input_data['kin'] = 'hand'
input_data['dec'] = dec


input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
#input_data['amp_list'] = [0,10,20,80]
input_data['hand_vel_PDs'] = hand_vel_PDs

input_data['freq'] = 200 # Hz
input_data['n_pulses'] = 40 
input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
input_data['dir_func'] = 'bio_mdl'
input_data['decay_prob'] = 0 # don't bother decaying FR throughout course of train

input_data['trans_func'] = 'none'
input_data['vae'] = vae
input_data['all_joint_vel_norm'] = kin_var_norm
input_data['all_joint_ang'] = joint_angs
input_data['map_data'] = mdl.locmap()
input_data['block_size'] = 0.05 # mm

input_data['joint_vel'] = joint_vels
input_data['n_trials'] = 100
input_data['n_stim_chans'] = 1 # please don't change to more, I'm not sure if it works....
input_data['n_sets'] = 2

input_data['stim_chans_to_use'] = [idx_good,idx_bad]

input_data['PD_tol'] = np.pi/8

single_loc_exp, loc_idx, amp, stim_exp_out, stim_chan_list = stim_exp_utils.run_single_location_exp(input_data)

exp_fname = 'single_loc_exp_vaedec'

#%% save data
x=datetime.datetime.now()
dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
fname = project_folder+'\\'+exp_fname+dname+'.pkl'
f=open(fname,'wb')
pickle.dump([single_loc_exp,loc_idx,amp,input_data],f)
f.close()

#%% load data (assuming a break)
fname = project_folder+'\\single_loc_exp_vaedec_ratemult10_nchans1_2021-11-05-113235.pkl'
f=open(fname,'rb')
temp = pickle.load(f)
f.close()

single_loc_exp = temp[0]
loc_idx = temp[1]
amp = temp[2]
input_data = temp[3]

#%% magnitude of effect, error bars across trials of the same location
good_hex_color = '#01E5D5'
bad_hex_color = '#FF00FF'

colors = [good_hex_color,bad_hex_color]
m = ['s','o']
ms = [6,6]

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])


mean_mag = np.zeros((input_data['n_sets'],len(input_data['amp_list'])))
std_mag = np.zeros_like(mean_mag)
offset = [0,0,0,0,0]*1

idx_look = 0
for i_set in range(input_data['n_sets']):
    for i_amp in range(len(input_data['amp_list'])):
        delta_mag_stim = single_loc_exp[idx_look][1]
        data = delta_mag_stim[:,0]/max_hand_vel_mag*100
        mean_mag[i_set,i_amp] = np.mean(data)
        std_mag[i_set,i_amp] = np.std(data)
        idx_look=idx_look+1
    ax.errorbar(np.array(input_data['amp_list'])[:]+offset[i_set],mean_mag[i_set,:], std_mag[i_set,:], capsize=5,elinewidth=2, \
                     markerfacecolor='none',color=colors[i_set],markeredgecolor=colors[i_set],markeredgewidth=3,marker=m[i_set], markersize=ms[i_set],linewidth=2)

plt.xlim([-5,85])
plt.ylim([0,4])
plt.xticks(ticks=np.arange(0,85,20))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

sns.despine()

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Magnitude (% max speed)')
# angular error

fig=plt.figure(figsize=(5,5))
ax=fig.add_axes([0.2,0.15,0.75,0.75])

mean_err = np.zeros((input_data['n_sets'],len(input_data['amp_list'])))
std_err = np.zeros_like(mean_mag)
offset = [0,0,0,0]*1
idx_look = 0
for i_set in range(input_data['n_sets']):
    for i_amp in range(len(input_data['amp_list'])):
        delta_dir_stim = single_loc_exp[idx_look][-2]
        pred_delta_dir = single_loc_exp[idx_look][-1]
        error_all = abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:]))*180/np.pi
        mean_err[i_set,i_amp] = np.mean(error_all)
        std_err[i_set,i_amp] = np.std(error_all)
        idx_look=idx_look+1
    ax.errorbar(np.array(input_data['amp_list'])[:]+offset[i_set],mean_err[i_set,:], std_err[i_set,:], capsize=5,elinewidth=2, \
                     markerfacecolor='none',color=colors[i_set],markeredgecolor=colors[i_set],markeredgewidth=3,marker=m[i_set], markersize=ms[i_set],linewidth=2)


plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,85])
plt.ylim([0,190])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,200,30))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(10))
    
#%% PD plot zoomed in to idx good and idx bad
PD_map = vae_utils.convert_list_to_map(hand_vel_PDs.reshape(1,-1),mdl.locmap().astype(int))

fig = plt.figure(figsize=(6,5))
ax=fig.gca()
to_plot = PD_map[:,:,0]

to_plot[to_plot<0] = to_plot[to_plot<0] + 2*np.pi

elec_space = 1

alpha = np.zeros_like(to_plot)
x = np.arange(0, 80, elec_space, dtype=np.float32) 
x = np.append(x,-1*x[1:])
y = np.arange(0, 80, elec_space, dtype=np.float32)
y = np.append(y,-1*y[1:])
xv, yv = np.meshgrid(x, y)
xv = np.reshape(xv, (xv.size, 1))
yv = np.reshape(yv, (yv.size, 1))
grid = np.hstack((xv, yv))

grid = grid + np.transpose(loc_bad)

keep_idx = np.argwhere(np.all(np.logical_and(grid>=0, grid < 80),axis=1)).reshape((-1,))
grid = grid[keep_idx,:].astype(int)

alpha[grid[:,0],grid[:,1]] = 1
im=plt.imshow(to_plot*180/np.pi,cmap=cmasher.infinity, vmin=0,vmax=360,alpha=alpha)


space_min = 6
space_max = 12
min_x = np.minimum(loc_good[0],loc_bad[0])-space_min
max_x = np.maximum(loc_good[0],loc_bad[0])+space_max

desired_diff = max_x-min_x

min_y = np.minimum(loc_good[1],loc_bad[1])
max_y = np.maximum(loc_good[1],loc_bad[1])

curr_diff = max_y - min_y
min_y = min_y - (desired_diff-curr_diff)/2
max_y = max_y + (desired_diff-curr_diff)/2


plt.plot(loc_good[1],loc_good[0],'.',markersize=10,color='black')
plt.plot(loc_bad[1],loc_bad[0],'.',markersize=10,color='magenta')

plt.xlim([min_y,max_y])  # this manages to flip the map vertically....
plt.ylim([min_x,max_x])
ax.set_aspect('equal')
ax.set_xticks(ticks=[])
ax.set_yticks(ticks=[])
#%% neighbor similarity histogram with idx_good and idx_bad bars colored

fig = plt.figure(figsize=(4,5))
ax=fig.add_axes([0.25,0.25,0.3,0.6])
N,bin_edges,patches = plt.hist(neigh_sim,15,edgecolor='white',linewidth=1) # second arg = n_bins 
   
plt.xlim([0,125])
plt.xlabel('Neighborhood \nPD error (deg)') 
plt.ylabel('Number of neurons')
sns.despine()

for i in range(len(patches)):
    patches[i].set_facecolor('black')

# find bin_edge with idx_good and idx_bad
neigh_good = neigh_sim[idx_good]
neigh_bad = neigh_sim[idx_bad]

bin_good = np.digitize(neigh_good, bin_edges) - 1
bin_bad = np.digitize(neigh_bad, bin_edges) - 1

patches[bin_good].set_facecolor(good_hex_color) # green
patches[bin_bad].set_facecolor(bad_hex_color) # red

plt.xticks(ticks=np.arange(0,140,40))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(100))
#%% circle vector plots of stim-effect velocity for a single amp
    

idx_look = [2,9] # 10uA stim for both sets
#idx_look = [1,5]
n_plot = 100

for i_set in range(len(idx_look)):
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig = plt.figure(figsize=(4,4))
    ax=fig.gca()
    delta_dir_stim = single_loc_exp[idx_look[i_set]][-2][:,0]
    
    delta_dir_stim[delta_dir_stim > np.pi] = delta_dir_stim[delta_dir_stim > np.pi] - 2*np.pi
    delta_mag_stim = single_loc_exp[idx_look[i_set]][1][:,0]
    mag_data = delta_mag_stim/max_hand_vel_mag*100
    
    for i_vec in range(n_plot):
        theta = [0,delta_dir_stim[i_vec]]
        r = [0,mag_data[i_vec]]

        
        #ax.plot(theta,r,color='black',alpha=random.uniform(0.25,1))
        
        plt.plot([0,mag_data[i_vec]*np.cos(delta_dir_stim[i_vec])],[0,mag_data[i_vec]*np.sin(delta_dir_stim[i_vec])], \
                  linewidth=0.5, alpha=random.uniform(1,1), color='black')
        #plt.arrow(0,0,mag_data[i_vec]*np.cos(delta_dir_stim[i_vec]),mag_data[i_vec]*np.sin(delta_dir_stim[i_vec]), \
         #         width=0.05,head_width=0.25,head_length=0.25,edgecolor='none',facecolor='black',alpha=random.uniform(1,1))
        
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    lims = 5
    plt.xlim([-lims,lims])
    plt.ylim([-lims,lims])
    
    # plot circle 
    ax=fig.gca()
    ax.add_patch(plt.Circle((0,0),lims,color='black',fill=False))
    plt.plot([-lims,lims],[0,0],color='grey',linewidth=1)
    plt.plot([0,0],[-lims,lims],color='grey',linewidth=1)
    ax.set_aspect('equal')
    
    bin_edges = np.linspace(-np.pi,np.pi,20)

    r_plot, bin_edges = np.histogram(delta_dir_stim,bin_edges)
    
    bin_centers = bin_edges[0:-1] + np.mean(np.diff(bin_edges))/2
    
    plt.figure(figsize=(4,4))
    ax=plt.subplot(111,polar=True)
    plt.bar(x=bin_centers,
            height=r_plot,
            width=2*np.pi/len(bin_centers),
            edgecolor='none',linewidth=0,color='#377eb8')
    ax.set_rlim([0,100])

    ax.set_yticklabels(labels=[])
    ax.set_xticklabels(labels=[])
    ax.set_aspect('equal')
#%%

plt.xlabel('Amplitude (' + u"\u03bcA" + ')')     
plt.ylabel('Mean angular error (deg)')

plt.xlim([-5,85])
plt.ylim([0,180])
sns.despine()
plt.xticks(ticks=np.arange(0,85,20))
plt.yticks(ticks=np.arange(0,200,40))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(10))  
    
    
    
    
    
    