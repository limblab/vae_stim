# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:22:41 2021

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

# get training data folder
training_data_folder = r'D:\Lab\Data\StimModel\training_data'

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

kin_var_norm = joint_vels_norm

#%% pick experiments to run
run_amp_elec_exp = 0
run_amp_freq_exp = 0
run_amp_sim_exp = 0
run_amp_blocksize_exp = 0
run_PD_neighborhood_exp = 0
run_amp_elec_repeat_exp = 0
run_PD_distribution_exp = 1
run_PD_hash_exp = 0
#%% run analysis for many maps
# get project folder and all relevant sub folders
fpath = r'D:\Lab\Data\StimModel\models'
folders = glob.glob(fpath + '\\Han_*')

#folders_use = np.random.choice(len(folders),10,replace=False)

for i_folder in [0]:#range(1,len(folders)):
    print('Folder: ' + str(i_folder))
    # load parameter file. This is currently convoluted, but works
    path_to_model_dict = glob.glob(folders[i_folder] + r'\*model_params*')[0]
    path_to_model_yaml = glob.glob(folders[i_folder] + r'\*.yaml')[0]
    gl.Params.load_params(gl.params,path_to_model_yaml)

    # set params['cuda'] to false since my computer doesn't have a GPU ( :( )
    gl.params.params['cuda']=False
    
    # load in vae, hand decoder, PDs
    vae = vae_utils.load_vae_parameters(fpath=path_to_model_dict,input_size=kin_var_norm.shape[1]) 
    
    path_to_dec = glob.glob(folders[i_folder] + '\\*hand_vel_dec*')
    f=open(path_to_dec[0],'rb')
    dec = pickle.load(f)
    f.close()
    
    path_to_PDs = glob.glob(folders[i_folder] + '\\*PD_multiunit_hash_pow2*')
    f=open(path_to_PDs[0],'rb')
    hand_vel_PDs = pickle.load(f)[0]
    f.close()
    
    # run experiments for multiple amplitudes and number of electrodes
    if(run_amp_elec_exp):
        input_data = {}
        input_data['kin'] = 'hand';
        input_data['dec'] = dec
        
        input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
        #input_data['amp_list'] = [0,1,2,3,4,5]
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
        input_data['n_stim_chans_list'] = [1,4]
        
        input_data['PD_tol'] = np.pi/8
        
        amp_elec_exp_out, n_chans, amp, stim_exp_out = stim_exp_utils.run_elec_amp_stim_exp(input_data)
        
        exp_fname = 'amp_elec_exp'
        
        # save data
        x=datetime.datetime.now()
        dname = '_ratemult' + str(gl.rate_mult) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
        fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
        f=open(fname,'wb')
        pickle.dump([amp_elec_exp_out,n_chans,amp,input_data],f)
        f.close()
    
    if(run_amp_freq_exp):
        n_chans = [1,4]
        for i_chan in range(len(n_chans)):
            # run experiments comparing effect of amplitude and frequency
            input_data = {}
            input_data['kin'] = 'hand';
            input_data['dec'] = dec
            
            input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
            input_data['freq_list'] = [50,100,200] # Hz
            
            input_data['hand_vel_PDs'] = hand_vel_PDs
            
            input_data['n_pulses_list'] = [10,20,40]
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
            input_data['n_stim_chans'] = n_chans[i_chan]
            
            input_data['PD_tol'] = np.pi/8
            
            amp_freq_exp_out, freq, amp = stim_exp_utils.run_amp_freq_stim_exp(input_data)
            
            exp_fname = 'amp_freq_exp'
            
            # save data
            x=datetime.datetime.now()
            dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
            fname = folders[i_folder] +'\\'+exp_fname+dname+'.pkl'
            f=open(fname,'wb')
            pickle.dump([amp_freq_exp_out,freq,amp,input_data],f)
            f.close()
        
    if(run_amp_sim_exp):
        n_chans = [1,4]
        for i_chan in range(len(n_chans)):
            PD_sim_mat = (np.pi - np.abs(vae_utils.circular_diff(hand_vel_PDs.reshape(1,-1),hand_vel_PDs.reshape(-1,1))))/np.pi
    
            neigh_sim = vae_utils.get_neighbor_sim(mdl.locmap(), PD_sim_mat, max_neigh_dist=3)
            neigh_map = vae_utils.convert_list_to_map(neigh_sim.reshape(1,-1),mdl.locmap().astype(int))
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
            input_data['trans_func'] = 'none'
            input_data['vae'] = vae
            input_data['all_joint_vel_norm'] = kin_var_norm
            input_data['all_joint_ang'] = joint_angs
            input_data['map_data'] = mdl.locmap()
            input_data['block_size'] = 0.05 # mm
            
            input_data['joint_vel'] = joint_vels
            input_data['n_trials'] = 500
            input_data['n_stim_chans'] = n_chans[i_chan]
            
            input_data['PD_tol'] = np.pi/8
            
            input_data['sim_score'] = neigh_sim
            input_data['sim_tol'] = np.percentile(neigh_sim,85) #
            
            amp_high_sim_exp_out, amp, step, stim_chan_list = stim_exp_utils.run_amp_high_sim_exp(input_data)
    
            # save data
            exp_fname= 'amp_highsim_exp'
            x=datetime.datetime.now()
            dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
            fname = folders[i_folder] +'\\'+exp_fname+dname+'.pkl'
            f=open(fname,'wb')
            pickle.dump([amp_high_sim_exp_out,amp,step,stim_chan_list,neigh_sim,input_data],f)
            f.close()
            
            
    if(run_amp_blocksize_exp):
        n_chans = [1,4]
        for i_chan in range(len(n_chans)):
            # run experiments for multiple amplitudes at different block sizes
            input_data = {}
            input_data['kin'] = 'hand';
            input_data['dec'] = dec
            
            input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
            input_data['block_size_list'] = [0.033,0.05,0.067,0.075] # mm
            
            
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
            
            input_data['joint_vel'] = joint_vels
            input_data['n_trials'] = 500
            input_data['n_stim_chans'] = n_chans[i_chan]
            
            input_data['PD_tol'] = np.pi/8
            
            amp_blocksize_exp_out, block_size, amp, stim_exp_out = stim_exp_utils.run_blocksize_amp_stim_exp(input_data)
            
            exp_fname = 'amp_blocksize_exp'
            
            # save data
            x=datetime.datetime.now()
            dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
            fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
            f=open(fname,'wb')
            pickle.dump([amp_blocksize_exp_out,n_chans,amp,input_data],f)
            f.close()
            
    if(run_PD_neighborhood_exp):
        exp_fname = 'neigh_dist_exp'
        unit_dist, pd_diff = vae_utils.get_pd_dist(mdl.locmap(), hand_vel_PDs) # in blocks
            
        x=datetime.datetime.now()
        dname = '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
        fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
        f=open(fname,'wb')
        pickle.dump([pd_diff,unit_dist],f)
        f.close()
            
    if(run_amp_elec_repeat_exp):
        input_data = {}
        input_data['kin'] = 'hand';
        input_data['dec'] = dec
        
        #input_data['amp_list'] = [0,5,10,15,20,40,80] # uA
        input_data['amp_list'] = [0,10,20]
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
        input_data['n_trials'] = 25
        input_data['n_stim_chans'] = 4 # can't be a list
        input_data['n_sets'] = 500
        
        input_data['PD_tol'] = np.pi/8
        
        single_loc_exp, loc_idx, amp, stim_exp_out, stim_chan_list = stim_exp_utils.run_single_location_exp(input_data)
        
        exp_fname = 'multi_loc_repetition_few_conditions_exp_multiunitPDs'
            
        # save data
        x=datetime.datetime.now()
        dname = '_ratemult' + str(gl.rate_mult) + '_nchans' + str(input_data['n_stim_chans']) + '_' + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
        fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
        f=open(fname,'wb')
        pickle.dump([single_loc_exp,loc_idx,amp,input_data,stim_chan_list],f)
        f.close()
            
    if(run_PD_distribution_exp):    
        n_bins = 10
        PD_hist, PD_bin_edges = np.histogram(hand_vel_PDs,bins=n_bins)
        
        n_data = np.sum(PD_hist)
        PD_hist = PD_hist/n_data
        exp_count = 1/n_bins
        
        non_uniform_meas = np.linalg.norm(PD_hist,ord=2)
        chi_square = np.sum(np.divide(np.square(PD_hist-exp_count),exp_count))

        exp_fname = 'PD_distribution_uniformity_multiUnitPDs_v4'
        x=datetime.datetime.now()
        dname = exp_fname + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
        fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
        f=open(fname,'wb')
        pickle.dump([PD_hist,non_uniform_meas,chi_square],f)
        f.close()
            
    if(run_PD_hash_exp):
        hash_PDs, hash_params, rates, hash_rates = vae_utils.get_hash_PDs(vae, kin_var_norm, hand_vels, mdl.locmap()*0.05) # locations in mm
        exp_fname = 'PD_multiunit_hash_pow2_small'
        x=datetime.datetime.now()
        dname = exp_fname + x.strftime("%G")+'-'+x.strftime("%m")+'-'+x.strftime("%d")+'-'+x.strftime("%H")+x.strftime("%M")+x.strftime("%S")
        fname = folders[i_folder]+'\\'+exp_fname+dname+'.pkl'
        f=open(fname,'wb')
        pickle.dump([hash_PDs],f)
        f.close()  
            
            
            
            
            
            
            