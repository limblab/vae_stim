# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:26:49 2021

@author: Joseph Sombeck
"""

import vae_utils
import opensim as osim
import osim_utils
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import interp1d
import global_vars as gl

"""
inputs to a trial:
    stimulation parameters: 
        stim_chan : stim chan(s) = location(s)
        amp : stim amp
        freq : stim freq
        n_pulses : train length = number of pulses
        stim_start_t: time to start stimulating
        dir_func : direct activation function
        trans_func : transsynaptic activation function
        
    vae
    joint_vel : base joint velocity for the trial
    dec : decoder
    map_data : matrix mapping 6400x1 array of neurons to 80x80 map (or whatever the dims are)
    block_size : size of a single block (in mm)
"""

def get_activated_neurons(input_data, stim_chan_loc):
    
    # get distance from stim_loc to other neurons 
    dist_mat = euclidean_distances(stim_chan_loc, input_data['map_data'])*input_data['block_size']
    dist_mat[0,np.argwhere(dist_mat<=0)] = input_data['block_size']*0.5
    
    # activate neurons based on direct act func
    if(input_data['dir_func'].lower() == "exp_decay"):
        # use exponential decay activation function. parameters from Karthik's biophysical modeling
        amp_list = np.array([15,30,50,100])
        space_constants = np.array([100,250,325,500])/1000 # in mm
        
        interp_fn = interp1d(amp_list,space_constants,kind='linear',bounds_error=False,fill_value='extrapolate')
        space_constant = np.max([interp_fn(input_data['amp']),0.000001])
        
        prob_act = np.transpose(np.exp(-dist_mat/space_constant))
        prob_act = np.tile(prob_act,input_data['n_pulses'])
        prob_act[np.argwhere(prob_act<0.01)] = 0
        
        is_act = np.random.rand(prob_act.shape[0],prob_act.shape[1]) < prob_act
        
        # TODO: IMPLEMENT THIS
   # elif(input_data['dir_func'].lower() == "sigmoid"):
        # use sigmoidal activation function. parameters from Karthik's biophysical modeling
    else:
        
        is_act = np.zeros((input_data['map_data'].shape[0],input_data['n_pulses']))
        
    
    
    return is_act


def get_chan_loc(chan_idx, map_data):
    return map_data[chan_idx].reshape(1,-1)


def update_rates_stim(input_data, rates, is_act,bin_edges):
    # add to rates if neuron is activated....
    updated_rate = np.copy(rates) # copy rates, update copy. 
    
    # get which bin each stim pulse is in
    pulse_time = input_data['stim_start_t'] + np.arange(0,input_data['n_pulses'])/input_data['freq']
    pulse_bin_idx = np.digitize(pulse_time,bin_edges)-1 # start counting bins at 0
    
    for i_bin in range(updated_rate.shape[0]):
        num_act = np.sum(is_act[:,pulse_bin_idx==i_bin],axis=1)
        
        # lower rates in bin to account for inhibition
        updated_rate[i_bin,:] = updated_rate[i_bin,:] - np.multiply(updated_rate[i_bin,:],np.transpose(num_act*(0.01/gl.bin_size))) # remove 10 ms of data
        # then add spikes to rates based on number of times activated and bin size
        updated_rate[i_bin,:] = updated_rate[i_bin,:] + np.transpose(num_act)/gl.bin_size
    
    return updated_rate

def stim_cortical_map(input_data, rates):
    # TODO: implement synaptic activation
    
    # preallocate whether a neuron was activated for each pulse
    is_act = np.zeros((input_data['map_data'].shape[0],input_data['n_pulses']),dtype=bool)
    
    # if stimulation amplitude or train length is 0 or small, return original rates
    if(input_data['amp'] <= 0 or input_data['n_pulses'] <= 0):
        return np.copy(rates), is_act # a new object of rates
    
   # for each stimulation location, get activated list of neurons for each pulse
    for chan in input_data['stim_chan']:
        is_act = np.logical_or(is_act, get_activated_neurons(input_data, get_chan_loc(chan, input_data['map_data'])))
            
    # update rates based on activation
    bin_edges = np.arange(0,input_data['joint_vel'].shape[0]+1)*gl.bin_size
    stim_rates = update_rates_stim(input_data,rates,is_act,bin_edges)
    
    # get synaptic transmission
    
    # update rates based on synaptic activated

    # return rates, is_act
    return stim_rates, is_act    
    

def run_stim_trial(input_data):
    
    # pass joint vels through forward model to get cortical rates'
    rates = vae_utils.vae_get_rates(input_data['vae'],input_data['joint_vel'],gl.bin_size)
    
    # sample "spikes" and then bin <- adds noise
    samp_rates = vae_utils.sample_rates(rates)
    samp_rates_stim = vae_utils.sample_rates(rates)
    
    # stimulate map
    samp_rates_stim,is_act = stim_cortical_map(input_data, samp_rates_stim)
        
    # decode joint velocity
    joint_vel_true = vae_utils.linear_dec_forward(dec=input_data['dec'],x=rates)
    joint_vel_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=samp_rates_stim)
    joint_vel_no_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=samp_rates)
    
    # integrate decoded joint vel to get predicted joint angle
    int_joint_ang_true = vae_utils.integrate_vel(input_data['init_joint_ang'],joint_vel_true,gl.bin_size)
    int_joint_ang_stim = vae_utils.integrate_vel(input_data['init_joint_ang'],joint_vel_stim,gl.bin_size)
    int_joint_ang_no_stim = vae_utils.integrate_vel(input_data['init_joint_ang'],joint_vel_no_stim,gl.bin_size)
    
    # use opensim to obtain elbow and hand cartesian velocities
    true_point_kin_data = osim_utils.get_pointkin(int_joint_ang_true)
    stim_point_kin_data = osim_utils.get_pointkin(int_joint_ang_stim)
    no_point_kin_data = osim_utils.get_pointkin(int_joint_ang_no_stim)
    
    # output samp_rates, samp_rates stim (so that the effect of stim can be seen), 
    # output decoded joint velocity, elbow and hand cartesian vels 
    return [rates, samp_rates, samp_rates_stim, is_act, 
            joint_vel_true, joint_vel_no_stim, joint_vel_stim, 
            true_point_kin_data, no_point_kin_data, stim_point_kin_data]


def run_many_stim_trials(input_data):
    # this function runs many stimulation trials (defined by input_data['n_trials'])
    # and outputs the change in hand and elbow velocity due to stim for all trials
    # Also outputs PD of the stimulated location(s)
    
    # randomly chooses stimulation locations, if doing multi-electrode (defined by input_data['n_stim_chan']),
    # picks electrodes that are similar, based on input_data['similarity_mat']
    # similarity mat is a N-neurons x N-neurons matrix with some similarity metric between neurons
    # randomly chooses joint velocities for each trial  

    """
    input_data['amp'] = 100 # uA
    input_data['freq'] = 100 # Hz
    input_data['n_pulses'] = 10 # number of pulses.
    input_data['stim_start_t'] = gl.bin_size; # equivalent to 1 bin.
    input_data['dir_func'] = 'exp_decay'
    input_data['trans_func'] = 'none'
    input_data['vae'] = vae
    input_data['joint_vel'] = joint_vel_samp
    input_data['init_joint_ang']
    input_data['dec'] = dec
    input_data['map_data'] = mdl.locmap()
    input_data['block_size'] = 0.05 # mm
    
    input_data['all_joint_vel_norm'] = joint_vels
    input_data['all_joint_ang'] = joint_angs
    input_data['n_trials'] = 10
    input_data['n_stim_chans'] = 1
    
    input_data['sim_mat'] = PD_sim_mat
    input_data['sim_tol'] = 0.92; # corresponds to 22.5 degrees
    """
    
    # initilize useful variables
    n_neurons = input_data['map_data'].shape[0]
    # add extra bins because opensim needs at least a bunch of bins
    trial_len_idx = np.ceil((input_data['stim_start_t'] + input_data['n_pulses']/input_data['freq'])/gl.bin_size + 20).astype(int) 
    
    # initialize output data
    stim_chan_list = np.zeros((input_data['n_trials'],input_data['n_stim_chans']))
    joint_ang_0_list = np.zeros((input_data['n_trials'],input_data['all_joint_ang'].shape[1]))
    true_joint_vels_list = np.zeros((input_data['n_trials'],trial_len_idx,input_data['all_joint_vel_norm'].shape[1]))
    stim_rates_list = np.zeros((input_data['n_trials'],trial_len_idx,n_neurons))
    no_stim_rates_list = np.zeros_like(stim_rates_list)
    true_rates_list = np.zeros_like(stim_rates_list)
    
    no_stim_joint_vels_list = np.zeros_like(true_joint_vels_list)
    stim_joint_vels_list = np.zeros_like(true_joint_vels_list)
    
    true_point_kin_list = []
    no_stim_point_kin_list = []
    stim_point_kin_list = []
    
    # iterate through each trial
    for i_trial in range(input_data['n_trials']):
        # get stim chan(s)
        # get initial stim_chan randomly
        temp_stim_chans = np.array([np.random.randint(0,n_neurons)])
        
        if(input_data['n_stim_chans']>1): # this only adds stimulation channels if doing multi-elec stim
            # get a random channel that is sufficiently similar to the original (temp_stim_chans[0])
            # sim scores in input_data['sim_mat'][temp_stim_chans[0],:]
            good_chans = np.reshape(np.argwhere(input_data['sim_mat'][temp_stim_chans] > input_data['sim_tol']),(-1,))
            temp_stim_chans = np.append(temp_stim_chans, np.random.choice(good_chans,size=input_data['n_stim_chans']-1,replace=False))
            
        
        # get normalized joint vels and initial joint angle for this trial
        trial_start = np.random.randint(0,input_data['all_joint_ang'].shape[0] - 2*trial_len_idx)
        
        # run stim trial
        stim_trial_input = input_data
        stim_trial_input['joint_vel'] = input_data['all_joint_vel_norm'][trial_start:trial_start+trial_len_idx,:]
        stim_trial_input['init_joint_ang'] = input_data['all_joint_ang'][trial_start,:]
        stim_trial_input['stim_chan'] = temp_stim_chans
        
        stim_out = run_stim_trial(input_data)
        # stim_out :
        # [rates, samp_rates, samp_rates_stim, is_act, joint_vel_true, joint_vel_no_stim, joint_vel_stim, 
        # true_point_kin_data, no_point_kin_data, stim_point_kin_data]

        
        
        # package outputs    
        joint_ang_0_list[i_trial] = stim_trial_input['init_joint_ang'] # initial joint angle
        true_joint_vels_list[i_trial] = stim_trial_input['joint_vel'] # underlying joint velocity
        true_rates_list[i_trial] = stim_out[0] # underlying rates
        no_stim_rates_list[i_trial] = stim_out[1] # rates sampled with and without stim
        stim_rates_list[i_trial] = stim_out[2]
        stim_chan_list[i_trial,:] = temp_stim_chans # stim channels

        # joint vels with and without stim
        true_joint_vels_list[i_trial] = stim_out[4]
        no_stim_joint_vels_list[i_trial] = stim_out[5]
        stim_joint_vels_list[i_trial] = stim_out[6]

        # elbow and hand pos/vel with and without stim, also underlying rates
        true_point_kin_list.append(stim_out[7])
        no_stim_point_kin_list.append(stim_out[8])
        stim_point_kin_list.append(stim_out[9])
        
    
    # output metrics
    return [joint_ang_0_list, true_rates_list, no_stim_rates_list, stim_rates_list, 
                stim_chan_list,  true_joint_vels_list, no_stim_joint_vels_list, stim_joint_vels_list, 
                true_point_kin_list, no_stim_point_kin_list, stim_point_kin_list]


def compute_exp_metrics(stim_data):
    # stim data contains:


    
    
    return 0






















    







