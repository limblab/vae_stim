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
import matplotlib.pyplot as plt
import scipy as sp

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
    dist_mat = euclidean_distances(stim_chan_loc, input_data['map_data'])*input_data['block_size'] # in mm

    # activate neurons based on direct act func
    if(input_data['dir_func'].lower() == "exp_decay"):
        # use exponential decay activation function. parameters from Karthik's biophysical modeling
        amp_list = np.array([15,30,50,100])
        space_constants = np.array([100,250,325,500])/1000 # in mm
        
        interp_fn = interp1d(amp_list,space_constants,kind='linear',bounds_error=False,fill_value='extrapolate')
        space_constant = np.max([interp_fn(input_data['amp']),0.000001])
        
        prob_act = np.transpose(np.exp(-dist_mat/space_constant))
        prob_act = np.tile(prob_act,input_data['n_pulses'])
        
        is_act = np.random.rand(prob_act.shape[0],prob_act.shape[1]) < prob_act
        
        # TODO: IMPLEMENT THIS
    elif(input_data['dir_func'].lower() == "bio_mdl"):
        # use activation data from Karthik's modeling.
        
        # find amplitude in list
        amp_idx = np.argwhere(gl.karth_amp == input_data['amp'])
        
        # if it isn't there, interpolate
        if amp_idx.size == 0:
            is_act = np.zeros((input_data['map_data'].shape[0],input_data['n_pulses']))
            # find amplitudes in bio model that are below and above given amp
            amp_idx_down = np.argwhere(input_data['amp'] > gl.karth_amp)[-1]
            amp_idx_up = np.argwhere(input_data['amp'] < gl.karth_amp)[0]
            
            # get probability for those amplitudes
            bin_idx = np.digitize(dist_mat*1000, gl.karth_dist) # convert dist to um
            bin_idx[bin_idx==0] = 1
            bin_idx[bin_idx==gl.karth_dist.shape[0]] = gl.karth_dist.shape[0]-1
            bin_idx = bin_idx - 1
            
            prob_act_down = gl.karth_act_prop[amp_idx_down,bin_idx].reshape(-1,1)
            prob_act_up = gl.karth_act_prop[amp_idx_up,bin_idx].reshape(-1,1)
            
            # take weighted mean to do linear
            prob_act = np.concatenate((prob_act_down,prob_act_up),axis=1)
            
            val = (input_data['amp']-gl.karth_amp[amp_idx_down])/(gl.karth_amp[amp_idx_up]-gl.karth_amp[amp_idx_down])
            amp_weights = np.transpose(np.tile(np.array([1-val,val]),prob_act.shape[0])) 
            
            prob_act = np.average(prob_act,axis=1,weights=amp_weights).reshape(-1,1)
            
            
        # if it's there, get distance bin for each neuron, use probability associated with that bin
        else:
            bin_idx = np.digitize(dist_mat*1000, gl.karth_dist) # convert dist to um
            bin_idx[bin_idx==0] = 1
            bin_idx[bin_idx==gl.karth_dist.shape[0]] = gl.karth_dist.shape[0]-1
            bin_idx = bin_idx - 1
            
            prob_act = gl.karth_act_prop[amp_idx,bin_idx].reshape(-1,1)
            
            
        prob_act[prob_act<0.05] = 0
        prob_act = np.tile(prob_act,input_data['n_pulses'])
        is_act = np.random.rand(prob_act.shape[0],prob_act.shape[1]) < prob_act
            
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
        
        # set rates in bin to 0 if responsive at all
        #updated_rate[i_bin,np.argwhere(num_act>0)] = 0
        
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
        
    
    if(input_data['kin'].lower() == 'joint'):
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
        
        # extract hand and elbow vel
        hand_vel_true = true_point_kin_data[1][:,1:]
        elbow_vel_true = true_point_kin_data[4][:,1:]
        
        hand_vel_stim = stim_point_kin_data[1][:,1:]
        elbow_vel_stim = stim_point_kin_data[4][:,1:]
        
        hand_vel_no = no_point_kin_data[1][:,1:]
        elbow_vel_no = no_point_kin_data[4][:,1:]
        
        return [rates, samp_rates, samp_rates_stim, is_act, 
            hand_vel_true, hand_vel_stim, hand_vel_no,
            elbow_vel_true, elbow_vel_stim, elbow_vel_no]
        
    elif(input_data['kin'].lower() == 'hand'): # currently, this is not meant to be ran with run many stim trials as output of function is different than with joint
        kin_var_true = vae_utils.vae_decoder(vae=input_data['vae'],samples=rates,bin_size=gl.bin_size)
        kin_var_stim = vae_utils.vae_decoder(vae=input_data['vae'],samples=samp_rates_stim,bin_size=gl.bin_size)
        kin_var_no_stim = vae_utils.vae_decoder(vae=input_data['vae'],samples=samp_rates,bin_size=gl.bin_size)
        
        hand_vel_true = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_true)
        hand_vel_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_stim)
        hand_vel_no_stim = vae_utils.linear_dec_forward(dec=input_data['dec'],x=kin_var_no_stim)
        
        return [rates, samp_rates, samp_rates_stim, is_act, 
            hand_vel_true, hand_vel_stim, hand_vel_no_stim]
        
        
    else:
        raise Exception('kin not implemented')
    # output samp_rates, samp_rates stim (so that the effect of stim can be seen), 
    # output decoded joint velocity, elbow and hand cartesian vels 
    


def run_many_stim_trials(input_data):
    # this function runs many stimulation trials (defined by input_data['n_trials'])
    # and outputs the change in hand and elbow velocity due to stim for all trials
    # Also outputs PD of the stimulated location(s)
    
    # randomly chooses stimulation locations, if doing multi-electrode (defined by input_data['n_stim_chan']),
    # picks electrodes that are similar, based on input_data['similarity_mat']
    # similarity mat is a N-neurons x N-neurons matrix with some similarity metric between neurons
    # randomly chooses joint velocities for each trial  

    """
    input_data['hand_vel_PDs'] = hand_vel_PDs

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
    input_data['PD_tol'] = 0.92; # corresponds to 22.5 degrees
    
    input_data['stim_chan_mask'] : mask of channels that can be used
    """
    
    # initilize useful variables
    n_neurons = input_data['map_data'].shape[0]
    # add extra bins because opensim needs at least a bunch of bins
    trial_len_idx = np.ceil((input_data['stim_start_t'] + input_data['n_pulses']/input_data['freq'])/gl.bin_size + 20).astype(int) 
    
    # initialize output data
    stim_chan_list = np.zeros((input_data['n_trials'],input_data['n_stim_chans']))

    true_hand_vel_list = []
    no_hand_vel_list = []
    stim_hand_vel_list = []
    
    true_elbow_vel_list = []
    no_elbow_vel_list = []
    stim_elbow_vel_list = []
    
    # get channel sampling weight based on PD distribution
    
    bin_counts, bin_edges = np.histogram(input_data['hand_vel_PDs'],bins=20,range=(-np.pi,np.pi))  
    bin_idx = np.digitize(input_data['hand_vel_PDs'],bins=bin_edges)
    bin_idx = bin_idx-1
    
    choice_p = 1/(bin_counts[bin_idx]/np.sum(bin_counts))
    choice_p[input_data['stim_chan_mask']==0] = 0 # only look at good chans
    
    choice_p = choice_p/np.sum(choice_p)
    # iterate through each trial
    for i_trial in range(input_data['n_trials']):
        # get stim chan(s)
        # get initial stim_chan randomly -- sample based on PD distribution to get uniform PD sampling
        
        temp_stim_chans = np.array([np.random.choice(n_neurons,p=choice_p)])
        ang_diff = np.abs(vae_utils.circular_diff(input_data['hand_vel_PDs'][temp_stim_chans],input_data['hand_vel_PDs']))
        ang_diff[input_data['stim_chan_mask']==0]=1000
        if(input_data['n_stim_chans']>1): # this only adds stimulation channels if doing multi-elec stim
            # get a random channel that is sufficiently similar to the original (temp_stim_chans[0])
            # sim scores in input_data['sim_mat'][temp_stim_chans[0],:]
            good_chans = np.reshape(np.argwhere(ang_diff <= input_data['PD_tol']),(-1,))
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
        # return [rates, samp_rates, samp_rates_stim, is_act, 
        #    hand_vel_true, hand_vel_stim, hand_vel_no,
        #    elbow_vel_true, elbow_vel_stim, elbow_vel_no]

        # package outputs    
        stim_chan_list[i_trial,:] = temp_stim_chans # stim channels

        # elbow and hand vel with and without stim 
        true_hand_vel_list.append(stim_out[4])
        no_hand_vel_list.append(stim_out[6])
        stim_hand_vel_list.append(stim_out[5])
        
        true_elbow_vel_list.append([7])
        no_elbow_vel_list.append([9])
        stim_elbow_vel_list.append([8])
        
        del stim_out
    
    # output metrics
    return [stim_chan_list, true_hand_vel_list, stim_hand_vel_list, no_hand_vel_list,
            true_elbow_vel_list, stim_elbow_vel_list, no_elbow_vel_list]


def compute_kin_metrics(stim_data, hand_vel_PDs, stim_start_idx, stim_end_idx, make_plots=False):
   
    
    stim_chan_list = stim_data[0]

    stim_hand_vel_list = stim_data[2]
    no_hand_vel_list = stim_data[3]
    
    
    delta_vel_stim = np.zeros((len(stim_hand_vel_list),2,2)) # trial, hand/elbow, vel-x, vel-y, vel-z
    #delta_vel_no = np.zeros_like(delta_vel_stim)
    
    delta_mag_stim = np.zeros((len(stim_hand_vel_list),2)) # trial, hand/elbow
    #delta_mag_no = np.zeros_like(delta_mag_stim)
    
    delta_dir_stim = np.zeros_like(delta_mag_stim)
    #delta_dir_no = np.zeros_like(delta_mag_stim)
    pred_delta_dir = np.zeros((len(stim_hand_vel_list),))
    
    for i_trial in range(len(stim_hand_vel_list)): # each stim trial
        # hand, get mean difference in velocity during stim
        delta_vel_stim[i_trial,0] = np.mean(stim_hand_vel_list[i_trial][stim_start_idx:stim_end_idx,:] - no_hand_vel_list[i_trial][stim_start_idx:stim_end_idx,:],axis=0)
        # elbow
        #delta_vel_stim[i_trial,1] = np.mean(point_kin_stim[i_trial][4][stim_start_idx:stim_end_idx,1:] - point_kin_no[i_trial][4][stim_start_idx:stim_end_idx,1:],axis=0)
        
        # compute magnitude
        delta_mag_stim[i_trial,:] = np.linalg.norm(delta_vel_stim[i_trial,:])
        #delta_mag_no[i_trial,:] = np.linalg.norm(delta_vel_no[i_trial,:])
        
        # compute angle
        delta_dir_stim[i_trial,:] = np.arctan2(delta_vel_stim[i_trial,:,1],delta_vel_stim[i_trial,:,0])
        #delta_dir_no[i_trial,:] = np.arctan2(delta_vel_no[i_trial,:,1],delta_vel_stim[i_trial,:,0])
        
        # get predicted stim dir
        pred_delta_dir[i_trial] = sp.stats.circmean(hand_vel_PDs[stim_chan_list[i_trial].astype(int)],low=-np.pi, high=np.pi)
   
    
    if(make_plots):
        plt.figure()
        #plt.hist(delta_mag_stim[:,0]-delta_mag_no[:,0])
        plt.hist(delta_mag_stim[:,0])
        
        plt.figure()
        plt.hist(abs(vae_utils.circular_diff(delta_dir_stim[:,0],pred_delta_dir[:])))
    
    
    return delta_vel_stim, delta_mag_stim, delta_dir_stim, pred_delta_dir



def run_amp_stim_exp(input_data):
    # run many stimulation trials for each amplitude, compute metrics for each amp, compare
    # amp_list contains list of amplitudes to test
    # hand_vel_PDs contains PDs
    
    stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
    stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)
    
    metrics = []
    for amp in input_data['amp_list']:
        input_data['amp'] = amp
        stim_exp_out=run_many_stim_trials(input_data)
        metrics.append(compute_kin_metrics(stim_exp_out, input_data['hand_vel_PDs'], stim_start_idx, stim_end_idx, make_plots=False))
    
    
    
    
    return metrics
    
    
def run_elec_amp_stim_exp(input_data):
    # run many stimulation trials for each amplitude, compute metrics for each amp, compare
    # amp_list contains list of amplitudes to test
    # n_stim_chans_list constains list of n_stim_chans to test
    # hand_vel_PDs contains PDs
    
    stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
    stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)
    
    input_data['stim_chan_mask'] = np.ones_like(input_data['hand_vel_PDs'])
    metrics = []
    n_list = []
    a_list = []
    for n_chans in input_data['n_stim_chans_list']:
        input_data['n_stim_chans'] = n_chans
        print(n_chans)
        for amp in input_data['amp_list']:
            print(amp)
            input_data['amp'] = amp
            stim_exp_out=run_many_stim_trials(input_data)
            metrics.append(compute_kin_metrics(stim_exp_out, input_data['hand_vel_PDs'], stim_start_idx, stim_end_idx, make_plots=False))
            n_list.append(n_chans)
            a_list.append(amp)
    
    
    return metrics, n_list, a_list,stim_exp_out




def run_amp_freq_stim_exp(input_data):
    # run many stimulation trials for each amplitude, compute metrics for each amp, compare
    # amp_list contains list of amplitudes to test
    # n_stim_chans_list constains list of n_stim_chans to test
    # hand_vel_PDs contains PDs
    
    stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
    input_data['stim_chan_mask'] = np.ones_like(input_data['hand_vel_PDs'])
    
    metrics = []
    freq_list = []
    a_list = []
    
    for i_freq in range(len(input_data['freq_list'])):
        input_data['freq'] = input_data['freq_list'][i_freq]
        input_data['n_pulses'] = input_data['n_pulses_list'][i_freq]
        stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)
        print(input_data['freq_list'][i_freq])
        for amp in input_data['amp_list']:
            print(amp)
            input_data['amp'] = amp
            stim_exp_out=run_many_stim_trials(input_data)
            metrics.append(compute_kin_metrics(stim_exp_out, input_data['hand_vel_PDs'], stim_start_idx, stim_end_idx, make_plots=False))
            freq_list.append(input_data['freq'])
            a_list.append(amp)
    
    
    return metrics, freq_list, a_list


def run_amp_high_sim_exp(input_data):
    
    stim_start_idx = np.round(input_data['stim_start_t']/gl.bin_size).astype(int)
    stim_end_idx = stim_start_idx + np.ceil(input_data['n_pulses']/input_data['freq']/gl.bin_size).astype(int)
    
    metrics = []
    a_list = []
    step_list = []
    stim_chan_list = []
    
    for step in input_data['step_list']:
        print(step)
        # get acceptable stim channels
        good_chans = np.argwhere(input_data['sim_score'] > input_data['sim_tol']).reshape(-1,)
        good_locs = input_data['map_data'][good_chans,:]
        # take a step of size step in random dir (N, E, S, W) from these good chans
        use_locs = good_locs
        
        for i_loc in range(len(good_locs)):
            i_dir = np.random.choice(4)
            temp_loc = good_locs[i_loc,:]
            if(i_dir==0): # N 
                if(temp_loc[1]-step >= 0):
                    temp_loc[1] = temp_loc[1]-step
                else:
                    temp_loc[1] = temp_loc[1]+step
            if(i_dir==1): # E
                if(temp_loc[0]+step < np.sqrt(input_data['map_data'].shape[0])):
                    temp_loc[0] = temp_loc[0]+step
                else:
                    temp_loc[0] = temp_loc[0]-step
            if(i_dir==2): # S
                if(temp_loc[1]+step < np.sqrt(input_data['map_data'].shape[0])):
                    temp_loc[1] = temp_loc[1]+step
                else:
                    temp_loc[1] = temp_loc[1]-step
            if(i_dir==3): # W
                if(temp_loc[0]-step >= 0):
                    temp_loc[0] = temp_loc[0]-step
                else:
                    temp_loc[0] = temp_loc[0]+step
            use_locs[i_loc] = temp_loc
        

        use_chans = vae_utils.convert_loc_to_idx(np.transpose(use_locs),input_data['map_data'])
        # update stim chan mask 
        input_data['stim_chan_mask'] = np.zeros_like(input_data['hand_vel_PDs'])
        input_data['stim_chan_mask'][use_chans.astype(int)] = 1
        
        
        for amp in input_data['amp_list']:
            input_data['amp'] = amp
            stim_exp_out=run_many_stim_trials(input_data)
            metrics.append(compute_kin_metrics(stim_exp_out, input_data['hand_vel_PDs'], stim_start_idx, stim_end_idx, make_plots=False))
            a_list.append(amp)
            step_list.append(step)
            stim_chan_list.append(stim_exp_out[0])
    
    
    return metrics, a_list, step_list, stim_chan_list









    







