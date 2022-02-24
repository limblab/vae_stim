# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:26:52 2021

@author: Joseph Sombeck
"""
import torch
import vae_model_code as mdl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import osim_utils as osim
import statsmodels.api as sm
import global_vars as gl

import stim_exp_utils
import scipy as sp

def load_vae_parameters(fpath, input_size):
    encoder = mdl.Encoder(input_size=input_size)
    decoder = mdl.Decoder(input_size=input_size)
    lateral = mdl.lateral_effect()
    
    vae = mdl.VAE(encoder,decoder,lateral)
    vae.load_state_dict(torch.load(fpath,map_location=torch.device('cpu')))
    return vae

def vae_decoder(vae,samples,bin_size):
    samples = torch.from_numpy(samples).type(torch.FloatTensor)
    out_sig = vae.decoder(samples*bin_size/gl.rate_mult)
    #out_sig = vae.decoder(samples)
    #out_sig = out_sig*bin_size/gl.rate_mult
    
    return out_sig.detach().numpy()

def vae_forward(vae, in_sig):
    # make sure to set eval mode so that we dont drop neurons out
    vae.eval()
    in_sig = torch.from_numpy(in_sig[:,:]).type(torch.FloatTensor)
    out_sig,rates = vae(in_sig)
    
    out_sig = out_sig.detach().numpy()
    rates = rates.detach().numpy()
    
    return out_sig,rates/gl.bin_size

def vae_get_rates(vae, in_sig,bin_size):
    # run encoder to get firing rates. Converts to Hz based on bin_size
    vae.eval()
    in_sig = torch.from_numpy(in_sig[:,:]).type(torch.FloatTensor)
    rates = vae.encoder(in_sig)
    rates = rates.detach().numpy()
    
    return rates

def sample_rates(rates):
    rates = torch.from_numpy(rates[:,:]).type(torch.FloatTensor)
    posterior = torch.distributions.Poisson(rates/gl.bin_size)
    samples = posterior.sample()
    samples = samples.detach().numpy()
    return gl.rate_mult*samples


class LinearDecoder(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_rate):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size, bias=True)
        )
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)

        return x

def make_linear_decoder(x, y, drop_rate=0.95, n_iters=1000,lr=0.001,init_weights=[]):
    
    # find decoder: x * dec + bias = y
    # dec shape = x.shape[1], y.shape[1]
    x_tr = torch.from_numpy(x[:,:]).type(torch.FloatTensor)
    y_tr = torch.from_numpy(y[:,:]).type(torch.FloatTensor)
    
    dec = LinearDecoder(x.shape[1],y.shape[1],drop_rate)

    # use initial weights (numpy array) if provided
    if(len(init_weights)>0):
        dec.state_dict()['layer1.0.weight'][:] = torch.Tensor(init_weights)

    dec.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dec.parameters(), lr=lr)
    
    for i_iter in range(n_iters):
        # get predictions and error
        yhat = dec(x_tr)
        recon_error = criterion(yhat,y_tr)
        # update weights
        optimizer.zero_grad()
        recon_error.backward()
        optimizer.step()
        
        # print current MSE
        if(i_iter % 50 == 0):
            print(str(i_iter) + ": " + str(recon_error.detach().numpy()))
    
    return dec
    
def make_constrained_linear_decoder(x, y, PDs, drop_rate=0.95, n_iters=1000,lr=0.001,init_weights=[]):
    
    # find decoder: x * dec + bias = y
    # dec shape = x.shape[1], y.shape[1]
    x_tr = torch.from_numpy(x[:,:]).type(torch.FloatTensor)
    y_tr = torch.from_numpy(y[:,:]).type(torch.FloatTensor)
        
    dec = LinearDecoder(x.shape[1],1,drop_rate)
        
    dec.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dec.parameters(), lr=lr)
    
    PD_mag = np.vstack((np.cos(PDs),np.sin(PDs)))
    mag_x = np.tile(PD_mag[0,:].reshape(1,-1),(x_tr.shape[0],1))
    mag_y = np.tile(PD_mag[1,:].reshape(1,-1),(x_tr.shape[0],1))
    x_tr_x = (x_tr*mag_x).type(torch.FloatTensor)
    x_tr_y = (x_tr*mag_y).type(torch.FloatTensor)
        
    for i_iter in range(n_iters):
        # get predictions and error for each direction
        yhat_x = dec(x_tr_x)
        yhat_y = dec(x_tr_y)
        
        yhat = torch.hstack((yhat_x,yhat_y))
        
        recon_error = criterion(yhat,y_tr)
        
        # update weights
        optimizer.zero_grad()
        recon_error.backward()
        optimizer.step()
        for p in dec.parameters():
            p.data.clamp_(0)
            
        # print current MSE
        if(i_iter % 50 == 0):
            print(str(i_iter) + ": " + str(recon_error.detach().numpy()))
    
    
    return dec


def linear_dec_forward(dec, x):
    
    dec.eval()
    yhat = dec(torch.from_numpy(x).type(torch.FloatTensor))
    yhat = yhat.detach().numpy()    
    return yhat


def linear_constrained_dec_forward(dec, x, PDs):
    PD_mag = np.vstack((np.cos(PDs),np.sin(PDs)))
    mag_x = np.tile(PD_mag[0,:].reshape(1,-1),(x.shape[0],1))
    mag_y = np.tile(PD_mag[1,:].reshape(1,-1),(x.shape[0],1))
    
    y_hat_x = linear_dec_forward(dec=dec,x=x*mag_x).reshape((-1,1))
    y_hat_y = linear_dec_forward(dec=dec,x=x*mag_y).reshape((-1,1))

    y_hat = np.hstack((y_hat_x,y_hat_y))
    
    return y_hat

def get_decoder_params(dec):
    param_list = []
    for params in dec.parameters():
        param_list.append(params.detach().numpy())
            
    return param_list


def convert_list_to_map(data,mapping):
    data_as_map = np.zeros((np.max(mapping[:,0])+1,np.max(mapping[:,1])+1,data.shape[0]))
    
    for i_time in range(data.shape[0]):
        for i_data in range(data.shape[1]):
            data_as_map[mapping[i_data,0],mapping[i_data,1],i_time] = data[i_time,i_data]
        
    return data_as_map

def visualize_activation_map(stim_rates, no_stim_rates, idx=1):
    # visualize activation as a heatmap
    mapping = mdl.locmap().astype(int)
    no_stim_rates_map = convert_list_to_map(no_stim_rates,mapping)
    stim_rates_map = convert_list_to_map(stim_rates,mapping)

    fig, ax = plt.subplots(1,2)
    x=ax[0].imshow(no_stim_rates_map[:,:,idx])
    fig.colorbar(x,ax=ax[0])
    
    x=ax[1].imshow(stim_rates_map[:,:,idx],vmin=0)
    fig.colorbar(x,ax=ax[1])
    
    #x=ax[2].imshow(stim_rates_map[:,:,idx]-no_stim_rates_map[:,:,idx])
    #fig.colorbar(x,ax=ax[2])
    
    return no_stim_rates_map, stim_rates_map
    
def visualize_activation_dist(is_act, stim_chans, block_size):
    # visualize activation function (change in FR vs. distance)
    num_act = np.sum(is_act==1,axis=1)
    mapping = mdl.locmap()
    
    dist_to_stim = 100000*np.ones(num_act.shape)
    
    for i_chan in stim_chans:
        stim_chan_loc = mapping[i_chan,:]
        temp_dist = euclidean_distances(np.transpose(stim_chan_loc.reshape(-1,1)), mapping)*block_size
        dist_to_stim = np.minimum(dist_to_stim, temp_dist)
    
    dist_to_stim = dist_to_stim.reshape(-1,)
    
    # bin neurons based on their distance to stim, get probability of activation for neurons in each distance bin
    dist_edges = np.arange(0,10,0.05) 
    bin_idx = np.digitize(dist_to_stim,dist_edges) # 0 corresponds to below the first bin, 1 is the first bin
    bin_idx[bin_idx==0]=1
    bin_idx = bin_idx-1
    prob_act_bin = np.zeros(dist_edges.shape[0] - 1)
    
    for i_bin in range(prob_act_bin.shape[0]):
        prob_act_bin[i_bin] = np.sum(num_act[bin_idx==i_bin])/np.sum(bin_idx==i_bin)/is_act.shape[1] # number of pulses
    
    plt.figure()
    plt.plot(dist_edges[0:-1]+np.mean(np.diff(dist_edges))/2,prob_act_bin)
    
    return 0
 
    
def integrate_vel(kin_0, vels, bin_size):
    
    int_kin = np.zeros((vels.shape[0],len(kin_0)))
    int_kin[0] = kin_0
    
    for i in range(1,vels.shape[0]):
        int_kin[i,:] = int_kin[i-1] + vels[i-1]*gl.bin_size
    
    return int_kin

def get_correlation_similarity(vae,joint_vels_norm):
    rates = vae_get_rates(vae,joint_vels_norm,gl.bin_size)
    corr_sim_mat = np.corrcoef(rates,rowvar=False)
    
    return corr_sim_mat


def get_PD_similarity(vae,joint_vels_norm,hand_vels):
    rates = vae_get_rates(vae,joint_vels_norm,gl.bin_size)

    glm_in = hand_vels
    glm_in = sm.add_constant(glm_in,prepend=False)
    
    hand_vel_PDs = np.zeros(rates.shape[1])
    hand_vel_params = np.zeros((rates.shape[1],2))
    for i_unit in range(rates.shape[1]):
        if(i_unit % 100 == 0):
            print(i_unit)
            
        glm_out = rates[:,i_unit]
        glm_PD_mdl = sm.GLM(glm_out,glm_in,family=sm.families.Poisson(sm.families.links.log()))
        res = glm_PD_mdl.fit()
        # 0,1,2 = hand pos; 3,4 = hand vel (x,y); 5,6 = hand vel z, constant
        hand_vel_PDs[i_unit] = np.arctan2(res.params[1],res.params[0])
        hand_vel_params[i_unit,0] = res.params[0]
        hand_vel_params[i_unit,1] = res.params[1]
    
    
    return hand_vel_PDs, hand_vel_params
    
def get_hash_PDs(vae, joint_vels_norm, hand_vels, locs):
    rates = vae_get_rates(vae,joint_vels_norm,gl.bin_size)

    glm_in = hand_vels
    glm_in = sm.add_constant(glm_in,prepend=False)
    
    hash_PDs = np.zeros(rates.shape[1])
    hash_params = np.zeros((rates.shape[1],2))
    hash_rates = np.zeros_like(rates)
    
    # get distance matrix
    dist_mat = euclidean_distances(locs)
    #np.fill_diagonal(dist_mat,10000)
    #min_dist = np.min(dist_mat)
    #np.fill_diagonal(dist_mat,min_dist*2/3)
    
    for i_unit in range(rates.shape[1]):
        if(i_unit % 100 == 0):
            print(i_unit)
            
        # sum rates across neighboring neurons (as if we did a multi-unit hash instead of single neuron)
        # do sum(1/dist^2 * rates), dist is dist from current location
       
        #dist_factor = 1/np.square(dist_mat[i_unit,:])

        
        
        # do linear activation. 0mm = 0.12, 0.160mm = 0
        # slope = -0.12/0.16 mm
        # b = 0.12
        
        dist_factor = 0.12*(1-dist_mat[i_unit,:]/0.16)
        
        dist_factor[dist_factor < 0] = 0
        dist_factor = dist_factor.reshape(-1,1)
        
        
        hash_rates[:,i_unit] = np.matmul(rates,dist_factor).reshape((-1,))
        
        # set hash rates as glm out and fit model
        glm_out = hash_rates[:,i_unit]
        glm_PD_mdl = sm.GLM(glm_out,glm_in,family=sm.families.Poisson(sm.families.links.log()))
        res = glm_PD_mdl.fit()
        
        # store parameters
        # 0,1,2 = hand pos; 3,4 = hand vel (x,y); 5,6 = hand vel z, constant
        hash_PDs[i_unit] = np.arctan2(res.params[1],res.params[0])
        hash_params[i_unit,0] = res.params[0]
        hash_params[i_unit,1] = res.params[1]
        
    return hash_PDs, hash_params, rates, hash_rates

def get_neighbor_sim(loc_map, sim_map, max_neigh_dist):
    
    # for each neuron, get similarity between it and it's neighbors
    neigh_sim = np.zeros((loc_map.shape[0],))
    
    for i_unit in range(len(neigh_sim)):
        dist_mat = euclidean_distances(loc_map[i_unit,:].reshape(1,-1), loc_map)
        dist_mat = dist_mat.reshape((-1,))
        dist_mat[i_unit] = 10000 # set current unit's distance as absurdly large
        neigh_idx = np.argwhere(dist_mat <= max_neigh_dist)
        sim_scores = sim_map[i_unit,neigh_idx]
        neigh_sim[i_unit] = np.mean(sim_scores)
    
    return neigh_sim
        

def get_pd_neighbor_dist(loc_map, pds, max_neigh_dist):
    neigh_pd_diff = []
    neigh_dist = []
    non_neigh_pd_diff = []
    non_neigh_dist = []
    
    for i_unit in range(len(loc_map)):
        dists = euclidean_distances(loc_map[i_unit,:].reshape(1,-1),loc_map)
        dists=dists.reshape(-1,)
        angle_diffs = circular_diff(pds[i_unit],pds)
        for j_unit in range(i_unit+1,len(loc_map)):
            if(dists[j_unit]<=max_neigh_dist):
                # is a neightbor
                neigh_dist.append(dists[j_unit])
                neigh_pd_diff.append(angle_diffs[j_unit])
            else:
                non_neigh_dist.append(dists[j_unit])
                non_neigh_pd_diff.append(angle_diffs[j_unit])
    
    return neigh_pd_diff, neigh_dist, non_neigh_pd_diff, non_neigh_dist

def get_pd_dist(loc_map, pds):
    dists = euclidean_distances(loc_map)
    
    pd_vec = np.array([np.cos(pds),np.sin(pds)])
    cos_sim = cosine_similarity(np.transpose(pd_vec))
    cos_sim[cos_sim>1] = 1
    cos_sim[cos_sim<-1] = -1
    ang_diff = np.arccos(cos_sim)
    
    dists = dists.reshape((-1,))
    ang_diff = ang_diff.reshape((-1,))
    
    return dists, ang_diff
    
def circular_diff(data_1, data_2):
    # expects data in radians
    diff = data_1 - data_2
    
    max_diff = np.pi
    
    if hasattr(diff,"__len__"):
        diff[diff<-max_diff] = diff[diff<-max_diff] + 2*max_diff
        diff[diff>max_diff] = diff[diff>max_diff] - 2*max_diff
    elif(diff > max_diff):
        diff = diff - 2*max_diff
    elif(diff < -max_diff):
        diff = diff + 2*max_diff
        
    return diff
    
    
def convert_loc_to_idx(locs, mapping):
    idx_list = np.zeros((locs.shape[1],))
    
    for i in range(locs.shape[1]):
        curr_loc = locs[:,i]
        idx_list[i] = np.argwhere(np.all(curr_loc == mapping,1))
        
        
    return idx_list
    
    

def compute_multielec_pred(stim_chans_loc,map_data,amp,hand_vel_PDs,elec_space=8): # dist in blocks, each block is 50 um
    
    x = np.arange(0, 80, elec_space, dtype=np.float32) 
    x = np.append(x,-1*x[1:])
    y = np.arange(0, 80, elec_space, dtype=np.float32)
    y = np.append(y,-1*y[1:])
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv, (xv.size, 1))
    yv = np.reshape(yv, (yv.size, 1))
    grid = np.hstack((xv, yv))
    
    
    dist_all = np.array([])
    PD_all = np.array([])
    
    for i in range(stim_chans_loc.shape[0]):
        curr_loc = stim_chans_loc[i,:]
        
        # mask based on electrode_spacing (make a grid, shift grid onto curr_loc)
        grid_curr = grid + curr_loc
        # remove any entries that are negative or bigger than 80
        keep_mask = np.all(np.logical_and(grid_curr < 80,  grid_curr >= 0),axis=1)
        grid_curr = grid_curr[keep_mask==1,:]
        
        grid_idx = convert_loc_to_idx(np.transpose(grid_curr), map_data).astype(int)
        
        # get distance from stim electrode for all neurons
        dist_mat = euclidean_distances(curr_loc.reshape(1,-1), map_data).reshape(-1,1) # in blocks
        # then only use those in the correct grid
        dist_mat = dist_mat[grid_idx]
        PDs_use = hand_vel_PDs[grid_idx]
        
        # store PDs and distance from stim electrode
        dist_all = np.append(dist_all,dist_mat)
        PD_all = np.append(PD_all, PDs_use)
    
    # make prediction across neurons, accounting for distance and amplitude of stim
    prob_act = stim_exp_utils.get_prob_act(100, dist_all*0.05) # convert distance to mm
    
    # take a weighted mean based on prob act and PD distribution. scipy.circ_mean does not have a weight, so instead we will make a matrix with # entries corresponding to weight
    PDs_as_weights = np.vstack((np.cos(PD_all), np.sin(PD_all)))

    # normalize PDs_as_weights based on PD distribution
    PD_distribution, bin_edges = np.histogram(hand_vel_PDs,20)
    bin_idx = np.digitize(PD_all, bin_edges) - 1
    bin_idx[bin_idx < 0] = 0
    bin_idx[bin_idx >= len(PD_distribution)] = len(PD_distribution)-1
    n_bin = PD_distribution[bin_idx]
    
    PDs_as_weights = PDs_as_weights/n_bin*6400
    weight = np.sqrt(np.sum(np.square(PDs_as_weights),axis=0))


    num_reps = np.round_(100*np.multiply(prob_act.reshape((-1,)),weight), decimals=0).reshape((-1,))
    
    PD_rep = np.repeat(PD_all, num_reps.astype(int))
    


    return sp.stats.circmean(PD_rep,low=-np.pi,high=np.pi), grid_idx, dist_all, PD_all
    
    
    
    
def compute_histogram_overlap(data1, data2,bin_size=5): 
    # this function bins the data in [min(data1, data2), max(data1,data2)] using bin_size
    # then computes distribution overlap based on those bins (summing the min value in each bin)
    # normalize by bin size such that the same data produces an overlap of 1.
    
    
    data_range = [np.min(np.vstack((data1,data2))), np.max(np.vstack((data1,data2)))]

    bin_edges = np.arange(data_range[0],data_range[1]+bin_size,bin_size)

    data1_hist,bin_edges = np.histogram(data1,bins=bin_edges)
    data2_hist,bin_edges = np.histogram(data2,bins=bin_edges)
    
    data1_hist = data1_hist/len(data1)
    data2_hist = data2_hist/len(data2)
    
    perc_over = np.sum(np.minimum(data1_hist,data2_hist))
    
    
    return perc_over, data1_hist,data2_hist, bin_edges