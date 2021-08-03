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

def load_vae_parameters(fpath, input_size):
    encoder = mdl.Encoder(input_size=input_size)
    decoder = mdl.Decoder(input_size=input_size)
    lateral = mdl.lateral_effect()
    
    vae = mdl.VAE(encoder,decoder,lateral)
    vae.load_state_dict(torch.load(fpath,map_location=torch.device('cpu')))
    return vae


def vae_forward(vae, in_sig):
    # make sure to set eval mode so that we dont drop neurons out
    vae.eval()
    in_sig = torch.from_numpy(in_sig[:,:]).type(torch.FloatTensor)
    out_sig,rates = vae(in_sig)
    
    out_sig = out_sig.detach().numpy()
    rates = rates.detach().numpy()
    
    return out_sig,rates

def vae_get_rates(vae, in_sig,bin_size):
    # run encoder to get firing rates. Converts to Hz based on bin_size
    vae.eval()
    in_sig = torch.from_numpy(in_sig[:,:]).type(torch.FloatTensor)
    rates = vae.encoder(in_sig)
    rates = rates.detach().numpy()
    
    return rates/bin_size

def sample_rates(rates):
    rates = torch.from_numpy(rates[:,:]).type(torch.FloatTensor)
    posterior = torch.distributions.Poisson(rates*gl.params.params['n_samples'])
    samples = posterior.sample()/gl.params.params['n_samples']
    return samples.detach().numpy()


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

def make_linear_decoder(x, y, drop_rate=0.95, n_iters=1000,lr=0.001):
    
    # find decoder: x * dec + bias = y
    # dec shape = x.shape[1], y.shape[1]
    x_tr = torch.from_numpy(x[:,:]).type(torch.FloatTensor)
    y_tr = torch.from_numpy(y[:,:]).type(torch.FloatTensor)
    
    dec = LinearDecoder(x.shape[1],y.shape[1],drop_rate)

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
    
def linear_dec_forward(dec, x):
    
    dec.eval()
    yhat = dec(torch.from_numpy(x).type(torch.FloatTensor))
    yhat = yhat.detach().numpy()    
    return yhat


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

    plt.figure()
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(no_stim_rates_map[:,:,idx])
    ax[1].imshow(stim_rates_map[:,:,idx])
    ax[2].imshow(stim_rates_map[:,:,idx]-no_stim_rates_map[:,:,idx])
    
    return 0
    
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
    dist_edges = np.arange(0,10,0.1) 
    bin_idx = np.digitize(dist_to_stim,dist_edges) # 0 corresponds to below the first bin, 1 is the first bin
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


def get_PD_similarity(vae,joint_vels_norm,joint_angs):
    rates = vae_get_rates(vae,joint_vels_norm,gl.bin_size)
    point_kin = osim.get_pointkin(joint_angs)
    hand_vels = point_kin[1][:,1:-1] # remove time column and z-axis
        
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
    
    
    PD_sim_mat = cosine_similarity(hand_vel_params)
    return PD_sim_mat, hand_vel_PDs, hand_vel_params
    
        
    
def circular_diff(data_1, data_2):
    
    diff = data_1 - data_2
    
    max_diff = np.pi
    
    diff[diff<-max_diff] = diff[diff<-max_diff] + 2*max_diff
    diff[diff>max_diff] = diff[diff>max_diff] - 2*max_diff
    
    return diff
    
    
    
    
    
    
    
    
    
    
    
    
    