# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:26:52 2021

@author: Joseph Sombeck
"""
import yaml
import torch
import vae_model_code as mdl
import numpy as np

def load_model_params():
    return 1


class Params:
    params={}
    def __init__(self):
        # initialize parameters
        self.params = {'batch_size' : 1024,
            'latent_shape' : [80,80],
            'n_epochs' : 3000,
            'sampling' : 'poisson',
            'n_samples' : 20,
            'layers' : [20,40],
            'cuda' : True,        
            'sigma' : 2.0,
            'eta' : 0.00001,
            'lateral' : 'mexican',
            'lambda_l' : 0,
            'default_rate' : 3.0, # the expected number of spikes in each bin
            'save_path' : '/home/jts3256/projects/stimModel/models',
            'dropout' : 99, # as a percentage
        }
        
        
    def load_params(self,filename):
        # load in parameters from a file. Return dictionary of parameters
        with open(filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        # set integer parameters as integers
        self.params=config
        
    def save_params(self,filename):
        # write parameters to a  file
        with open(filename,'w') as f:
            doc = yaml.dump(self.params,f)


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

def global_vars():
    global params;
    params = Params;
    global bin_size
    bin_size = 0.05; # s
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    