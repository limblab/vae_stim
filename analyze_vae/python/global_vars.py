# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:55:59 2021

@author: Joseph Sombeck
"""

import yaml

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


def global_vars():
    global params;
    global bin_size
    global path_to_osim_model
    global path_to_analysis_file
    global mot_fname
    global osim_dir
    global results_fname
    
    params = Params;
    bin_size = 0.05; # s
    path_to_osim_model = r'D:\Lab\GIT\monkeyArmModel\monkeyArm_current.osim'
    path_to_analysis_file = r'D:\Lab\GIT\monkeyArmModel\matlab_pointKin_settings.xml'
    mot_fname = r'temp.mot'
    osim_dir = r'D:\Lab\GIT\monkeyArmModel\Analysis'
    results_fname = r'D:\Lab\GIT\monkeyArmModel\Analysis\Temp.xml'