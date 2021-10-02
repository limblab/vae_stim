# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:55:59 2021

@author: Joseph Sombeck
"""

import yaml
import opensim as osim
import numpy as np

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
    global osim_analysis 
    global arm
    global karth_act_prop
    global karth_dist
    global karth_amp
    global rate_mult
    
    params = Params;
    bin_size = 0.05; # s
    path_to_osim_model = r'D:\Lab\GIT\monkeyArmModel\monkeyArm_current.osim'
    path_to_analysis_file = r'D:\Lab\GIT\monkeyArmModel\matlab_pointKin_settings.xml'
    mot_fname = r'temp.mot'
    osim_dir = r'D:\Lab\GIT\monkeyArmModel\Analysis'
    results_fname = r'D:\Lab\GIT\monkeyArmModel\Analysis\Temp.xml'
    
    osim_analysis = osim.AnalyzeTool(path_to_analysis_file)
    arm = osim.Model(path_to_osim_model)
    osim_analysis.setModel(arm)
    
    osim_analysis.setModelFilename(path_to_osim_model)
    osim_analysis.setCoordinatesFileName(osim_dir + '\\' + mot_fname)
    osim_analysis.setLoadModelAndInput(True)
    osim_analysis.setResultsDir(osim_dir)

    karth_act_prop = np.genfromtxt(r'D:\Lab\GIT\vae_stim\analyze_biophysical_mdl\activation_probability.txt',delimiter=',')
    karth_dist = np.genfromtxt(r'D:\Lab\GIT\vae_stim\analyze_biophysical_mdl\distance_bins.txt',delimiter=',')
    karth_amp = np.genfromtxt(r'D:\Lab\GIT\vae_stim\analyze_biophysical_mdl\amplitude.txt',delimiter=',')
    
    rate_mult = 10
    