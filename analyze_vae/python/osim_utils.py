# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:11:15 2021

@author: Joseph Sombeck
"""

import opensim as osim
import numpy as np
import global_vars as gl

def write_mot_file(data):
    # write .mot file with joint vel data for opensim
    f = open(gl.osim_dir + '\\' + gl.mot_fname,'w') # overwrite contents
    # write header
    f.write("Coordinates\n")
    f.write("version=1\n")
    f.write("nRows=%i\n" % data.shape[0])
    f.write("nColumns=%i\n" % (data.shape[1]+1)) #add 1 for time
    f.write("inDegrees=yes\n\n") # two new lines
    f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
    f.write("Angles are in degrees.\n\n") # two new lines
    f.write("endheader\n")
    
    # write column names (time, [joint_ang_names]), tab delimiter
    f.write("time\t"); f.write("shoulder_adduction\t"); f.write("shoulder_rotation\t");
    f.write("shoulder_flexion\t"); f.write("elbow_flexion\t"); f.write("radial_pronation\t"); f.write("wrist_flexion\t"); f.write("wrist_abduction\n");
    
    # write data, tab delimiter, rows are time points. Need to generate time stamps....
    t_write = np.arange(0,data.shape[0]+0)*gl.bin_size
    
    for i_row in range(data.shape[0]):
        str_to_write = "%.7f\t" % t_write[i_row] # write time
        for i_col in range(data.shape[1]):
            str_to_write = str_to_write + "%.7f\t" % data[i_row,i_col] # write data
        str_to_write = str_to_write + "\n" # go to next line
    
        f.write(str_to_write)
    
    f.close()
    
    
def run_osim_analysis():    
    # put joint vels through opensim
    arm = osim.Model(gl.path_to_osim_model)
    
    osim_analysis = osim.AnalyzeTool(gl.path_to_analysis_file)
    motion = osim.Storage(gl.osim_dir + '\\' + gl.mot_fname)
    
    osim_analysis.setInitialTime(motion.getFirstTime())
    osim_analysis.setFinalTime(motion.getLastTime())
    
    osim_analysis.setModel(arm)
    osim_analysis.setModelFilename(gl.path_to_osim_model)
    osim_analysis.setCoordinatesFileName(gl.osim_dir + '\\' + gl.mot_fname)
    osim_analysis.setLoadModelAndInput(True)
    osim_analysis.setResultsDir(gl.osim_dir)
    osim_analysis.run()
    
def get_osim_pointkin_results():
    
    fname_pref = "subject01_walk1_PointKinematics_"
    fname_postf = ".sto"
    
    # rotate axes to match handle --- opensim: 0 = t, 1 = handle-y; 2 = handle-z; 3 = handle-x
    # so hand_vel = hand_vel[0,3,1,2]
    
    # get elbow kinematics
    elbow_pos = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'elbow_pos' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    elbow_vel = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'elbow_vel' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    elbow_acc = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'elbow_acc' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    # get hand kinematics
    hand_pos = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'hand_pos' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    hand_vel = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'hand_vel' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    hand_acc = np.genfromtxt(gl.osim_dir + '\\' + fname_pref + 'hand_acc' + fname_postf, delimiter='\t', skip_header=8)[:,[0,3,1,2]]
    
    return [hand_pos,hand_vel,hand_acc,elbow_pos,elbow_vel,elbow_acc]
    

def get_pointkin(data): # data should be joint angles, not velocities...
    write_mot_file(data) # write data to mot file
    run_osim_analysis() # run analysis on mot file
    
    return get_osim_pointkin_results() # read in results and return
