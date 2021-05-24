%% look at effect of stimulation on whole arm kinematics. 
    % pick a trial (start idx) and other inputs
    start_idx = 50; 
    stim_lag = 20;
    stim_len = 0.2; % s
    
    stim_effect_input = [];
    stim_effect_input.FR = td.VAE_firing_rates(stim_bins,:);
    stim_effect_input.dec = dec;
    stim_effect_input.act_func = 'exp_decay';
    stim_effect_input.locs = locs*0.06; % mm per block
    
    stim_effect_input.stim_loc = [1,1];
    stim_effect_input.amp = 20; % uA
    stim_effect_input.pulse_bin_idx = reshape(repmat([1:1:numel(stim_bins)],20,1),1,[]);
    stim_effect_input.n_pulses = numel(stim_effect_input.pulse_bin_idx);
    stim_effect_input.bin_size = td.bin_size;
    
    analysis_data = [];
    analysis_data.data_dir = 'D:\Lab\Data\StimModel\opensim_data';
    analysis_data.settings_path = 'D:\Lab\GIT\monkeyArmModel';
    analysis_data.settings_fname = 'matlab_pointKin_settings.xml';
    analysis_data.mot_path = 'D:\Lab\Data\StimModel\opensim_data\IKResults';
    analysis_data.mot_name = 'test.mot';
    analysis_data.joint_names = td.joint_names;
    analysis_data.in_deg = 1;
    
    % code below, get stim bins and so on.
    stim_bins = start_idx + stim_lag+[1:ceil(stim_len/td.bin_size)] - 1;
    stim_bins_adj = stim_lag+[1:ceil(stim_len/td.bin_size)];
    end_idx = stim_bins(end)+5;
    
    % decode joint angles without stim
    dec_joint_vel_no_stim = predictData(td.VAE_firing_rates(start_idx:end_idx,:),dec,bias);
    dec_joint_ang_no_stim = integrateVel(td.joint_ang(start_idx,:),dec_joint_vel_no_stim,td.bin_size);
    
    % apply stim, decode joint angles with stim
    stim_effect = getStimEffect(stim_effect_input);
    
    VAE_FR = td.VAE_firing_rates(start_idx:end_idx,:);
    VAE_FR(stim_bins_adj,:) = stim_effect.FR_act;
    
    dec_joint_vel = predictData(VAE_FR,dec,bias);
    dec_joint_ang = integrateVel(td.joint_ang(start_idx,:),dec_joint_vel,td.bin_size);   
 
    % no stim
    analysis_data.joint_ang = dec_joint_ang_no_stim;
    analysis_data.t = ((1:1:size(dec_joint_ang_no_stim,1))-1)*td.bin_size;
    point_kin_no = getPointKinematics(analysis_data);
    
    % stim
    analysis_data.joint_ang = dec_joint_ang;
    analysis_data.t = ((1:1:size(dec_joint_ang,1))-1)*td.bin_size;
    point_kin_stim = getPointKinematics(analysis_data);
    
% visualize in a 3D plot.
    % plot hand and elbow locations right before and after stim for no stim
    % and stim condition
    
    figure(); hold on;
    
    % plot pre-stim hand and elbow position
    idx_plot = stim_bins_adj(1);
    to_plot = [0,0,0; ...
        point_kin_no.elbow_pos.X(idx_plot), point_kin_no.elbow_pos.Y(idx_plot), point_kin_no.elbow_pos.Z(idx_plot); ...
        point_kin_no.hand_pos.X(idx_plot), point_kin_no.hand_pos.Y(idx_plot), point_kin_no.hand_pos.Z(idx_plot)];
    
    plot3(to_plot(:,1),to_plot(:,2),to_plot(:,3),'k.','markersize',20,'linestyle','-','linewidth',2)
    
    % plot no stim, post-stim hand and elbow position
    idx_plot = stim_bins_adj(2)+1;
    to_plot = [0,0,0; ...
        point_kin_no.elbow_pos.X(idx_plot), point_kin_no.elbow_pos.Y(idx_plot), point_kin_no.elbow_pos.Z(idx_plot); ...
        point_kin_no.hand_pos.X(idx_plot), point_kin_no.hand_pos.Y(idx_plot), point_kin_no.hand_pos.Z(idx_plot)];
    
    plot3(to_plot(:,1),to_plot(:,2),to_plot(:,3),'.','color',[0.5,0.5,0.5],'markersize',20,'linestyle','-','linewidth',2)
    
    % plot stim, post-stim hand and elbow position
    idx_plot = stim_bins_adj(2)+1;
    to_plot = [0,0,0; ...
        point_kin_stim.elbow_pos.X(idx_plot), point_kin_stim.elbow_pos.Y(idx_plot), point_kin_stim.elbow_pos.Z(idx_plot); ...
        point_kin_stim.hand_pos.X(idx_plot), point_kin_stim.hand_pos.Y(idx_plot), point_kin_stim.hand_pos.Z(idx_plot)];
    
    plot3(to_plot(:,1),to_plot(:,2),to_plot(:,3),'.','color',getColorFromList(1,0),'markersize',20,'linestyle','-','linewidth',2)
    
    
    
    
    
    
    