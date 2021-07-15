function [output_data] = runVAEOpenSimExperiment(input_data, opensim_data)
% How does the direction of stim effect compare to the PD of the stimulated
% location during single electrode stimulation?

    % Independent Variables: Amplitude, Activation Function, Frequency. 

    % Dependent Variables : Direction and magnitude of stim effect,
    % measured from predicted hand velocity from the linear decoder

    % Will do simulations for multiple locations, can then sanity check
    % with a neighborhood metric.
    % Will also do simulations for multiple hand kinematics, which will
    % affect firing rate of neurons, and thus the effect of stim.

% input_data contains:
% td : trial data structure
% n_moves : number of movement trials to test
% n_locs : number of stimulation locations to test
% amps_test : amplitude(s) of stimulation to test
% freqs_test : frequenc(ies) of stimulation to test
% direct_acts_test : activation functions to test
% locs : location of neurons
% dec : decoder from firing rate to hand velocity
% bias : decoder bias;

    stim_len = floor(input_data.train_length/input_data.td.bin_size);
    
    % pick stimulation locations
    loc_idx = datasample(1:1:size(input_data.locs,1),input_data.n_locs,'Replace',false);
    stim_locs = input_data.locs(loc_idx,:);
    
    % pick stim start times
    stim_start_idx = datasample(1:1:(size(input_data.td.vel,1)-10*stim_len),input_data.n_moves,'Replace',false);
        
    %% setup experiment    
    % for each amplitude, frequency, and activation function, location,
    % movement, get stimulation effect
    % initialize output matrices for experiment
    n_out = numel(input_data.direct_acts_test)*numel(input_data.trans_acts_test)*numel(input_data.amps_test)*...
        numel(input_data.freqs_test)*input_data.n_moves*input_data.n_locs;
    
    n_bins = ceil(input_data.train_length/input_data.td.bin_size);
    stim_vel = zeros(n_out,n_bins,size(input_data.dec,2));
    base_vel = zeros(size(stim_vel));
    num_pulses_active = zeros(n_out, size(input_data.locs,1));
    amp_list = zeros(n_out,1);
    dir_act_func = cell(n_out,1);
    trans_act_func = cell(n_out,1);
    dir_act_fact = zeros(n_out,1);
    freq_list = zeros(size(amp_list));
    move_idx = zeros(size(amp_list));
    stim_loc = zeros(n_out,2);
    point_kin_no = cell(n_out,1);
    point_kin_stim = cell(n_out,1);
    
    % initialize params for experiment
    stim_effect_data = [];
    stim_effect_data.locs = input_data.locs;
    stim_effect_data.bin_size = input_data.td.bin_size;
    stim_effect_data.dec = input_data.dec;
    stim_effect_data.bias = input_data.bias;
    stim_effect_data.corr_table = input_data.corr_table;
    stim_effect_data.block_size = input_data.block_size;
    stim_effect_data.PD = input_data.PD;
    
    % initialize opensim -- NOTE: this has been taken from
    % getPointKinematics.m because of an apparent memory leak caused by running
    % opensim in a function hundreds of time
    if(input_data.do_opensim)
        % Pull in the modeling classes straight from the OpenSim distribution
        import org.opensim.modeling.*
        % Go to the folder in the subject's folder where IK Results are
        ik_results_folder = fullfile(opensim_data.data_dir, 'IKResults');
        % specify where setup files will be printed.
        setupfiles_folder = fullfile(opensim_data.data_dir, 'AnalyzeSetup');
        % specify where results will be printed.
        opensim_data.results_folder = fullfile(opensim_data.data_dir, 'AnalyzeResults');
        % Get and operate on the files
        genericSetupForAn = [opensim_data.settings_path filesep opensim_data.settings_fname];
        % make analysis tool
        analyzeTool = AnalyzeTool(genericSetupForAn);
    end
    
    counter = 1;
    for i_dir_act = 1:numel(input_data.direct_acts_test)
        stim_effect_data.direct_act_func = input_data.direct_acts_test{i_dir_act};
        for i_trans_act = 1:numel(input_data.trans_acts_test)
            stim_effect_data.trans_act_func = input_data.trans_acts_test{i_trans_act};
            for i_dir_fact = 1:numel(input_data.dir_act_fact)
                stim_effect_data.dir_act_fact = input_data.dir_act_fact(i_dir_fact);
                for i_amp = 1:numel(input_data.amps_test)
                    stim_effect_data.amp = input_data.amps_test(i_amp);
                    for i_freq = 1:numel(input_data.freqs_test)

                        % get which bin in the stim train each pulse is in
                        stim_timing = 0:1/input_data.freqs_test(i_freq):input_data.train_length;
                        bin_edges = [0:1:ceil(input_data.train_length/input_data.td.bin_size)]*input_data.td.bin_size;
                        [~,~,stim_effect_data.pulse_bin_idx] = histcounts(stim_timing, bin_edges);
                        stim_effect_data.n_pulses = numel(stim_timing);

                        for i_loc = 1:input_data.n_locs
                            stim_effect_data.can_be_act = [];
                            for i_move = 1:input_data.n_moves
                                stim_effect_data.FR = input_data.td.VAE_firing_rates(stim_start_idx(i_move):stim_start_idx(i_move)+stim_len-1,:);
                                stim_effect_data.stim_loc = stim_locs(i_loc,:);
                                
                                % get 4 stimulation locations, stack them.
                                % Require PD to be within 20 degs of
                                % stimulated electrode
                                stim_loc_idx = getStimLocationIdx(stim_locs(i_loc,:), input_data.locs);
                                stim_loc_PD = input_data.PD(stim_loc_idx);
                                
                                % get list of locations with simlar PDs,
                                PD_diff = input_data.PD - stim_loc_PD;
                                suitable_elecs = find(abs(PD_diff)<20*pi/180);
                                % remove stim loc
                                suitable_elecs(suitable_elecs==stim_loc_idx) = [];
                                
                                extra_stim_elecs = datasample(suitable_elecs,3,'Replace',false);
                                
                                stim_effect_data.stim_loc = input_data.locs([stim_loc_idx;extra_stim_elecs],:);
                                
                                stim_effect_temp = getStimEffect(stim_effect_data);

                                stim_effect_data.stim_loc = stim_locs(i_loc,:); % this is a hack to do multi-electrode stim....
                                % get activation pattern if i_move = 1;
                                if(i_move==1)
                                    stim_effect_data.can_be_act = stim_effect_temp.can_be_act;
                                end

                                % store velocity in each bin
                                stim_vel(counter,:,:) = stim_effect_temp.stim_vel;
                                base_vel(counter,:,:) = stim_effect_temp.base_vel;

                                % if doing opensim, predict 
                                if(input_data.do_opensim)
                                    % get initial joint ang
                                    init_joint_ang = input_data.td.joint_ang(stim_start_idx(i_move),:);

                                    % integrate vel
                                    stim_dec_joint_ang = integrateVel(init_joint_ang,stim_effect_temp.act_vel,input_data.td.bin_size);   
                                    nostim_dec_joint_ang = integrateVel(init_joint_ang,stim_effect_temp.base_vel,input_data.td.bin_size);

                                    % put data through opensim, nonstim first
                                    for i_opensim = 1:2
                                        if(i_opensim==1)
                                            joint_ang_use = nostim_dec_joint_ang;
                                        else
                                            joint_ang_use = stim_dec_joint_ang;
                                        end
                                        t_use = ((1:1:size(nostim_dec_joint_ang,1))-1)*input_data.td.bin_size;
                                        writeMotFile(opensim_data.mot_path, opensim_data.mot_name, t_use,...
                                            joint_ang_use, opensim_data.joint_names, opensim_data.in_deg); % in_deg = 1;

                                         % get the name of the file for this trial
                                        trialsForAn = dir([opensim_data.mot_path filesep, opensim_data.mot_name]);
                                        motIKCoordsFile = trialsForAn(1).name;

                                        % create name of trial from .trc file name
                                        name = regexprep(motIKCoordsFile,'.mot','');

                                        % get .mot data to determine time range
                                        motCoordsData = Storage(fullfile(ik_results_folder, motIKCoordsFile));

                                        % run analyzeTool
                                        initial_time = motCoordsData.getFirstTime();
                                        final_time = motCoordsData.getLastTime();

                                        analyzeTool.setName(name);
                                        analyzeTool.setResultsDir(opensim_data.results_folder);
                                        analyzeTool.setCoordinatesFileName(fullfile(ik_results_folder, motIKCoordsFile));
                                        analyzeTool.setInitialTime(initial_time);
                                        analyzeTool.setFinalTime(final_time);   

                                        outfile = ['Setup_Analyze_.xml'];
                                        analyzeTool.print(fullfile(setupfiles_folder, outfile));

                                        analyzeTool.run();

                                        % get data from opensim outputted file
                                        temp = getDataFromOpensimFile(opensim_data);
                                        if(i_opensim==1)
                                            temp_point_kin_no = temp;
                                        else
                                            temp_point_kin_stim = temp;
                                        end

                                    end
                                else
                                    temp_point_kin_no = [];
                                    temp_point_kin_stim = [];
                                end

                                % store which neurons were activated
                                num_pulses_active(counter,:) = sum(stim_effect_temp.is_act,2)';

                                % store relevant parameters for this run
                                dir_act_func{counter,1} = input_data.direct_acts_test{i_dir_act};
                                dir_act_fact(counter,1) = input_data.dir_act_fact(i_dir_fact);
                                trans_act_func{counter,1} = input_data.trans_acts_test{i_trans_act};
                                amp_list(counter,1) = input_data.amps_test(i_amp);
                                freq_list(counter,1) = input_data.freqs_test(i_freq);
                                move_idx(counter,1) = stim_start_idx(i_move);
                                stim_loc(counter,:) = stim_locs(i_loc,:);

                                point_kin_no{counter} = temp_point_kin_no;
                                point_kin_stim{counter} = temp_point_kin_stim;

                                counter = counter + 1;
                            end % end loc
                        end % end move
                    end % end freq
                end % end amp
            end % end direct activation factor
        end % end trans activation
    end % end direct activation
    
    
    
    % package outputs
    output_data = [];
    output_data.locs = input_data.locs;
    output_data.dir_act_func = dir_act_func;
    output_data.trans_act_func = trans_act_func;
    output_data.dir_act_fact = dir_act_fact;
    output_data.amp_list = amp_list;
    output_data.freq_list = freq_list;
    output_data.move_idx = move_idx;
    output_data.stim_loc = stim_loc;
    output_data.num_pulses_active = num_pulses_active;
    output_data.stim_vel = stim_vel;
    output_data.base_vel = base_vel;
    output_data.point_kin_no_stim = point_kin_no;
    output_data.point_kin_stim = point_kin_stim;
end