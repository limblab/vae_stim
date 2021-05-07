function [output_data] = getStimModelSimulationData(input_data)
% this function runs all simulation experiments for a given trial data.
% trial data will have the VAE rates. This function gets called by
% run_simulations_many_maps
    
    % extract data from input_data
    td = input_data.td;
    locs = input_data.locs;
    map_dim = input_data.map_dim;
    dec_path = input_data.dec_path; % no file sep afterwards
    dec_fname = input_data.dec_fname;


    %% get PDs
    pd_params = [];
    pd_params.out_signals = 'VAE_firing_rates';
    pd_params.in_signals = {'vel'};
    pd_params.num_boots = 0;
    pd_params.verbose = false;
    splitParams.split_idx_name = 'idx_startTime';
    splitParams.linked_fields = {'trialID','result'};
    bin_edges = [0:0.2:2*pi];
    
    td_reward = splitTD(td,splitParams);
    td_reward = td_reward([td_reward.result]=='R');
    pd_table = getTDPDs(td_reward, pd_params);


    %% get PD differences for neighboring neurons and distant neurons -- TO DO
    % test PD neighborhood 
    nbor_input = [];
    nbor_input.nbor_max_r = 2.5; % blocks (or neurons away). Not a um distance
    % currently, r=2.5 makes the histogram match Weber et. al well. That
    % means 2.5 corresponds to ~150 um. 1 block ~= 0.06 mm
    
    nbor_input.nbor_min_r = 0;
    nbor_input.num_sample = 1000;
    % I wrote this function to handle arbitrary metrics instead of just PDs
    % if metric has multiple columns, function defaults to taking mean
    % difference across columns
    nbor_input.metric = pd_table.velPD; % if angle, assumes radians
    nbor_input.metric_is_angle = 1; % if the metric is an angle, flag this
    nbor_input.locs = locs;

    nbor_output = getNeighborMetric(nbor_input);
    
    %% build decoder or load one in (Building can take awhile)
    % if loading a decoder, fill these out. Otherwise ignore
    
    % check to see if decoder exists. If not, make one and save it.
    if(exist([dec_path filesep dec_fname]) <= 0)
        dec_input_data = [];
        dec_input_data.lr = 0.00001;
        dec_input_data.num_iters = 10000;
        dec_input_data.dropout_rate = 0.93;
        
        train_idx = datasample(1:1:size(td.VAE_firing_rates,1),ceil(0.85*size(td.VAE_firing_rates,1)),'Replace',false);
        
        dec_input_data.fr = td.VAE_firing_rates(train_idx,:)';
        dec_input_data.hand_vel = td.vel(train_idx,:);
        
        dec_output_data = buildDecoderDropout(dec_input_data);
        dec = dec_output_data.dec; bias = dec_output_data.bias;
        
        save([dec_path filesep dec_fname],'dec','bias');
    else
        load([dec_path filesep dec_fname]);
    end
    
    td.pred_vel = predictData(td.VAE_firing_rates, dec, bias);
    vaf_pred = compute_vaf(td.vel, td.pred_vel);
    PD_dec_diff = rad2deg(angleDiff(pd_table.velPD, atan2(dec(:,2),dec(:,1)),1));
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% run stimulation experiments %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Experiment 1 : 
    % How does the direction of stim effect compare to the PD of the stimulated
    % location during single electrode stimulation?

    % Independent Variables: Amplitude, Activation Function, Frequency. 

    % Dependent Variables : Direction and magnitude of stim effect,
    % measured from predicted hand velocity from the linear decoder

    % Will do simulations for multiple locations, can then sanity check
    % with a neighborhood metric.
    % Will also do simulations for multiple hand kinematics, which will
    % affect firing rate of neurons, and thus the effect of stim.
    
    amp_input_data = [];
    amp_input_data.amps_test = [15,30,50,100]; % uA
    amp_input_data.acts_test = {'model_based'};
    amp_input_data.freqs_test = [100]; % Hz
    amp_input_data.train_length = [1.0]; % in s 

    amp_input_data.n_locs = 200;
    amp_input_data.n_moves = 10;
    
    amp_input_data.td = td;
    amp_input_data.locs = locs*input_data.block_size; % block size is 0.06 mm currently.
    amp_input_data.dec = dec;
    amp_input_data.bias = bias;
    
    amp_output_data = runSingleElectrodeLocationExperiment(amp_input_data);
    
    
    % look at how activated population compares to stimulation location PD
    % compare across amplitudes and activation functions. 
    % get neuron idx for each stimulation
    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    
    stim_PD = pd_table.velPD(stim_idx);
    PD_diff_stim_loc = stim_PD - pd_table.velPD';
    
    % remove stimulation location
    PD_diff_stim_loc = makeStimLocationNan(PD_diff_stim_loc,stim_idx, size(locs,1));

    % mask PD diff for neurons that were active only.
    PD_diff_act = PD_diff_stim_loc.*(amp_output_data.num_pulses_active > 0);
    amp_output_data.PD_diff_act = PD_diff_act;
    
    
    % compare similarity between activated population against some neighborhood
    % metric. The goal here is to simply show that being in the middle of a
    % cluster is better than being near the edge. Also, some variance among
    % the similarity of the activated population would be nice.
    
    % for each sim, compute a similarity between activated population and
    % stim location. Also compute a similarity between neighbors and stim
    % location
    
    sim_input_data = amp_output_data;
    sim_input_data.metric_name = 'PD';
    sim_input_data.PD = pd_table.velPD; % PD must be in radians!
    sim_input_data.is_ang = 1;
    
    act_pop_similarity = getActivatedPopulationSimilarity(sim_input_data);
   
    sim_input_data.nbor_max_r = 0.3; % in mm
    sim_input_data.nbor_min_r = 0;
    
    neighbor_similarity = getNeighborsSimilarity(sim_input_data);
    
    amp_output_data.act_pop_similarity = act_pop_similarity;
    amp_output_data.neighbor_similarity = neighbor_similarity;
    
    % look at how effect of stimulation compares to PDs of stimulation location
    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    stim_PD = pd_table.velPD(stim_idx);
    
    stim_vel = squeeze(mean(amp_output_data.stim_vel,2));
    stim_ang = atan2(stim_vel(:,2),stim_vel(:,1));
    
    amp_output_data.stim_PD = stim_PD;
    amp_output_data.stim_ang = stim_ang;
    
    
    %% Experiment 2 : Classify movements in one of two directions (can rotate axis)
    % then stimulate and look at bias in classification 
    % this is pretty close to recreating Tucker's experiments
    class_input = [];

    class_input.n_stim_elec = 4;
    class_input.amps_test = [15,20,25,30,50]; % uA
    class_input.act_func = 'model_based';
    class_input.freq = [100]; % Hz
    class_input.n_locs = 1; % number of different stimulation sets per target axis
    class_input.n_runs = 100; % number of target axes per array
    class_input.use_array = 0; % 1 to drop a fake array; 0 to use all blocks
    class_input.elec_spacing = 0.4; % mm, electrode spacing for the array
    class_input.block_size = input_data.block_size; % mm
    class_input.locs = locs*class_input.block_size; 
    class_input.bias = bias;
    
    class_input.dec = dec;
    class_input.bias = bias;
    class_input.td = td;
    class_input.PD = pd_table.velPD;
    class_input.PD_tol = 22.5; % degrees
    
    class_input.move_len = 0.5; % s
    class_input.in_sig = 'pred_vel';
    class_input.min_sig = 10;
    class_input.max_sig = 40;
    class_input.n_train = 250;
    class_input.n_test = 250;
    class_input.classifier_name = 'linear';
        
    
    [class_output_data] = simulatePerceptualEffectStimulation(class_input);
    
    
    % package outputs
    output_data = [];

    output_data.dec = dec; 
    output_data.bias = bias;
    output_data.pd_table = pd_table;
    output_data.nbor_output = nbor_output;
    output_data.dec_vaf_pred = vaf_pred;
    output_data.PD_dec_diff = PD_dec_diff;
    output_data.td = td;
    output_data.amp_output_data = amp_output_data;
    output_data.amp_input_data = amp_input_data;
    output_data.class_output_data = class_output_data;
    output_data.class_input_data = class_input;
end

