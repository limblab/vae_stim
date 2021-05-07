%% load file
    pathname = 'D:\Joseph\stimModel';
    maps_folder_name = 'Han_20160315_maps';
    td_filename = 'Han_20160315_RW_SmoothKin_50ms.mat';
    fr_file = 'rates_Han_20160315_RW_sigma2.0_drop93_lambda20_learning1e-05_n-epochs600_n-neurons1600_rate6.0_2021-04-21-194045.csv';
    
    underscore_idx = strfind(fr_file,'_');
    underscore_idx = underscore_idx(find(underscore_idx,1,'last'));
    
    file_id = fr_file(underscore_idx+1:end-4);
    
    load([pathname filesep 'td' filesep td_filename]);
    firing_rates = readtable([pathname,filesep,maps_folder_name filesep fr_file]);
    firing_rates = firing_rates{:,:};

    rel= version('-release');
    is_old_matlab = str2num(rel(1:4)) < 2018;

    %add to td
    bin_size = 0.05;
    td.VAE_firing_rates = firing_rates(:,:)/bin_size;
    map_dim = sqrt(numel(firing_rates(1,:)) + [0,0]);

    locs = zeros(map_dim(1)*map_dim(2),2);
    [locs(:,1), locs(:,2)] = ind2sub([map_dim(1), map_dim(2)],1:map_dim(1)*map_dim(2)); % get location of each neurons
    
    
% match up data lengths

    field_len = length(td.vel);
    td_fieldnames = fieldnames(td);
    [~,mask] = rmmissing(td.vel);

    for i_field = 1:numel(td_fieldnames)
        if(length(td.(td_fieldnames{i_field})) == field_len)
            td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
        end
    end    

    
%% analyze firing rates
    % plot histogram of firing rates (and compare to real thing at some
    % point). Also plot firing rate during reaches in different directions
    % heatmap showing neuron firing rate against speed and movement
    % direction
    
    % for each neuron, build a glm predicting firing rate from cosine(dir) + speed
    td=getNorm(td,'vel');
    n_neurons = size(td.VAE_firing_rates,2);
    n_params = 6; %constant, hand(x,y) position, hand (x,y) vel, speed
    glm_fits = nan(n_neurons,n_params);
    pr2_fits = nan(n_neurons,2); % train, test
    state = [td.pos, td.vel, td.vel_norm]; % offset term automatically included
    
    for i_unit = 1:size(td.VAE_firing_rates,2)
        fr = td.VAE_firing_rates(:,i_unit);
        train_idx = datasample(1:1:length(state),ceil(0.9*length(state)),'Replace',false);
        test_idx =setdiff(1:1:length(state),train_idx);
        mdl = fitglm(state(train_idx,:),fr(train_idx),'Distribution','Poisson');
        
        glm_fits(i_unit,:) = mdl.Coefficients.Estimate;
        
        % get pR2 for each neuron
        pr2_fits(i_unit,2) = compute_pseudo_R2(fr(test_idx,:), predict(mdl,state(test_idx,:)),mean(fr(test_idx,:)));
        pr2_fits(i_unit,1) = compute_pseudo_R2(fr(train_idx,:), predict(mdl,state(train_idx,:)),mean(fr(train_idx,:)));
    end
    
%% get histogram of pr2 and heatmap of predicted firing rates for an example neuron (speed and direction vs FR)
    figure();
    histogram(pr2_fits(:,2)) % pr2_fits is (train, test)
    xlabel('Pseudo R2');
    ylabel('Proportion of neurons');
    formatForLee(gcf); set(gca,'fontsize',14);
    
    i_unit = 1530;
    speed_grid = 0:1:max(td.speed);
    dir_grid = -180:10:180;
    fr_grid = zeros(numel(speed_grid),numel(dir_grid));
    
    for i_speed = 1:numel(speed_grid)
        for i_dir = 1:numel(dir_grid)
            % constant, hand(x,y) position, hand (x,y) vel, speed
            curr_vel = speed_grid(i_speed)*[cos(deg2rad(dir_grid(i_dir))), sin(deg2rad(dir_grid(i_dir)))];
            curr_state = [median(td.pos(:,:)), curr_vel, speed_grid(i_speed)];
            fr_grid(i_speed,i_dir) = glmval(glm_fits(i_unit,:)',curr_state,'log','Constant','on');
        end
    end
    
    figure();
    imagesc(fr_grid);
    xlabel('Direction (degrees)');
    ylabel('Speed (cm/s)');
    b=colorbar;
    b.Label.String = 'Firing rate (sp/s)';
    formatForLee(gcf);
    set(gca,'fontsize',14);
    
    
%% analyze structure of map
    % get PDs
    pd_params = [];
    pd_params.out_signals = 'VAE_firing_rates';
    pd_params.in_signals = {'vel'};
    pd_params.num_boots = 0;
    splitParams.split_idx_name = 'idx_startTime';
    splitParams.linked_fields = {'trialID','result'};
    bin_edges = [0:0.2:2*pi];
    
    td_reward = splitTD(td,splitParams);
    td_reward = td_reward([td_reward.result]=='R');
    pd_table = getTDPDs(td_reward, pd_params);

    
% plot polar histogram of PDs and map of PDs
    f=figure();
    polarhistogram(pd_table.velPD,bin_edges,'DisplayStyle','bar') % bar or stair. Stair is useful if plotting multiple
    title('PDs (degrees)');
    f.Name = [file_id,'_PD_histogram'];
    
    f=figure();
    pd_map = rad2deg(reshape(pd_table.velPD,map_dim));
    pd_map(pd_map<0) = pd_map(pd_map<0)+360; % convert to same limits as polarhistogram
    imagesc(pd_map);
    colormap(colorcet('C3'));
    b=colorbar;
    b.Label.String = 'PD (degrees)';
    b.Label.FontSize = 14;
    f.Name = [file_id,'_PD_cortical_map'];
    
%% get and plot PD differences for neighboring neurons and distant neurons -- TO DO
    shuffle_map_slightly = 0;
    % test PD neighborhood 
    nbor_input = [];
    nbor_input.nbor_max_r = 2.5; %2.25-2.8 gives same values; % blocks (or neurons away). Not a um distance
    % currently, r=2.5 makes the histogram match Weber et. al well. That
    % means 2.5 corresponds to ~150 um. 1 block ~= 0.06 mm
    
    nbor_input.nbor_min_r = 0;
    nbor_input.num_sample = 1600;
    % I wrote this function to handle arbitrary metrics instead of just PDs
    % if metric has multiple columns, function defaults to taking mean
    % difference across columns
    nbor_input.metric = pd_table.velPD; % if angle, assumes radians
    nbor_input.metric_is_angle = 1; % if the metric is an angle, flag this
    nbor_input.locs = locs;

    if(shuffle_map_slightly)
        n_swap = 200;
        for i = 1:n_swap
            swap_idx = datasample(1:1:numel(pd_table.velPD),2,'Replace',false);
            
            nbor_input.metric(swap_idx) = nbor_input.metric(flip(swap_idx),:);
        end
        
        figure();
        pd_map = rad2deg(reshape(nbor_input.metric,map_dim));
        pd_map(pd_map<0) = pd_map(pd_map<0)+360; % convert to same limits as polarhistogram
        imagesc(pd_map);
        colormap(colorcet('C3'));
        b=colorbar;
        b.Label.String = 'PD (degrees)';
        b.Label.FontSize = 14;
    end

    nbor_output = getNeighborMetric(nbor_input);

    bin_edges = [0:10:180];
    
    f=figure(); hold on;
    histogram(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==1))),bin_edges,...
        'EdgeColor',getColorFromList(1,1),'DisplayStyle','stairs','Normalization','probability','Linewidth',2);
    histogram(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==0))),bin_edges,...
        'EdgeColor',getColorFromList(1,0),'DisplayStyle','stairs','Normalization','probability','Linewidth',2);
    formatForLee(gcf);
    xlabel('PD Diff (degrees)');
    ylabel('Proportion of data');
    l=legend('Neighbor','Non-neighbor'); set(l,'box','off');
    set(gca,'fontsize',14)
    xlim([0,180]);
    
    f.Name = [file_id,'_PD_neighbor_dist'];
    
%% get neighbor score for all neurons
    sim_input_data = [];
    sim_input_data.locs = locs*0.06; % 60 mm block size
    sim_input_data.stim_loc = sim_input_data.locs;
    sim_input_data.metric_name = 'PD';
    sim_input_data.PD = pd_table.velPD; % PD must be in radians!
    sim_input_data.is_ang = 1;
       
    sim_input_data.nbor_max_r = 0.08; % in mm
    sim_input_data.nbor_min_r = 0;
    
    neigh_sim_all = getNeighborsSimilarity(sim_input_data);

    f=figure();
    sim_map = reshape(neigh_sim_all,map_dim);
    imagesc(sim_map);
    colormap(colorcet('L1'));
    b=colorbar;
    b.Label.String = 'Cosine Similarity';
    b.Label.FontSize = 14;
    f.Name = [file_id,'_sim_cortical_map'];
    
    
%% build decoder or load one in (Building can take awhile)
    build_decoder = 0;
    % if loading a decoder, fill these out. Otherwise ignore
    dec_path = 'D:\Joseph\stimModel\decoders'; % no file sep afterwards
    dec_fname = [fr_file(1:end-4), '_dec.mat'];
    
    if(build_decoder==1)
        dec_input_data = [];
        dec_input_data.lr = 0.00001;
        dec_input_data.num_iters = 10000;
        dec_input_data.dropout_rate = 0.93;
        
        train_idx = datasample(1:1:size(td.VAE_firing_rates,1),ceil(0.5*size(td.VAE_firing_rates,1)),'Replace',false);
        
        dec_input_data.fr = td.VAE_firing_rates(train_idx,:)';
        dec_input_data.hand_vel = td.vel(train_idx,:);
        
        dec_output_data = buildDecoderDropout(dec_input_data);
        dec = dec_output_data.dec; bias = dec_output_data.bias;
    else
        load([dec_path filesep dec_fname]);
    end
    
        td.pred_vel = predictData(td.VAE_firing_rates, dec, bias);
    vaf_pred = compute_vaf(td.vel, td.pred_vel)
%% use decoder to get predicted hand velocities
    f=figure();
    ax1=subplot(1,2,1); hold on;
    plot(td.vel(:,1), td.pred_vel(:,1),'.');
    plot([-30,30],[-30,30],'k--','linewidth',2);
    xlabel('Hand vel (cm/s)');
    ylabel('Pred hand vel (cm/s)');
    formatForLee(gcf); set(gca,'fontsize',14);
    
    ax2=subplot(1,2,2); hold on;
    plot(td.vel(:,2), td.pred_vel(:,2),'.');
    plot([-30,30],[-30,30],'k--','linewidth',2);
    formatForLee(gcf); set(gca,'fontsize',14);
    xlabel('Hand vel (cm/s)');
    ylabel('Pred hand vel (cm/s)');
    linkaxes([ax1,ax2],'xy');
    
    f.Name = [file_id,'_decoder_predictions'];
%% compare decoder PD to PDs found using movement and an encoder
    if(exist('pd_table')==0)
        error('nothing was done. Need to compute PDs from above');
    end

    PD_dec_diff = rad2deg(angleDiff(pd_table.velPD, atan2(dec(:,2),dec(:,1)),1));
    bin_edges = [0:10:180];

    f=figure();
    histogram(abs(PD_dec_diff),bin_edges,'Normalization','probability')
    xlabel('Absolute difference between PDs (degrees)');
    ylabel('Proportion of neurons');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    f.Name = [file_id,'_PD_encoder_decoder_diff'];
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% run stimulation experiments %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Experiment 0: 
% stimulate and look at effect on hand velocity/position. Make example
% figures using this code
    % stimulate at a chosen electrode and visualize spread
    locs_cort = locs*0.06;
    trial_idx = 120;
    move_len_idx = ceil(1.0/td.bin_size);
    
    [~,idx] = datasample(1:1:size(locs_cort,1),1);
    stim_loc = locs_cort(idx,:);
    
    stim_in_data = [];
    stim_in_data.FR = td.VAE_firing_rates(trial_idx:trial_idx+move_len_idx-1,:);
    stim_in_data.dec = dec; stim_in_data.bias = bias;
    stim_in_data.act_func = 'model_based_floor';
    stim_in_data.locs = locs_cort;
    stim_in_data.stim_loc = stim_loc;
    stim_in_data.amp = 20;
    stim_in_data.pulse_bin_idx = reshape(repmat([2:1:move_len_idx],5,1),[],1); % each pulse within a bin is 20 Hz stim.
    stim_in_data.n_pulses = numel(stim_in_data.pulse_bin_idx);
    stim_in_data.bin_size = td.bin_size;

    stim_eff = getStimEffect(stim_in_data);
    num_act = sum(stim_eff.is_act,2);
    
    f=figure();
    pd_map = rad2deg(reshape(pd_table.velPD,map_dim));
    pd_map(pd_map<0) = pd_map(pd_map<0)+360; % convert to same limits as polarhistogram
    imagesc(pd_map);
    colormap(colorcet('C3'));
    alpha(0.6);
    b=colorbar;
    b.Label.String = 'PD (degrees)';
    b.Label.FontSize = 14;
    
    % plot activated neurons, shade by how often they were activated
    max_act = max(num_act);
    num_act_map = reshape(num_act,map_dim);
    color_list = colorcet('L5');
    color_list = color_list(50:end,:);
    for x_loc = 1:size(num_act_map,1)
        for y_loc = 1:size(num_act_map,2)
            if(num_act_map(x_loc,y_loc) > 0)
                color_idx = ceil(size(color_list,1)*num_act_map(x_loc,y_loc)/max_act);
                color_use = color_list(color_idx,:);
                rectangle('Position',[y_loc-0.5,x_loc-0.5,1,1],'EdgeColor',color_use,'FaceColor','none','linewidth',3);
            end
        end
    end
    
    % plot stim location
    loc_idx = locs(idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    
%     f.Name = [file_id,'_PD_cortical_map'];
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
    amp_input_data.train_length = [0.5]; % in s 

    amp_input_data.n_locs = 500;
    amp_input_data.n_moves = 1;
    
    amp_input_data.td = td;
    amp_input_data.locs = locs*0.06; % block size is 0.06 mm currently.
    amp_input_data.dec = dec;
    amp_input_data.bias = bias;
    
    amp_output_data = runSingleElectrodeLocationExperiment(amp_input_data);
    
    bin_edges = 0:0.1:4;
    % get neuron idx for each stimulation
    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    stim_loc = amp_output_data.locs(stim_idx,:);
    dist_to_stim = sqrt((amp_output_data.locs(:,1)-stim_loc(:,1)').^2 + (amp_output_data.locs(:,2)-stim_loc(:,2)').^2)';
    
    % get was activated mask
    
    unique_amps = unique(amp_output_data.amp_list);
    color_list = inferno(numel(unique_amps)+1);
    figure('Position',[681 559 950 420]);
    for i_amp = 1:numel(unique_amps)
        amp_mask = amp_output_data.amp_list == unique_amps(i_amp);
        dist_amp = dist_to_stim(amp_mask,:);
        was_act = amp_output_data.num_pulses_active(amp_mask,:) > 0;
        num_act = amp_output_data.num_pulses_active(amp_mask,:);
        
        dist_amp= reshape(dist_amp,[],1);
        was_act = reshape(was_act,[],1);
        num_act = reshape(num_act,[],1);
        
        dist = dist_amp(was_act);
        dist = repelem(dist,num_act(was_act));
        spike_counts = histcounts(dist,bin_edges);
        spike_counts = spike_counts;
        num_neurons = histcounts(dist_amp,bin_edges);
        
        subplot(1,2,1); hold on;
        x_data = 1000*(bin_edges(1:end-1)+mode(diff(bin_edges))/2);
        plot(x_data,spike_counts./num_neurons/(amp_input_data.train_length*amp_input_data.freqs_test),...
            'o','markersize',8,'linewidth',1.5,'color',color_list(i_amp,:));
        % plot activation function for this amplitude
        [a_val,b_val] = getModelBasedActivationParameters(unique_amps(i_amp));
        plot(x_data, 1-1./(1+exp(-a_val*(x_data - b_val))),'--','linewidth',1.5,'color',color_list(i_amp,:));
        
        subplot(1,2,2); hold on;
        plot(1000*(bin_edges(1:end-1)+mode(diff(bin_edges))/2),spike_counts/size(amp_output_data.amp_list,1),'o','markersize',8,'linewidth',1.5,'color',color_list(i_amp,:));
    end
    subplot(1,2,1);
    xlabel('Distance from stim (\mum)');
    ylabel('Proportion of activated neurons');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlim([0,2000]);
    
    subplot(1,2,2);
    xlabel('Distance from stim (\mum)');
    ylabel('Number of spikes per train');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlim([0,2000]);
%% look at how activated population compares to stimulation location PD
    % compare across amplitudes and activation functions. 
        
        f=plotActivatedPopulationVsStimPD(amp_input_data,amp_output_data,pd_table);
        f.Name = [file_id,'_activatedPop_vs_stimPD'];    
        
%%
    % compare similarity between activated population against some neighborhood
    % metric. The goal here is to simply show that being in the middle of a
    % cluster is better than being near the edge. Also, some variance among
    % the similarity of the activated population would be nice.
    
    % for each sim, compute a similarity between activated population and
    % stim location. Also compute a similarity between neighbors and stim
    % location
    sim_input_data = [];
    sim_input_data = amp_output_data;
    sim_input_data.metric_name = 'PD';
    sim_input_data.PD = pd_table.velPD; % PD must be in radians!
    sim_input_data.is_ang = 1;
    
    amp_output_data.act_pop_similarity = getActivatedPopulationSimilarity(sim_input_data);
   
    sim_input_data.nbor_max_r = 0.08; % in mm
    sim_input_data.nbor_min_r = 0;
    
    amp_output_data.neighbor_similarity = getNeighborsSimilarity(sim_input_data);
    
    plotNeighborSimilarityVsActivatedSimilarity(amp_input_data, amp_output_data, pd_table.velPD);
%     plotStimEffectVsNeighborSimilarity(amp_input_data,amp_output_data,pd_table);

%% look at how effect of stimulation compares to PDs of stimulation location

    f=plotStimEffectVsStimPD(amp_input_data, amp_output_data, pd_table);
    f.Name = [file_id,'_activatedPop_vs_stimEffect_75thPrctileSim'];
   
    
%% Experiment 2 : Classify movements in one of two directions (can rotate axis)
% then stimulate and look at bias in classification 
% this is pretty close to recreating Tucker's experiments
    class_input = [];

    class_input.n_stim_elec = 4;
    class_input.amps_test = [10,15,20,50]; % uA
    class_input.act_func = 'model_based_act';
    class_input.freq = [100]; % Hz
    class_input.n_locs = 1; % number of different stimulation sets per target axis
    class_input.n_runs = 5; % number of target axes per array
    class_input.use_array = 0; % 1 to drop a fake array; 0 to use all blocks
    class_input.elec_spacing = 0.4; % mm, electrode spacing for the array
    class_input.block_size = 0.06; % mm
    class_input.locs = locs*class_input.block_size; 
    class_input.bias = bias;
    
    class_input.dec = dec;
    class_input.bias = bias;
    class_input.td = td;
    class_input.PD = pd_table.velPD; % in radians
    class_input.PD_tol = 22.5; % degrees
    
    class_input.move_len = 0.5; % s
    class_input.in_sig = 'pred_vel';
    class_input.min_sig = 10;
    class_input.max_sig = 40;
    class_input.n_train = 250;
    class_input.n_test = 250;
    class_input.classifier_name = 'linear';
        
    [class_output_data] = simulatePerceptualEffectStimulation(class_input);
    
    PSD_data = computePsychometricCurveShift(class_output_data); 
    
    disp(sum(PSD_data>=0)/class_input.n_runs);
%% make psychophysical curves for each run
    class_plot.bin_edges = 0:10:180; % in degrees
% %     
    figs = makePsychCurves(class_output_data,class_plot);
    
    
%%  look at neighbor similarity around stimulation electrodes
    sim_input_data = [];
    sim_input_data.locs = class_input.locs;
    sim_input_data.metric_name = 'PD';
    sim_input_data.PD = pd_table.velPD; % PD must be in radians!
    sim_input_data.is_ang = 1;
       
    sim_input_data.nbor_max_r = 0.2; % in mm
    sim_input_data.nbor_min_r = 0;
    
    neigh_sim = getClassNeighborsSimilarity(sim_input_data,class_output_data);
    neigh_sim = mean(neigh_sim,2);

    mask = neigh_sim > prctile(neigh_sim, 75);
    prop_good_neigh = sum(PSD_data(mask,:)>=0)/sum(mask)
    
    mask = neigh_sim < prctile(neigh_sim, 25);
    prop_bad_neigh = sum(PSD_data(mask,:)>=0)/sum(mask)

    tgt_axes = [class_output_data.tgt_axis];

    pred = PSD_data>=0;

    bin_edges = 0:20:360;
    
    [~,~,idx] = histcounts(tgt_axes,bin_edges);
    prop_pred = zeros(numel(bin_edges)-1,size(pred,2));
    sim = zeros(numel(bin_edges)-1,1);
    for i = 1:size(prop_pred,1)
        prop_pred(i,:) = sum(pred(idx==i,:))/sum(idx==i);
        sim(i) = mean(neigh_sim(idx==i));
    end
    figure();
    plot(bin_edges(1:end-1),prop_pred); hold on
    plot(bin_edges(1:end-1),sim);

    
        
    
    
    
    
    
    
    
    