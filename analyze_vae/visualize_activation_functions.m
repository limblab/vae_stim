%% this script is intended to visualize the different activation functions (direct and transsynaptic)
% it assumes a map has been loaded in

%% compute useful variables
% PDs
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
% correlation

    corr_table = corr(td.VAE_firing_rates);
% map dimensions
    map_dim = sqrt(size(corr_table));

%% compare direct activations between two methods for a random location
    stim_in_data = [];
    stim_in_data.block_size = 0.06; % mm
    locs_cort = locs*stim_in_data.block_size;
    
    trial_idx = 120;
    move_len_idx = ceil(10.0/td.bin_size);
    
    [~,stim_elec_idx] = datasample(1:1:size(locs_cort,1),1);
    stim_loc = locs_cort(stim_elec_idx,:);
    
    stim_in_data.FR = td.VAE_firing_rates(trial_idx:trial_idx+move_len_idx-1,:);
    stim_in_data.dec = dec; stim_in_data.bias = bias;
    stim_in_data.trans_act_func = '';
    
    stim_in_data.locs = locs_cort;
    stim_in_data.stim_loc = stim_loc;
    stim_in_data.amp = 20;
    stim_in_data.pulse_bin_idx = reshape(repmat([2:1:move_len_idx],5,1),[],1); % each pulse within a bin is 20 Hz stim.
    stim_in_data.n_pulses = numel(stim_in_data.pulse_bin_idx);
    stim_in_data.bin_size = td.bin_size;
    stim_in_data.corr_table = corr_table;
    stim_in_data.PD = pd_table.velPD;
    
    stim_eff = {};
    stim_in_data.direct_act_func = 'model_based_circle_corr';
    stim_in_data.dir_act_fact = [0];
    stim_eff{1} = getStimEffect(stim_in_data);
    
    
    stim_in_data.direct_act_func = 'model_based_circle_corr';
    stim_in_data.dir_act_fact = [1.0];
    stim_eff{2} = getStimEffect(stim_in_data);
    
    
%
% plot prob act vs distance for each one

    figure(); hold on
    dists = sqrt(sum((stim_in_data.locs - stim_in_data.stim_loc).^2,2));
    
    marker_list = {'.','*','o'}; markersize_list = [18,8,4];
    for i=1:numel(stim_eff)
        plot(dists,stim_eff{i}.prob_act,marker_list{i},'color',getColorFromList(1,i-1),'markersize',markersize_list(i))
    end
    ylabel('Probability');
    xlabel('Distance from Stim Elec (mm)');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlim([0,1]);
    
    
% plot activations side-by-side (image)    
  f=figure();
  ax1 = subplot(1,3,1);
  plotPDMap(ax1,pd_table.velPD,map_dim,locs,stim_elec_idx);
    
% plot activated neurons
    ax2=subplot(1,3,2);
    FR_use = mean(stim_eff{1}.FR_stim);
    FR_use_map = reshape(FR_use,[sqrt(numel(FR_use)),sqrt(numel(FR_use))]);
    imagesc(FR_use_map);
    colormap(ax2,flip(colorcet('L1'),1));
    curr_clim = caxis;
    caxis([0,max(curr_clim)]);
    
    loc_idx = locs(stim_elec_idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    b=colorbar();
    b.Label.String = 'Firing rate';
    b.Label.FontSize = 14;
    
    ax3=subplot(1,3,3);
    FR_use = mean(stim_eff{2}.FR_stim);
    FR_use_map = reshape(FR_use,[sqrt(numel(FR_use)),sqrt(numel(FR_use))]);
    imagesc(FR_use_map);
    colormap(ax3,flip(colorcet('L1'),1));
    curr_clim = caxis;
    caxis([0,max(curr_clim)]);
    
    loc_idx = locs(stim_elec_idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    b=colorbar();
    b.Label.String = 'Firing rate';
    b.Label.FontSize = 14;
        
	linkaxes([ax1,ax2,ax3],'xy');
%% plot difference between activations (image) next to PD map
    
    f=figure();
    ax1=subplot(1,2,1);    
    plotPDMap(ax1,pd_table.velPD,map_dim,locs,stim_elec_idx);
    
    % plot differences in firing rates between two maps neurons, shade by how often they were activated
    ax2=subplot(1,2,2);
    FR_use = mean(stim_eff{2}.FR_act - stim_eff{1}.FR_act);

    FR_use_map = reshape(FR_use,[sqrt(numel(FR_use)),sqrt(numel(FR_use))]);
    imagesc(FR_use_map);
    colormap(ax2,flip(colorcet('D3'),1));
    curr_clim = caxis;
    caxis(max(abs(curr_clim))*[-1,1]);
    loc_idx = locs(stim_elec_idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    b=colorbar();
    b.Label.String = 'Firing rate difference';
    b.Label.FontSize = 14;
    
	linkaxes([ax1,ax2],'xy');
    

%% compare transsynaptic activation to direct activation for a provided method

    stim_in_data = [];
    stim_in_data.block_size = 0.06; % mm
    locs_cort = locs*stim_in_data.block_size;
    
    trial_idx = 120;
    move_len_idx = ceil(10.0/td.bin_size);
    
    [~,stim_elec_idx] = datasample(1:1:size(locs_cort,1),1);
    stim_loc = locs_cort(stim_elec_idx,:);
%     stim_elec_idx = 2191;
%     stim_loc = locs_cort(stim_elec_idx,:);
    
    stim_in_data.FR = td.VAE_firing_rates(trial_idx:trial_idx+move_len_idx-1,:);
    stim_in_data.dec = dec; stim_in_data.bias = bias;
    
    
    stim_in_data.locs = locs_cort;
    stim_in_data.stim_loc = stim_loc;
    stim_in_data.amp = 1;
    stim_in_data.pulse_bin_idx = reshape(repmat([2:1:move_len_idx],5,1),[],1); % each pulse within a bin is 20 Hz stim.
    stim_in_data.n_pulses = numel(stim_in_data.pulse_bin_idx);
    stim_in_data.bin_size = td.bin_size;
    stim_in_data.corr_table = corr_table;
    stim_in_data.PD = pd_table.velPD;
    
    stim_eff = {};
    stim_in_data.direct_act_func = 'circular';
    stim_in_data.trans_act_func = 'mexican_hat';
    stim_eff{1} = getStimEffect(stim_in_data);
    
    stim_in_data.trans_act_func = 'corr_project';
    stim_eff{2} = getStimEffect(stim_in_data);


% visualize FR_syn
    % plot PD map
    f=figure();
    ax1 = subplot(1,3,1);
    plotPDMap(ax1,pd_table.velPD,map_dim,locs,stim_elec_idx);
    
% plot FR_syn
    ax2=subplot(1,3,2);
    FR_use = mean(stim_eff{1}.FR_syn);
    FR_use_map = reshape(FR_use,[sqrt(numel(FR_use)),sqrt(numel(FR_use))]);
    imagesc(FR_use_map);
    colormap(ax2,flip(colorcet('D3'),1));
    curr_clim = caxis;
    caxis(max(abs(curr_clim))*[-1,1]);
    
    loc_idx = locs(stim_elec_idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    b=colorbar();
    b.Label.String = 'Firing rate';
    b.Label.FontSize = 14;
     
    ax3=subplot(1,3,3);
    FR_use = mean(stim_eff{2}.FR_syn);
    FR_use_map = reshape(FR_use,[sqrt(numel(FR_use)),sqrt(numel(FR_use))]);
    imagesc(FR_use_map);
    colormap(ax3,flip(colorcet('D3'),1));
    curr_clim = caxis;
    caxis(max(abs(curr_clim))*[-1,1]);
    
    loc_idx = locs(stim_elec_idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',2);
    
    b=colorbar();
    b.Label.String = 'Firing rate';
    b.Label.FontSize = 14;
    
	linkaxes([ax1,ax2,ax3],'xy');
    
%% plot change in FR against distance from stim elec (absolute value and actual value
    figure();
    dist_from_stim = sqrt(sum((locs_cort-locs_cort(stim_elec_idx,:)).^2,2));
    
    bin_edges = [0:0.1:4];

    [n_units, ~, bin_idx] = histcounts(dist_from_stim, bin_edges);
    
    delta_FR = zeros(numel(bin_edges)-1, 2); % absolute value, total
    
    mean_FR_syn = mean(stim_eff{2}.FR_syn,1);
    
    for i_bin = 1:numel(bin_edges)-1
        delta_FR(i_bin,1) = sum(abs(mean_FR_syn(bin_idx==i_bin)));
        delta_FR(i_bin,2) = sum(mean_FR_syn(bin_idx==i_bin));
    end
    
    plot(bin_edges(1:end-1)+mode(diff(bin_edges))/2, delta_FR./n_units')
    
    
%% look at change in FR due to snyapses against tuning of stimulated electrode. Only include neurons near stim elec
    dist_thresh = 10000.0;
    bin_edges = [0:0.1:pi];
    
    dist_from_stim = sqrt(sum((locs_cort-locs_cort(stim_elec_idx,:)).^2,2));
    dist_mask = dist_from_stim < dist_thresh;

    PD_stim = pd_table.velPD(stim_elec_idx);
    PD_diff = angleDiff(pd_table.velPD, PD_stim,1,0); % use radians, don't preserve sign
    
    [n_units,~,bin_idx] = histcounts(PD_diff,bin_edges);
    
    mean_FR_syn = mean(stim_eff{2}.FR_syn,1);
    
    delta_FR = zeros(numel(bin_edges)-1,2); % absolute, total
    for i_bin = 1:numel(bin_edges)-1
        delta_FR(i_bin,1) = sum(abs(mean_FR_syn(bin_idx==i_bin & dist_mask == 1)));
        delta_FR(i_bin,2) = sum(mean_FR_syn(bin_idx==i_bin & dist_mask == 1));
    end
    
    figure();
    plot(bin_edges(1:end-1)+mode(diff(bin_edges))/2,delta_FR./n_units')
    
    
%% get metrics across many iterations of corr_project (transsynaptic activation)
    n_runs = 150;
    stim_in_data = [];
    stim_in_data.block_size = 0.06; % mm
    locs_cort = locs*stim_in_data.block_size;

    trial_idx = 120;
    move_len_idx = ceil(10.0/td.bin_size);
    bin_edges_dist = [0:0.1:4];
    bin_edges_tun = [0:0.1:pi];
    delta_FR_dist = zeros(numel(bin_edges_dist)-1, 2);
    delta_FR_tun = zeros(numel(bin_edges_tun)-1,2);
    
    for i = 1:n_runs
        [~,stim_elec_idx] = datasample(1:1:size(locs_cort,1),1);
        stim_loc = locs_cort(stim_elec_idx,:);

        stim_in_data.FR = td.VAE_firing_rates(trial_idx:trial_idx+move_len_idx-1,:);
        stim_in_data.dec = dec; stim_in_data.bias = bias;
        stim_in_data.direct_act_func = 'model_based_circle_corr';
        stim_in_data.dir_act_fact = 0.5;
        
        stim_in_data.locs = locs_cort;
        stim_in_data.stim_loc = stim_loc;
        stim_in_data.amp = 100;
        stim_in_data.pulse_bin_idx = reshape(repmat([2:1:move_len_idx],5,1),[],1); % each pulse within a bin is 20 Hz stim.
        stim_in_data.n_pulses = numel(stim_in_data.pulse_bin_idx);
        stim_in_data.bin_size = td.bin_size;
        stim_in_data.corr_table = corr_table;
        stim_in_data.PD = pd_table.velPD;

        stim_in_data.trans_act_func = 'corr_project';
        stim_eff = getStimEffect(stim_in_data);

        % compute metrics and append to total list
        
        % change in FR vs distance
        dist_from_stim = sqrt(sum((locs_cort-locs_cort(stim_elec_idx,:)).^2,2));
       
        [n_units, ~, bin_idx] = histcounts(dist_from_stim, bin_edges_dist);
        delta_FR = zeros(numel(bin_edges_dist)-1, 2); % absolute value, total
        mean_FR_syn = mean(stim_eff.FR_syn,1);
        for i_bin = 1:numel(bin_edges_dist)-1
            delta_FR(i_bin,1) = sum(abs(mean_FR_syn(bin_idx==i_bin)))/n_units(i_bin);
            delta_FR(i_bin,2) = sum(mean_FR_syn(bin_idx==i_bin))/n_units(i_bin);
        end
        delta_FR_dist = delta_FR_dist + delta_FR;
        
        % change in FR vs. tuning
        PD_stim = pd_table.velPD(stim_elec_idx);
        PD_diff = angleDiff(pd_table.velPD, PD_stim,1,0); % use radians, don't preserve sign

        [n_units,~,bin_idx] = histcounts(PD_diff,bin_edges_tun);
        mean_FR_syn = mean(stim_eff.FR_syn,1);
        delta_FR = zeros(numel(bin_edges_tun)-1,2); % absolute, total
        for i_bin = 1:numel(bin_edges_tun)-1
            delta_FR(i_bin,1) = sum(abs(mean_FR_syn(bin_idx==i_bin & dist_mask == 1)))/n_units(i_bin);
            delta_FR(i_bin,2) = sum(mean_FR_syn(bin_idx==i_bin & dist_mask == 1))/n_units(i_bin);
        end
        delta_FR_tun = delta_FR_tun + delta_FR;
    end
    
    delta_FR_tun = delta_FR_tun/n_runs;
    delta_FR_dist = delta_FR_dist/n_runs;
    %%
    
    figure();
    subplot(1,2,1)
    plot(bin_edges_dist(1:end-1)+mode(diff(bin_edges_dist))/2, delta_FR_dist(:,2))
    xlabel('Dist from stim elec (mm)');
    ylabel('\delta FR syn');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlim([0,2]);
    
    subplot(1,2,2)
    plot(rad2deg(bin_edges_tun(1:end-1)+mode(diff(bin_edges_tun))/2), delta_FR_tun(:,2))
    xlabel('PD diff from stim PD (deg)');
    ylabel('\Delta FR syn');
    formatForLee(gcf);
    set(gca,'fontsize',14);