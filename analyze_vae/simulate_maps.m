%% load file
    pathname = 'D:\Lab\Data\StimModel';
    td_filename = 'Han_201603015_RW_SmoothKin_50ms.mat';
    fr_file = 'vae_rates_Han_20160325_RW_dropout91_lambda20_learning1e-05_n-epochs600_n-neurons1600_rate6.0_2021-03-13-184825.csv';

    load([pathname filesep td_filename]);
    firing_rates = readtable([pathname,filesep, fr_file]);
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
    
    
%% get and plot PD distribution. Also plot neighborhood distribution
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

    
%% plot polar histogram of PDs and map of PDs
    figure();
    polarhistogram(pd_table.velPD,bin_edges,'DisplayStyle','bar') % bar or stair. Stair is useful if plotting multiple
    title('PDs (degrees)');

    figure();
    pd_map = rad2deg(reshape(pd_table.velPD,map_dim));
    pd_map(pd_map<0) = pd_map(pd_map<0)+360; % convert to same limits as polarhistogram
    imagesc(pd_map);
    colormap(colorcet('C3'));
    b=colorbar;
    b.Label.String = 'PD (degrees)';
    b.Label.FontSize = 14;
    
%% make map smaller
% replace locs and pd_table.velPD
    locs_all = locs;
    pd_table_all = pd_table;
    
    x_max = 20; y_max = 20;
    
    keep_mask = locs(:,1) <= x_max & locs(:,2) <= y_max;
    
    locs = locs_all(keep_mask,:);
    pd_table = pd_table_all(keep_mask,:);
    map_dim = [x_max, y_max];

    
%% get and plot PD differences for neighboring neurons and distant neurons -- TO DO
    
    % test PD neighborhood 
    nbor_input = [];
    nbor_input.nbor_max_r = 2; % blocks (or neurons away). Not a um distance
    nbor_input.nbor_min_r = 0;
    nbor_input.num_sample = 50;
    % I wrote this function to handle arbitrary metrics instead of just PDs
    % if metric has multiple columns, function defaults to taking mean
    % difference across columns
    nbor_input.metric = pd_table.velPD; % if angle, assumes radians
    nbor_input.metric_is_angle = 1; % if the metric is an angle, flag this
    nbor_input.locs = locs;

    nbor_output = getNeighborMetric(nbor_input);

    bin_edges = [0:10:180];
    nbor_counts = histcounts(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==1))),bin_edges,'Normalization','probability');
    non_nbor_counts = histcounts(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==0))),bin_edges,'Normalization','probability');
    
    figure(); hold on;
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
    
%% Genetic algorithm to simulate maps with neurons near each having similar PDs
%% turns out, this doesn't work very well....
% get neighborhood is slow currently. 

    gen_input = [];
    gen_input.n_iters = 1000;
    gen_input.pop_size = 400;
    gen_input.max_fit = 0.1;
    
    gen_input.prop_top = 0.02;
    gen_input.prop_new = 0.1;
    gen_input.locs = locs;
    gen_input.PDs = pd_table.velPD;
    
    gen_input.bin_edges = bin_edges; % matches neighborhood code
    gen_input.nbor_prop = nbor_counts;
    gen_input.non_nbor_prop = non_nbor_counts;
    
    % neighbor
    nbor_input.num_sample = 70;
    gen_output = generateMaps(gen_input, nbor_input);
    
    
    figure();
    plot(gen_output.fit_list);
    
    
% plot maps
    [~, sort_idx] = sort(gen_output.curr_fit,'ascend');
    gen_output.curr_fit = gen_output.curr_fit(sort_idx);
    gen_output.curr_pop = gen_output.curr_pop(sort_idx,:);
    
    for pop_idx = 1:10
        map_data = zeros(sqrt(size(gen_output.curr_pop,2))+[0,0]);
        for i = 1:size(gen_output.pop_keep,2)
            map_data(i) = pd_table.velPD(gen_output.curr_pop(pop_idx,i));
        end
        nbor_input.locs = gen_input.locs(gen_output.curr_pop(pop_idx,:),:);
        nbor_input.num_sample = 350;
        nbor_output = getNeighborMetric(nbor_input);

        figure();
        subplot(1,2,1);
        imagesc(map_data)
        
        subplot(1,2,2); hold on;
        histogram(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==1))),bin_edges,...
            'EdgeColor',getColorFromList(1,1),'DisplayStyle','stairs','Normalization','probability','Linewidth',2);
        histogram(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==0))),bin_edges,...
            'EdgeColor',getColorFromList(1,0),'DisplayStyle','stairs','Normalization','probability','Linewidth',2);
    end
    
    
%% 
    
    
    
    


    
