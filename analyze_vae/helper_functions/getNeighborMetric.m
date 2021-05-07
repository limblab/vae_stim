function [output_data] = getNeighborMetric(input_data)
% this function looks at whether a metric is similar when neurons are close
% together compared to neurons who are far apart. For example, Weber 2011
% looked at PDs for same electrode and different.

% input_data contains:
% nbor_max_r : neighbors are at most this value away
% nbor_min_r : neighbors are at least this value away
% num_sample : number of locations to sample
% metric : metric to compare across neurons (PD for example)
% metric_is_angle : flag if metric is an angle 
% locs : location of neurons


    output_data = [];
    
    % sample neuron numbers
    n_neurons = size(input_data.locs,1);
    if(input_data.num_sample > n_neurons)
        input_data.num_sample = n_neurons;
        warning('only sampling as many neurons as are in the map');
    end
    
    center_idx = datasample(1:1:n_neurons, input_data.num_sample, 'Replace',false);
    
    % for each sample:
    diff_list = [];
    is_neigh_list = [];
    center_iter = [];
    dist_list = [];
    for i_samp = 1:input_data.num_sample
        % compare neighbors and non-neighbors, store values and which
        % sample idx was used. Also store whether it was a neuron or not
        
        center_dist = getDistance(input_data.locs(center_idx(i_samp),:), input_data.locs);
        is_neigh = center_dist >= input_data.nbor_min_r & center_dist <= input_data.nbor_max_r & center_dist ~= 0;
        
        nbor_idx = find(is_neigh);
        non_nbor_idx = find(~is_neigh);
        
        % set number of entries the same
        if(numel(non_nbor_idx) > numel(nbor_idx))
            non_nbor_idx  = non_nbor_idx(datasample(1:1:numel(non_nbor_idx),numel(nbor_idx),'Replace',false));
        elseif(numel(nbor_idx) > numel(non_nbor_idx))
            nbor_idx = nbor_idx(datasample(1:1:numel(nbor_idx),numel(non_nbor_idx),'Replace',false));
        end
        
        % find difference in metric from center val
        if(input_data.metric_is_angle)
            nbor_diff = mean(angleDiff(input_data.metric(nbor_idx,:),input_data.metric(center_idx(i_samp),:),1),2);
            non_nbor_diff = mean(angleDiff(input_data.metric(non_nbor_idx,:) - input_data.metric(center_idx(i_samp),:),1),2);
        else
            nbor_diff = mean(input_data.metric(nbor_idx,:) - input_data.metric(center_idx(i_samp),:),2);
            non_nbor_diff = mean(input_data.metric(non_nbor_idx,:) - input_data.metric(center_idx(i_samp),:),2);
        end
        
        % find actual distance in case we want this
        nbor_dist = getDistance(input_data.locs(center_idx(i_samp),:), input_data.locs(nbor_idx,:));
        non_nbor_dist = getDistance(input_data.locs(center_idx(i_samp),:), input_data.locs(non_nbor_idx,:));
        
        % store
        diff_list = [diff_list; nbor_diff; non_nbor_diff];
        is_neigh_list = [is_neigh_list; ones(numel(nbor_diff),1); zeros(numel(non_nbor_diff),1)];
        center_iter = [center_iter; i_samp*ones(numel(nbor_diff) + numel(non_nbor_diff),1)];
        dist_list = [dist_list; nbor_dist; non_nbor_dist];
        
    end


    output_data.diff = diff_list;
    output_data.is_neigh = is_neigh_list;
    output_data.center_iter = center_iter;
    output_data.dist_from_center = dist_list;
    output_data.center_locs = input_data.locs(center_idx,:);


end

function [dist] = getDistance(x,y)
    dist = sqrt(sum((x-y).^2,2));
end