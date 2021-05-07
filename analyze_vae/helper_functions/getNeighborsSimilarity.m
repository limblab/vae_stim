function [similarity_out] = getNeighborsSimilarity(input_data)
% input data contains:
%   stim_loc : location of stimulation site for each sim
%   locs : location of neurons in the map
%   metric_name : name of field in input_data to use as similarity metric
%   "metric" : whatever metric_name is for each neuron
%   is_ang : is metric an angular value
%   nbor_max_r : maximum radius of a neighbor
%   nbor_min_r : minimum radius of a neighbor


% get metric for each sim
    metric = input_data.(input_data.metric_name);
    if(size(metric,2) == 1) % transpose metric
        metric = metric';
    end
    metric = repmat(metric, size(input_data.stim_loc,1),1);
    
% get stim location metric for each sim
    stim_idx = getStimLocationIdx(input_data.stim_loc, input_data.locs);
    stim_metric = input_data.(input_data.metric_name)(stim_idx);

% remove stim site from metric
    makeStimLocationNan(metric,stim_idx, size(input_data.locs,1));
    
% get neighbor mask, where a 1 indicates if it is a neighbor and a 0
% indicates if it is not.
    dist_from_stim = ((input_data.stim_loc(:,1) - input_data.locs(:,1)').^2 + (input_data.stim_loc(:,2) - input_data.locs(:,2)').^2);
    
    is_neighbor = dist_from_stim < input_data.nbor_max_r & dist_from_stim >= input_data.nbor_min_r;
    
% get similarity metric for each sim
    similarity_out = getSimilarityStimModel(metric, is_neighbor, stim_metric, input_data.is_ang);






end