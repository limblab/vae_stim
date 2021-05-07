function [similarity_out] = getActivatedPopulationSimilarity(input_data)
% input data contains:
%   stim_locs : location of stimulation site for each sim
%   locs : location of neurons in the map
%   num_pulses_active : the number of pulses each neuron was active for
%       each sim
%   metric_name : name of field in input_data to use as similarity metric
%   "metric" : whatever metric_name is for each neuron
%   is_ang : is metric an angular value

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
    
% get similarity metric for each sim
    
    similarity_out = getSimilarityStimModel(metric, input_data.num_pulses_active, stim_metric, input_data.is_ang);

    

end