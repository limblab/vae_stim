function [ stim_idx ] = getStimLocationIdx( stim_locs, neuron_locs )

    [stim_row,stim_col] = find(stim_locs(:,1) == neuron_locs(:,1)' & stim_locs(:,2) == neuron_locs(:,2)');
    stim_idx = nan(numel(stim_row),1);
    stim_idx(stim_row) = stim_col;

end

