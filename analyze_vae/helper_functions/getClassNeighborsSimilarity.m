function [sim_scores] = getClassNeighborsSimilarity(sim_input_data,class_data)

    sim_scores = nan(numel(class_data),numel(class_data(1).stim_idx));
    
    for i = 1:numel(class_data)
        stim_locs = sim_input_data.locs(class_data(i).stim_idx,:);
        for i_loc = 1:size(stim_locs,1)
            sim_input_data.stim_loc = stim_locs(i_loc,:);
            sim_scores(i,i_loc) = getNeighborsSimilarity(sim_input_data);
        end
    end



end