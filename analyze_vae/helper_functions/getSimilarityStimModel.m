function [sim_out] = getSimilarityStimModel(metric_all, num_pulses_active, metric_stim, is_angular)

    sim_out = nan(size(metric_all,1),1);

    for i_sim = 1:numel(sim_out)
        % remove stim idx from metric_all
        mask = num_pulses_active(i_sim,:) > 0;
        metric_list = metric_all(i_sim,mask);
        num_pulses_list = num_pulses_active(i_sim,mask);
        
        if(is_angular)
            sim_out(i_sim) = sum(num_pulses_list.*cos(angleDiff(metric_list, metric_stim(i_sim),1,0)))/sum(num_pulses_list); % use radians, return absolute value
        else
            sim_out(i_sim) = sum(num_pulses_list.*((metric_list-metric_stim(i_sim)).^2))/sum(num_pulses_list);
        end
        
    end
    


end