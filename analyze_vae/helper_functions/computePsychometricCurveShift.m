function [PSD_data] = computePsychometricCurveShift(input_data)

    % find 50 percent point for no stim and stim condition, compare
    % movement angle at that point

    PSD_data = [];
    PSE = [];
    bin_edges = 5:10:175; % degrees
    for i_run = 1:numel(input_data)
        
        % get move angle relative to target axis
        run_data = input_data(i_run);
        test_ang_diff = angleDiff(run_data.tgt_axis, run_data.ang,0,0); % use degrees, ignore sign

        [counts,~,idx] = histcounts(test_ang_diff,bin_edges); % get data close to 90 degrees

        % get proportions for no stim case (idx==end) and all stim cases
        prop_tgt_axis = zeros(numel(bin_edges)-1,size(run_data.stim_pred,2)+1);

        for i_bin = 1:size(prop_tgt_axis,1)
            prop_tgt_axis(i_bin,end) = sum(run_data.no_stim_pred(idx==i_bin) > 0.5)/sum(idx==i_bin);
            prop_tgt_axis(i_bin,1:end-1) = sum(run_data.stim_pred(idx==i_bin,:) > 0.5,1)/sum(idx==i_bin);
            
        end

        for i_cond = 1:size(prop_tgt_axis,2)
            temp_PSE = find(prop_tgt_axis(:,i_cond) < 0.5,1,'first');
            
            if(isempty(temp_PSE))
                temp_PSE = size(prop_tgt_axis,1);
            end
            
            PSE(i_cond) = bin_edges(temp_PSE) + mode(diff(bin_edges))/2;
        end

        PSD_data(i_run,:) = PSE(1:end-1) - PSE(end);
        
    end
end