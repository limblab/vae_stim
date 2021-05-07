function [arr] = makeStimLocationNan(arr, stim_idx, n_neurons)

    % transpose arr if neurons are not the rows
    did_transpose = 0;
    if(size(arr,2) == n_neurons)
        arr = arr';
        did_transpose = 1;
    elseif(size(arr,1)~= n_neurons)
        error('size of first argument seems wrong');
    end
        
    offset = n_neurons*(0:1:numel(stim_idx)-1)';
    arr(stim_idx + offset) = nan;

    if(did_transpose)
        arr = arr';
    end

end