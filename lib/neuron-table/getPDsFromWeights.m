%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function pdTable = getPDsFromWeights(weightTable)
%
%   Gets PD table for given out_signal. You need to define the out_signal
% and move_corr parameters at input.
%
% INPUTS:
%   weightTable : the neuron table that contains weights calculated from model
%
% OUTPUTS:
%   pdTable : calculated PD table with CIs
%
% Written by Raeed Chowdhury. Updated Nov 2017.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pdTable = getPDsFromWeights(weightTable)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preallocate final table
% get weight column indices
baseline_cols = contains(weightTable.Properties.VariableNames,'baseline');
weight_cols = endsWith(weightTable.Properties.VariableNames,'Weight') & ~baseline_cols;
% eval_cols = endsWith(weightTable.Properties.VariableNames,'eval');
% get key column indices
% key_cols = ~weight_cols & ~baseline_cols & ~eval_cols;
key_cols = contains(weightTable.Properties.VariableDescriptions,'meta');
keyTable = unique(weightTable(:,key_cols));
pdTable = unique(weightTable(:,key_cols));

% add columns to pdTable for each input signal
% first baseline columns
baseline_cols_idx = find(baseline_cols);
tab_append = cell(1,length(baseline_cols_idx));
for in_signal_idx = 1:length(baseline_cols_idx)
    mean_append = zeros(height(pdTable),1);
    CI_append = zeros(height(pdTable),2);
    % loop over keys in pdTable
    for key_idx = 1:height(keyTable)
        % get key
        key = keyTable(key_idx,:);
        % select all entries in weightTable matching key
        weightTable_select = weightTable(ismember(weightTable(:,key_cols),key),:);

        % get title of baseline column
        col_title = weightTable_select.Properties.VariableNames{baseline_cols_idx(in_signal_idx)};
        weights = weightTable_select.(col_title);

        % get mean weight
        mean_append(key_idx) = mean(weights);

        % get CIs
        CI_append(key_idx,:) = prctile(weights,[2.5 97.5]);
    end
    tab_append{in_signal_idx} = table(mean_append,CI_append,'VariableNames',{col_title,[col_title 'CI']});
end
pdTable = horzcat(pdTable,tab_append{:});

% now weight columns
weight_cols_idx = find(weight_cols);
tab_append = cell(1,length(weight_cols_idx));
for in_signal_idx = 1:length(weight_cols_idx)
    % preallocate
    PD_append = zeros(height(pdTable),1);
    moddepth_append = zeros(height(pdTable),1);
    PDCI_append = zeros(height(pdTable),2);
    moddepthCI_append = zeros(height(pdTable),2);
    % loop over keys in pdTable
    for key_idx = 1:height(pdTable)
        % get key
        key = keyTable(key_idx,:);
        % select all entries in weightTable matching key
        weightTable_select = weightTable(ismember(weightTable(:,key_cols),key),:);

        col_title = weightTable_select.Properties.VariableNames{weight_cols_idx(in_signal_idx)};
        in_signal_name = extractBefore(col_title,'Weight');
        weights = weightTable_select.(col_title);

        % get mean PD and moddepth
        [PD_append(key_idx,:),moddepth_append(key_idx,:)] = cart2pol(mean(weights(:,1)),mean(weights(:,2)));

        % now for CIs...
        % get th and r for tuning weights
        [th,r] = cart2pol(weights(:,1),weights(:,2));
        % get CI of moddepth
        moddepthCI_append(key_idx,:) = prctile(r,[2.5 97.5]);
        % get circular CI of PD
        % PD_append(key_idx,:) = circ_mean(th);
        PDCI_append(key_idx,:) = minusPi2Pi(prctile(minusPi2Pi(th-PD_append(key_idx,:)),[2.5 97.5]) + PD_append(key_idx,:));

    end
    tab_append{in_signal_idx} = table(PD_append,PDCI_append,moddepth_append,moddepthCI_append,'VariableNames',strcat(in_signal_name,{'PD','PDCI','Moddepth','ModdepthCI'}));
    tab_append{in_signal_idx}.Properties.VariableDescriptions = {'circular','circular','linear','linear'};
end
pdTable = horzcat(pdTable,tab_append{:});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
