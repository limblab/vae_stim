%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [avg_data, cond_idx] = neuronAverage(neuronTable, params)
% 
% Averages over a NeuronTable structure for each given condition. Returns
% a new NeuronTable struct with one row per unique condition, with columns
% indicating mean, CIlow and CIhigh over the condition.
%
% Behaves very similar to trialAverage in TrialData structure
%
% INPUTS:
%   neuronTable : the table
%   params - parameters struct
%       .keycols : (int array or cell array of strings) indices for the columns 
%           to be used as keys into the NeuronTable. This function will average over
%           all rows with the same key.
%       .do_ci - whether to calculate confidence bounds by percentile (default - true)
%
% OUTPUTS:
%   avg_data : struct representing average across trials for each condition
%   cond_idx : cell array containing indices for each condition
%
% EXAMPLES:
%   e.g. to average over all target directions and task epochs
%       avg_data = trialAverage(trial_data,{'target_direction','epoch'});
%       Note: gives a struct of size #_TARGETS * #_EPOCHS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [avgTable,cond_idx] = neuronAverage(neuronTable, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
keycols = strcmpi(neuronTable.Properties.VariableDescriptions,'meta');
do_ci = true;
assignParams(who,params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% transform keycols into logical index array for simplicity
if ~islogical(keycols)
    if isnumeric(keycols)
        keycols = ismember(1:width(neuronTable),keycols);
    else % it's probably a cell array of strings
        keycols = ismember(neuronTable.Properties.VariableNames,keycols);
    end
end
assert(any(keycols),'Key columns do not match given table')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loop along conditions and get unique values for each
keyTable = unique(neuronTable(:,keycols));
cond_idx = false(height(keyTable),height(neuronTable));
tab_append = cell(height(keyTable),1);
for key_idx = 1:height(keyTable)
    key = keyTable(key_idx,:);
    cond_idx(key_idx,:) = ismember(neuronTable(:,keycols),key);
    neuronTable_select = neuronTable(cond_idx(key_idx,:),:);

    % extract data
    dataTable = neuronTable_select(:,~keycols);

    % figure out which columns have circular data
    meta_cols = strcmpi(dataTable.Properties.VariableDescriptions,'meta');
    circ_cols = contains(dataTable.Properties.VariableDescriptions,'circular');
    lin_cols = ~circ_cols & ~meta_cols;

    % warning for linear columns that aren't labeled as such
    nolabel_cols = lin_cols & ~strcmpi(dataTable.Properties.VariableDescriptions,'linear');
    if any(nolabel_cols)
        msg = sprintf('Some columns have VariableDescriptions not labeled linear or circular... Defaulting to linear mean. Column names:\n%s',repmat('%s\n',1,length(nolabel_cols)));
        warning(msg,dataTable.Properties.VariableNames{nolabel_cols})
    end
    % for all linear columns take regular mean
    meanTable_lin = varfun(@mean,dataTable(:,lin_cols));

    % strip 'mean' from variable names
    meanTable_lin.Properties.VariableNames = strrep(meanTable_lin.Properties.VariableNames,'mean_','');

    % for all circular data columns, take circ_mean
    meanTable_circ = varfun(@circ_mean,dataTable(:,circ_cols));

    % strip 'mean' from variable names
    meanTable_circ.Properties.VariableNames = strrep(meanTable_circ.Properties.VariableNames,'circ_mean_','');

    if do_ci
        % calculate confidence intervals
        ciLoArr = prctile(dataTable{:,lin_cols},2.5,1);
        ciHiArr = prctile(dataTable{:,lin_cols},97.5,1);

        ciLoTable = meanTable_lin;
        ciHiTable = meanTable_lin;
        ciLoTable{:,:} = ciLoArr;
        ciHiTable{:,:} = ciHiArr;
        ciLoTable.Properties.VariableNames = strcat(ciLoTable.Properties.VariableNames,'CILo');
        ciHiTable.Properties.VariableNames = strcat(ciHiTable.Properties.VariableNames,'CIHi');
        tab_append{key_idx} = horzcat(meanTable_circ,meanTable_lin,ciLoTable,ciHiTable);
    else
        tab_append{key_idx} = horzcat(meanTable_circ,meanTable_lin);
    end
end
avgTable = horzcat(keyTable,vertcat(tab_append{:}));
