%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function neuronTable = makeNeuronTableStarter(trial_data,params)
%
%   Makes the neuron table starter from the trial data structure
%
% INPUTS:
%   trial_data : the struct
%   params     : parameter struct
%       .out_signal_names : names of signals to be used as signalID weightTable
%                           default - empty
%       .meta         : include any conditions that these fits should be tagged as
%                       Ex. meta = struct('spaceNum',1) % for PM space fits
% OUTPUTS:
%   neuronTable : neuron table starter (first few columns of a neuron table)
%
% Written by Raeed Chowdhury. Updated Nov 2017.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function neuronTable = makeNeuronTableStarter(trial_data,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFAULT PARAMETERS
out_signal_names = {};
assignParams(who,params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get height of table
tab_height = size(out_signal_names,1);
if numel(unique(cat(1,{trial_data.monkey}))) > 1
    error('More than one monkey in trial data')
end
monkey = repmat({trial_data(1).monkey},tab_height,1);
if isfield(trial_data,'date')
    if numel(unique(cat(1,{trial_data.date}))) > 1
        date = cell(tab_height,1);
        warning('More than one date in trial data')
    else
        date = repmat({trial_data(1).date},tab_height,1);
    end
elseif isfield(trial_data,'date_time')
    % split and check
    date_temp = regexp(cat(1,{trial_data.date_time}),'\d*\/\d*\/\d*','match');
    date_temp = vertcat(date_temp{:});
    if numel(unique(date_temp)) > 1
        date = cell(tab_height,1);
        warning('More than one date in trial data')
    else
        date = repmat(date_temp(1),tab_height,1);
    end
end

if numel(unique(cat(1,{trial_data.task}))) > 1
    task = cell(tab_height,1);
    warning('More than one task in trial data')
else
    task = repmat({trial_data(1).task},tab_height,1);
end


neuronTable = table(monkey,date,task,out_signal_names,'VariableNames',{'monkey','date','task','signalID'});

% add meta fields if there are any
if isfield(params,'meta')
    fields = fieldnames(params.meta);
    tab_append = cell(1,numel(fields));
    for fn = 1:numel(fields)
        metafield = params.meta.(fields{fn});
        arr_append = repmat(metafield,tab_height,1);
        tab_append{fn} = table(arr_append,'VariableNames',{fields{fn}});
    end
    neuronTable = horzcat(neuronTable,tab_append{:});
end

% Describe all starter columns as meta for future ease
neuronTable.Properties.VariableDescriptions = repmat({'meta'},1,width(neuronTable));
