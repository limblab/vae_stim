function compareTuning(neuron_table,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   comares tuning between different conditions with empirical
%   tuning curves and PDs.
%   Inputs - 
%       neuron_table - table of neurons with tuning curves and pds
%       params - parameters struct
%           .cond_cols - column names of table that define the condition
%           .signalIDs - (optional) matrix where rows indicate which signalIDs
%                       to plot. default - plots them all
%           .maxFR - (optional) array of maximum firing rates to
%                   display in polar plots. Default behavior is currently broken
%           .curve_colname - name of tuning curve column (default - 'velCurve')
%           .pd_colname - name of PD column (default - 'velPD')
%           .cond_colors - colors for conditions
%               (default - linspecer(num_conditions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial check
assert(istable(neuron_table),'neuron_table must be a table!')

% Default Params
cond_cols = {};
signalIDs = unique(neuron_table.signalID,'rows');
curve_colname = 'velCurve';
pd_colname = 'velPD';
cond_colors = [];
% get maxFR for each neuron
maxFR = 0;

assignParams(who,params)

% get information about table
conditions = unique(neuron_table(:,cond_cols),'rows');
if isempty(cond_colors)
    cond_colors = linspecer(height(conditions));
end

% check inputs
if numel(maxFR) == 1
    maxFR = repmat(maxFR,size(signalIDs,1),1);
end
assert(numel(maxFR)==size(signalIDs,1),'maxFR is wrong size')
assert(size(cond_colors,1)==height(conditions),'Number of colors must match number of conditions')

%% Plot tuning curves
% number of subplots (include plot for legends)
n_rows = ceil(sqrt(size(signalIDs,1)+1));
% make plots
for neuron_idx = 1:size(signalIDs,1)
    subplot(n_rows,n_rows,neuron_idx)

    % get maxFR for each neuron if not passed in
    if maxFR(neuron_idx) == 0
        [~,temp_table] = getNTidx(neuron_table,'signalID',signalIDs(neuron_idx,:));
        if ismember(sprintf('%sCIhigh',curve_colname),temp_table.Properties.VariableNames)
            maxFR(neuron_idx) = max(max(temp_table.(sprintf('%sCIhigh',curve_colname))));
        else
            maxFR(neuron_idx) = max(max(temp_table.(curve_colname)));
        end
    end

    for cond_idx = 1:height(conditions)
        % put cond entry into cell for expansion...
        cond_cell = [conditions.Properties.VariableNames; table2cell(conditions(cond_idx,:))];
        % get all entries pertaining to condition and neuron
        [~,temp_table] = getNTidx(neuron_table,'signalID',signalIDs(neuron_idx,:),cond_cell{:});

        plotTuning(temp_table,...
            struct('maxFR',maxFR(neuron_idx),...
                'unroll',true,...
                'color',cond_colors(cond_idx,:),...
                'plot_ci',false,...
                'pd_colname',pd_colname,...
                'curve_colname',curve_colname));
        hold on
    end
    if isnumeric(signalIDs(neuron_idx,:))
        label = ['Neuron ' num2str(signalIDs(neuron_idx,:))];
    else
        label = ['Neuron ' signalIDs(neuron_idx,:)];
    end

    title(label)
end

% for a legend...
subplot(n_rows,n_rows,n_rows^2)
for cond_idx = 1:height(conditions)
    plot([0 1],repmat(cond_idx,1,2),'-','linewidth',2,'color',cond_colors(cond_idx,:))
    hold on
end
ylabel 'Condition number'
set(gca,'box','off','tickdir','out','xtick',[],'ytick',1:height(conditions))
