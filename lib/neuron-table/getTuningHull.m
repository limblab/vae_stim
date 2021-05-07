%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getTuningHull gets the convex hull of bootstrapped tuning weights for each
%   neuron in a tuningTable. Returns a table of tuning hulls
%
%   Inputs:
%       tuningTable
%       params - params struct
%           .CIpercentile - confidence interval percentage (defaults to 95)
%
%   Outputs:
%       tuningHulls - table of tuning convex hulls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuningHulls = getTuningHull(tuningTable,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFAULT PARAMETERS
CIpercentile = 95; % percent confidence interval
if nargin > 1, assignParams(who,params); end % overwrite parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find non-bootstrapped headers
weight_header_idx = endsWith(tuningTable.Properties.VariableNames,'Weight');
meta_header_idx = contains(tuningTable.Properties.VariableDescriptions,'meta');

% Loop through and get mean shifts out of bootstrapped shifts
unitIDs = unique(tuningTable(:,meta_header_idx));

% make empty hull container
hullCell = {};

% loop over all units
for i = 1:size(unitIDs,1)
    % loop over each weight column
    ID = tuningTable(:,meta_header_idx);
    unit_idx = ismember(ID,unitIDs(i,:));
    tuningTable_unit = tuningTable(unit_idx,weight_header_idx);

    hullRow = {};
    % loop over all weight columns
    for weight_col = 1:sum(weight_header_idx)
        % now create hulls and plot
        % get cluster in easy to work with form
        clust = tuningTable_unit{:,weight_col};
        means = mean(clust);
        centered_clust = clust-repmat(means,size(clust,1),1);

        % figure out 95% confidence interval
        dists = sqrt(sum(centered_clust.^2,2));
        inliers = dists<prctile(dists,CIpercentile);
        clust = clust(inliers,:);
        centered_clust = centered_clust(inliers,:);
        
        if size(clust,2)==1
            % one dimensional hull
            clust_hull = [min(centered_clust)+means max(centered_clust)+means];
        elseif size(clust,2)==2
            % two dimensional hull
            hull_idx = convhull(centered_clust);

            % get the actual points
            clust_hull = {centered_clust(hull_idx,:) + repmat(means,length(hull_idx),1)};
        else
            error('More than 2 tuning weights')
        end

        hullRow = [hullRow,clust_hull];
    end
    hullCell = [hullCell;hullRow];
end

hullTab = cell2table(hullCell,'VariableNames',tuningTable.Properties.VariableNames(weight_header_idx));

tuningHulls = [unitIDs hullTab];
