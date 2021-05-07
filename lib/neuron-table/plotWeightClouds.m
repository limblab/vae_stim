%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotWeightClouds Plots patches corresponding to 95% confidence intervals
%   of glm weights. Uses tuningTable structure, where each row is
%   a bootstrap sample of weights for a particular signal
%
%   Inputs:
%       tuningTable
%       params - parameters struct
%           .move_corr - movement correlate to plot weights of
%                        (defaults to 'vel')
%           .filter_tuning - whether to not plot weights that overlap 0,0
%                           default: true
%           .CI_thresh - threshold for confidence interval width in filter_tuning
%                       (defaults to pi/4)
%       varargin - various plotting parameters, like color and facealpha,
%               input like patch (see patch for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotWeightClouds(tuningTable,params,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFAULT PARAMETERS
move_corr      =  'vel';
filter_tuning = true; % filter by cloud width in one or both of the tables (should be either empty, 1, or 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some undocumented parameters
if nargin > 1, assignParams(who,params); end % overwrite parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loop through and get mean shifts out of bootstrapped shifts
modelIDs = unique(tuningTable(:,{'monkey','date','signalID'}));

% set up figure axes
figure
% plot([-pi pi],[0 0],'-k','linewidth',2)
% plot([0 0],[-pi pi],'-k','linewidth',2)
% plot([-pi pi],[-pi pi],'--k','linewidth',2)
hold on
axis equal
set(gca,'box','off','tickdir','out')

for i = 1:size(modelIDs,1)
    % make new table set
    % Populate new table with only that unit's PD shifts
    ID = tuningTable(:,{'monkey','date','signalID'});
    unit_idx = ismember(ID,modelIDs(i,:));
    tuningTable_unit = tuningTable(unit_idx,:);

    % now create hulls and plot
    % get cluster in easy to work with form
    clust = tuningTable_unit.([move_corr 'Weight']);
    means = mean(clust);
    centered_clust = clust-repmat(means,size(clust,1),1);

    % figure out 95% confidence interval by 'euclidean' distance to circular mean
    % should I be calculating geodesic distance somehow? seems like no, based on a preliminary google search
    dists = sqrt(sum(centered_clust.^2,2));
    inliers = dists<prctile(dists,95);
    clust = clust(inliers,:);
    centered_clust = centered_clust(inliers,:);

    % generate convex hull of inliers
    hull_idx = convhull(centered_clust);

    % check whether to plot cluster
    if filter_tuning
        % skip cluster if hull includes 0,0
        if inpolygon(0,0,centered_clust(hull_idx,1)+means(1),centered_clust(hull_idx,2)+means(2))
            continue
        end
    end
    
    % plot hull
    patch(centered_clust(hull_idx,1)+means(1),centered_clust(hull_idx,2)+means(2),varargin{:})
    
    % plot clusters
    % scatter(centered_clust(:,1)+means(1),centered_clust(:,2)+means(2),[],get(p,'facecolor'));
end
