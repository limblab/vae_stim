%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% comparePDClouds Plots hollow patches corresponding to 95% confidence intervals
%   of PDs against each other. Uses pdTable structure, where each row is
%   a bootstrap sample of a PD for a particular signal
%
%   Inputs:
%       pdTable1 - PD table for x-axis
%       pdTable2 - PD table for y-axis
%       params - parameters struct
%           .move_corr - movement correlate that PD was calculated on
%                        (defaults to 'vel')
%           .filter_tuning - which of the pdTables to filter units out by
%                       Checks if 95% confidence interval along a particular
%                       direction < CI_thresh
%                       Possible values are: [],1,2 (defaults to [])
%           .CI_thresh - threshold for confidence interval width in filter_tuning
%                       (defaults to pi/4)
%       varargin - various plotting parameters, like color and linewidth,
%               input like plot (see plot for details)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function comparePDClouds(pdTable1,pdTable2,params,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFAULT PARAMETERS
move_corr      =  'vel';
filter_tuning = []; % filter by cloud width in one or both of the tables (should be either empty, 1, or 2)
CI_thresh = pi/4; % threshold for what's considered a tuned neuron in filtering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some undocumented parameters
if nargin > 1, assignParams(who,params); end % overwrite parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loop through and get mean shifts out of bootstrapped shifts
%error
keys = unique(pdTable1(:,{'monkey','date','signalID'}));

% set up figure axes
figure
plot([-pi pi],[0 0],'-k','linewidth',2)
hold on
plot([0 0],[-pi pi],'-k','linewidth',2)
plot([-pi pi],[-pi pi],'--k','linewidth',2)
axis equal
set(gca,'box','off','tickdir','out','xtick',[-pi pi],'ytick',[-pi pi],'xlim',[-pi pi],'ylim',[-pi pi],...
'xticklabel',{'-\pi','\pi'},'yticklabel',{'-\pi','\pi'})

for i = 1:size(keys,1)
    % make new table set
    % Populate new table with only that unit's PD shifts
    ID = pdTable1(:,{'monkey','date','signalID'});
    unit_idx = ismember(ID,keys(i,:));
    pdTable_unit1 = pdTable1(unit_idx,:);
    pdTable_unit2 = pdTable2(unit_idx,:);

    % now create hulls and plot

    % first plot cloud with center point
    %comparePDs(pdTable_unit1,pdTable_unit{modelctr},struct('move_corr','vel'),[model_cstrings{modelctr} 'o'],'linewidth',2)
    %plot(circ_mean(pdTable_unit1.([move_corr 'PD'])),circ_mean(pdTable_unit{modelctr}.([move_corr 'PD'])),'ko','linewidth',2);

    % get cluster in easy to work with form
    clust = [pdTable_unit1.([move_corr 'PD']) pdTable_unit2.([move_corr 'PD'])];
    means = [circ_mean(pdTable_unit1.([move_corr 'PD'])) circ_mean(pdTable_unit2.([move_corr 'PD']))];
    centered_clust = minusPi2Pi(clust-repmat(means,size(clust,1),1));

    % check whether to plot cluster
    if ~isempty(filter_tuning)
        if ismember(1,filter_tuning)
            % skip cluster if range is above CI_thresh
            if diff(prctile(centered_clust(:,1),[2.5 97.5]))>CI_thresh
                % keys(i,:)
                continue
            end
        end
        if ismember(2,filter_tuning)
            % skip if range is above CI_thresh
            % skip cluster if range is above CI_thresh
            if diff(prctile(centered_clust(:,2),[2.5 97.5]))>CI_thresh
                % keys(i,:)
                continue
            end
        end
    end
    
    % figure out 95% confidence interval by 'euclidean' distance to circular mean
    % should I be calculating geodesic distance somehow? seems like no, based on a preliminary google search
    dists = sqrt(sum(centered_clust.^2,2));
    inliers = dists<prctile(dists,95);
    centered_clust = centered_clust(inliers,:);

    % generate convex hull of inliers
    hull_idx = convhull(centered_clust);

    % plot cloud
    % plot hull (have to figure out what to do about wraparound)
    % patch(centered_clust(hull_idx,1)+means(1),centered_clust(hull_idx,2)+means(2),varargin{:});
    % patch(centered_clust(hull_idx,1)+means(1)-2*pi,centered_clust(hull_idx,2)+means(2),varargin{:});
    % patch(centered_clust(hull_idx,1)+means(1),centered_clust(hull_idx,2)+means(2)-2*pi,varargin{:});
    % patch(centered_clust(hull_idx,1)+means(1)-2*pi,centered_clust(hull_idx,2)+means(2)-2*pi,varargin{:});

    % plot central points
    scatter(means(1),means(2),50,'color','filled');

    
    % plot clusters
    % scatter(centered_clust(:,1)+means(1),centered_clust(:,2)+means(2),[],get(p,'facecolor'));
    % scatter(centered_clust(:,1)+means(1)-2*pi,centered_clust(:,2)+means(2),[],get(p,'facecolor'));
    % scatter(centered_clust(:,1)+means(1),centered_clust(:,2)+means(2)-2*pi,[],get(p,'facecolor'));
    % scatter(centered_clust(:,1)+means(1)-2*pi,centered_clust(:,2)+means(2)-2*pi,[],get(p,'facecolor'));
end
