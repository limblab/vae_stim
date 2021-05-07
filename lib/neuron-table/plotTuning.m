function plotTuning(neuron_table,params)
% PLOTFLATTUNING makes a single figure showing the tuning curve and PD with
% confidence intervals, unwrapped. Leave either entry blank to skip plotting it. Color
% is a 3 element vector for the color of the plotted tuning curve and PD.
% pdData is one row taken from binnedData object.
% Inputs:
%   neuron_table - table including PDs and tuning curves for one unit. Should either
%       be one row or several rows with the same signalID
%   params - parameters struct
%       .maxFR - (numeric) maximum value to plot (no default)
%       .minFR - (numeric) minimum value to plot (default: 0)
%       .unroll - (bool) whether to plot a flat curve or polar curve (default: false)
%       .color - (numeric or char) specify color to plot
%       .pd_colname - (char) name of PD column (default: 'velPD')
%       .curve_colname - (char) name of curve column (default: 'velCurve')
%       .plot_ci - (bool) whether to plot confidence interval (default: true)

maxFR = [];
minFR = 0;
unroll = false;
curve_colname = 'velCurve';
assignParams(who,params)

if isempty(maxFR)
    if ismember(sprintf('%sCIhigh',curve_colname),neuron_table.Properties.VariableNames)
        maxFR = max(max(neuron_table.(sprintf('%sCIhigh',curve_colname))));
    else
        maxFR = max(max(neuron_table.(curve_colname)));
    end
    % add to params for later
    params.maxFR = maxFR;
end
assert(maxFR>0, 'Need to provide a non-zero maxFR!')

hold on
if ~unroll
    % plot initial point
    h=polar(0,maxFR);
    set(h,'color','w')
end

% make sure there's only one unit here
sigIDs = unique(neuron_table.signalID,'rows');
if size(sigIDs,1)>1
    error('plotTuning:TooManyThings','neuron_table must contain only one unique signalID')
end

% do the plot
if height(neuron_table)==1
    plot_curve(neuron_table,params);
    plot_pd(neuron_table,params);
else
    % plot individual traces without CI (for now)
    indiv_params = params;
    indiv_params.plot_ci = false;
    for signum = 1:height(neuron_table)
        p = plot_curve(neuron_table(signum,:),indiv_params);
        set(p,'linewidth',0.45)
        p = plot_pd(neuron_table(signum,:),indiv_params);
        set(p,'linewidth',0.44)
    end

    % % plot average curve in thicker line
    % avg_table = neuronAverage(neuron_table,struct('keycols','signalID'));
    % p = plot_curve(avg_table,indiv_params);
    % set(p,'linewidth',2.1)

    % % plot average pd in thicker line
    % p = plot_pd(avg_table,indiv_params);
    % set(p,'linewidth',2.09)
end

% plot settings
if unroll
    set(gca,'box','off','tickdir','out',...
        'xlim',[-pi pi],'ylim',[minFR maxFR],...
        'xtick',[-pi, 0, pi],'xticklabel',{'-180','','180'})
end

end

function p = plot_curve(curve,params)
    % plotting params
    unroll = false;
    curve_colname = 'velCurve';
    color = 'k';
    plot_ci = true;
    assignParams(who,params)

    % only run if the column name is there
    if ismember(curve_colname,curve.Properties.VariableNames)
        bins = curve.bins;
        th_wrap = [bins-2*pi bins bins+2*pi];

        if plot_ci
            th_fill = [th_wrap fliplr(th_wrap)];
            r_fill = [repmat(curve.(sprintf('%sCIlow',curve_colname)),1,3),...
                fliplr(repmat(curve.(sprintf('%sCIhigh',curve_colname)),1,3))];
            th_fill = th_fill(~isnan(r_fill));
            r_fill = r_fill(~isnan(r_fill));
            % h=plot(th_fill,r_fill);
            % set(h,'linewidth',1.2,'color',color)
            if unroll
                patch(th_fill,r_fill,color,'edgealpha',0,'facealpha',0.3);
            else
                [x_fill,y_fill] = pol2cart(th_fill,r_fill);
                patch(x_fill,y_fill,color,'facealpha',0.3,'edgealpha',0);
            end
        end

        curve_wrap = repmat(curve.(curve_colname),1,3);
        th_wrap = th_wrap(~isnan(curve_wrap));
        curve_wrap = curve_wrap(~isnan(curve_wrap));
        if unroll
            p = plot(th_wrap,curve_wrap,'linewidth',2,'color',color);
        else
            p=polar(th_wrap,curve_wrap);
            set(p,'linewidth',2,'color',color)
        end
    end
end

function p = plot_pd(pdData,params)
    % plotting params
    maxFR = 0;
    unroll = false;
    pd_colname = 'velPD';
    color = 'k';
    pd_linspec = '-';
    plot_ci = true;
    assignParams(who,params)

    % only run if column name is there
    if ismember(pd_colname,pdData.Properties.VariableNames)
        if plot_ci
            % handle wraparound
            if pdData.(sprintf('%sCI',pd_colname))(2)<pdData.(sprintf('%sCI',pd_colname))(1)
                pdData.(sprintf('%sCI',pd_colname))(2) = pdData.(sprintf('%sCI',pd_colname))(2) + 2*pi;
            end
            th_fill = [pdData.(sprintf('%sCI',pd_colname))(1),...
                pdData.(sprintf('%sCI',pd_colname))(1),...
                pdData.(sprintf('%sCI',pd_colname))(2),...
                pdData.(sprintf('%sCI',pd_colname))(2)];
            r_fill = [0 maxFR maxFR 0];
            if unroll
                % h=plot(th_fill,r_fill);
                % set(h,'linewidth',1.2,'color',color)
                patch(th_fill,r_fill,color,'edgealpha',0,'facealpha',0.3);
                % plot wraparound
                % h=plot(th_fill-2*pi,r_fill);
                % set(h,'linewidth',1.2,'color',color)
                patch(th_fill-2*pi,r_fill,color,'edgealpha',0,'facealpha',0.3);
            else
                [x_fill,y_fill] = pol2cart(th_fill,r_fill);
                patch(x_fill,y_fill,color,'edgecolor','none','facealpha',0.3);
            end
        end

        if unroll
            p=plot(repmat(pdData.(pd_colname),2,1),maxFR*[0;1],pd_linspec);
            set(p,'linewidth',2,'color',color)
        else
            p=polar(repmat(pdData.(pd_colname),2,1),maxFR*[0;1],linspec);
            set(p,'linewidth',2,'color',color)
        end
    end
end
