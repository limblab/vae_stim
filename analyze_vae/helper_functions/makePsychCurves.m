function [figs] = makePsychCurves(class_data, plot_data)

    figs = [];
    for i_run = 1:numel(class_data)
        
        % get move angle relative to target axis
        run_data = class_data(i_run);
        test_ang_diff = angleDiff(run_data.tgt_axis, run_data.ang,0,0); % use degrees, ignore sign

        [counts,~,idx] = histcounts(test_ang_diff,plot_data.bin_edges);

        % get proportions for no stim case (idx==end) and all stim cases
        prop_tgt_axis = zeros(numel(plot_data.bin_edges)-1,size(run_data.stim_pred,2)+1);
        mean_post = zeros(size(prop_tgt_axis));

        for i_bin = 1:size(prop_tgt_axis,1)
            prop_tgt_axis(i_bin,end) = sum(run_data.no_stim_pred(idx==i_bin) > 0.5)/sum(idx==i_bin);
            prop_tgt_axis(i_bin,1:end-1) = sum(run_data.stim_pred(idx==i_bin,:) > 0.5,1)/sum(idx==i_bin);
            
            mean_post(i_bin,end) = mean(run_data.no_stim_pred(idx==i_bin));
            mean_post(i_bin,1:end-1) = mean(run_data.stim_pred(idx==i_bin,:),1);
        end

        figs(i_run) = figure(); hold on;
        
        % no stim
        plot(plot_data.bin_edges(1:end-1)+mode(diff(plot_data.bin_edges))/2,prop_tgt_axis(:,end),'k--','linewidth',2);
        
%         % stim
%         color_use = inferno(size(prop_tgt_axis,2));
%         for i_stim = 1:size(prop_tgt_axis,2)-1
%             plot(plot_data.bin_edges(1:end-1)+mode(diff(plot_data.bin_edges))/2,prop_tgt_axis(:,i_stim),...
%                 'color',color_use(i_stim,:),'linewidth',2);
%         end
        
        
        
        formatForLee(gcf);
        set(gca,'fontsize',14);
        xlabel('Movement direction (degrees)');
        ylabel('Proportion 0-degree target');

    end


end