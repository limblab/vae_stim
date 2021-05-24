function [f] = plotNeighborSimilarityVsActivatedSimilarity(amp_input_data, amp_output_data, pd)

    % get PD of stim loc
    stim_idx = getStimLocationIdx(amp_output_data.stim_loc, amp_output_data.locs);
    stim_pd = pd(stim_idx);
    stim_pd(stim_pd<0) = stim_pd(stim_pd<0)+2*pi;
    
    
    
    subplot_counter = 1;
    f=figure('Position',[584 516 1142 420]);
    % get colormap
    color_list = colormap(colorcet('C3'));
    ax_list = [];
    for i_act = 1:numel(amp_input_data.direct_acts_test)
        for i_amp = 1:numel(amp_input_data.amps_test)
            ax_list(end+1)=subplot(numel(amp_input_data.direct_acts_test),numel(amp_input_data.amps_test),subplot_counter); hold on;
            
            act_func_mask = strcmpi(amp_output_data.act_func,amp_input_data.direct_acts_test{i_act})==1;
            amp_mask = amp_output_data.amp_list == amp_input_data.amps_test(i_amp);
            mask = act_func_mask & amp_mask;
            
            for i_loc = 1:numel(stim_pd)
                if(mask(i_loc)==1)
                    color_idx = ceil(size(color_list,1)*(stim_pd(i_loc))/(2*pi));
                    color_idx = max(1,min(color_idx,size(color_list,1)));
                    color_plot = color_list(color_idx,:);
                    plot(amp_output_data.neighbor_similarity(i_loc), amp_output_data.act_pop_similarity(i_loc),'.','color',color_plot,'markersize',8);
                end
            end
            
            subplot_counter = subplot_counter + 1;
            
            if(i_amp == 1 && i_act == 1)
                xlabel('Neighbor Similarity');
                ylabel('Activated Population Similarity');
                b=colorbar('Location','North');
                colormap(colorcet('C3'));
                b.Ticks = [0,0.25,0.5,0.75,1.0];
                b.TickLabels = {'0','90','180','270','360'};
            end
            formatForLee(gcf);
            set(gca,'fontsize',14);
        end
    end
    
    linkaxes(ax_list,'xy');
%     ylim([-1,1]);
%     xlim([-1,1]);







end