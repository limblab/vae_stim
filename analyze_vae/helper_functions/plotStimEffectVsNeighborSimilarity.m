function [f] = plotStimEffectVsNeighborSimilarity(amp_input_data, amp_output_data, pd_table)

    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    stim_pd = pd_table.velPD(stim_idx);
    stim_pd(stim_pd<0) = stim_pd(stim_pd<0)+2*pi;
    
    stim_vel = squeeze(mean(amp_output_data.stim_vel,2));
    stim_ang = atan2(stim_vel(:,2),stim_vel(:,1));
    
    % get colormap
    color_list = colormap(colorcet('C3'));
    
    
    f=figure('Position',[681 559 1144 420]);

    subplot_counter = 1;
    ax = [];
    max_edge = pi;
    for i_act = 1:numel(amp_input_data.acts_test)
        for i_amp = 1:numel(amp_input_data.amps_test)
            ax(end+1) = subplot(numel(amp_input_data.acts_test),numel(amp_input_data.amps_test),subplot_counter); hold on;
            act_func_mask = strcmpi(amp_output_data.act_func,amp_input_data.acts_test{i_act})==1;
            amp_mask = amp_output_data.amp_list == amp_input_data.amps_test(i_amp); 
            mask = act_func_mask & amp_mask;
            
            PD_effect_diff = cos(angleDiff(stim_pd,stim_ang,1,0));
            for i = 1:numel(PD_effect_diff)
                if(mask(i)==1)
                    color_idx = ceil(size(color_list,1)*(stim_pd(i))/(2*pi));
                    color_idx = max(1,min(color_idx,size(color_list,1)));
                    color_plot = color_list(color_idx,:);
                    plot(amp_output_data.neighbor_similarity(i), PD_effect_diff(i),'.','color',color_plot,'markersize',8);
                end
            end
            
            subplot_counter = subplot_counter + 1;
            
            if(i_act==1 && i_amp == 1)
                xlabel('Neighbor similarity');
                ylabel('Stim dir similarity');
            end
            formatForLee(gcf);
            set(gca,'fontsize',14);
        end
    end
    
    linkaxes(ax,'xy');
    xlim([-1,1]);
    ylim([-1,1]);




end