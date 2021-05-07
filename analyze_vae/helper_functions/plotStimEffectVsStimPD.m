function [f] = plotStimEffectVsStimPD(amp_input_data, amp_output_data, pd_table)

    bin_edges = 0:10:180;

    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    stim_PD = pd_table.velPD(stim_idx);
    
    stim_vel = squeeze(mean(amp_output_data.stim_vel,2));
    stim_ang = atan2(stim_vel(:,2),stim_vel(:,1));
    
    f=figure('Position',[681 559 1144 420]);

    subplot_counter = 1;
    ax = [];
    max_edge = pi;
    for i_act = 1:numel(amp_input_data.acts_test)
        for i_amp = 1:numel(amp_input_data.amps_test)
            ax(end+1) = subplot(numel(amp_input_data.acts_test),numel(amp_input_data.amps_test),subplot_counter); hold on;
            act_func_mask = strcmpi(amp_output_data.act_func,amp_input_data.acts_test{i_act})==1;
            amp_mask = amp_output_data.amp_list == amp_input_data.amps_test(i_amp); 
            
            neigh_mask = amp_output_data.neighbor_similarity > prctile(amp_output_data.neighbor_similarity,75);
            mask = act_func_mask & amp_mask & neigh_mask;
            
            
            h=histogram(rad2deg(angleDiff(stim_PD(mask),stim_ang(mask),1,0)),bin_edges,'Normalization','Probability');
            max_edge = max(h.BinEdges);
            subplot_counter = subplot_counter + 1;
            
            if(i_act==1 && i_amp == 1)
                xlabel('Stim direction relative to PD (degrees)');
                ylabel('Number of trials');
            end
            formatForLee(gcf);
            set(gca,'fontsize',14);
        end
    end
    
    linkaxes(ax,'xy');
    xlim([0,max_edge]);





end