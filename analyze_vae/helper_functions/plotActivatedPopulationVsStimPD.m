function [f] = plotActivatedPopulationVsStimPD(amp_input_data, amp_output_data, pd_table)
    
     % get neuron idx for each stimulation
    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    
    stim_PD = pd_table.velPD(stim_idx);
    PD_diff_stim_loc = stim_PD - pd_table.velPD';
    
    % remove stimulation location
    PD_diff_stim_loc = makeStimLocationNan(PD_diff_stim_loc,stim_idx, size(amp_output_data.locs,1));

    % mask PD diff for neurons that were active only.
    PD_diff_act = PD_diff_stim_loc.*(amp_output_data.num_pulses_active > 0);

    % get neighbor similarity for each stimulation location
    % test PD neighborhood 
    sim_input_data = amp_output_data;
    sim_input_data.metric_name = 'PD';
    sim_input_data.PD = pd_table.velPD; % PD must be in radians!
    sim_input_data.is_ang = 1;
       
    sim_input_data.nbor_max_r = 0.3; % in mm
    sim_input_data.nbor_min_r = 0;
    
    neighbor_similarity = getNeighborsSimilarity(sim_input_data);
        
    % make plot
    f=figure();
    for i_act = 1:numel(amp_input_data.acts_test)
        subplot(1,numel(amp_input_data.acts_test),i_act); hold on;
        for i_amp = 1:numel(amp_input_data.amps_test)
            act_func_mask = strcmpi(amp_output_data.act_func,amp_input_data.acts_test{i_act})==1;
            amp_mask = amp_output_data.amp_list == amp_input_data.amps_test(i_amp);
            neigh_mask = neighbor_similarity > prctile(neighbor_similarity,90);
            mask = act_func_mask & amp_mask & neigh_mask;
            
            num_pulses_active_cond = amp_output_data.num_pulses_active(mask, :);
            PD_diff_cond = PD_diff_act(mask , :);
                        
            PD_diff_list = rad2deg(abs(PD_diff_cond(num_pulses_active_cond > 0)));
            num_active_list = num_pulses_active_cond(num_pulses_active_cond > 0);
            
            PD_diff_list = repelem(PD_diff_list,num_active_list);
            
            histogram(PD_diff_list,0:10:180,'DisplayStyle','stairs','EdgeColor',getColorFromList(1,i_amp-1),'linewidth',2,'Normalization','Probability');
        end
        formatForLee(gcf);
        set(gca,'fontsize',14);
        if(i_act==1)
            xlabel('|Activated PD - Stimulated PD| (degrees)');
            ylabel('Proportion of activated neurons');
            l=legend(num2str(amp_input_data.amps_test'));
            set(l,'box','off');
        end
    end
    
end