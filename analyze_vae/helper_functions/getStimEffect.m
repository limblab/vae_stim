function [output_data] = getStimEffectParallel (input_data)    
% input_data contains:
% FR : firing rate of neurons in Hz
% dec : decoder (just velocity part, not bias)
% direct_act_func : activation function for activateNeurons
% trans_act_func : activation function for getSynapticEffects
% locs : location of neurons
% stim_loc = location of stimulation
% amp : stimulation amplitude
% n_pulses : number of pulses to stimulate with
% pulse_bin_idx : which trial_data bin each pulse is within
% bin_size : size of bin
% corr_table : correlations across neurons, if using model_act_corr

    output_data = [];
    % set rng to use the fastest method when tested
    rng('shuffle','multFibonacci');
    
    % starting with FR_base (FR in input_data).
    % find FR_act by activating neurons for each pulse. Set FR_act as a
    % firing rate, not spike count
    store_pattern = 1;
    if(isfield(input_data,'can_be_act') && ~isempty(input_data.can_be_act))
        store_pattern = 0;
    else
        input_data.can_be_act = [];
    end
    
    % handles arbitrary number of stim electrodes, need to get activated
    % population for each electrode
    FR_act = input_data.FR;
    all_stim_locs = input_data.stim_loc;
    all_can_be_act = input_data.can_be_act;
    for i_loc = 1:size(all_stim_locs,1)
        input_data.stim_loc = all_stim_locs(i_loc,:);
        if(~isempty(all_can_be_act))
            input_data.can_be_act = all_can_be_act(:,i_loc);
        end
        
        if(i_loc==1)
            [is_act, can_be_act_temp, prob_act] = activateNeurons(input_data); % input_data is already formatted properly
        else
            [is_act_temp,can_be_act_temp] = activateNeurons(input_data);
            is_act = is_act | is_act_temp;
        end
        if(store_pattern)
            new_can_be_act(:,i_loc) = can_be_act_temp;
        end
    end
    
    % change activated neurons firing rates based on pulses they were
    % responsive to for each bin. This is for directly activated neurons
    is_act_bin = false(size(is_act,1),size(input_data.FR,1));
    for i_bin = 1:size(input_data.FR,1)
        is_act_bin(:,i_bin) = any(is_act(:,input_data.pulse_bin_idx==i_bin),2);
        FR_act(i_bin,is_act_bin(:,i_bin)==1) = sum(is_act(is_act_bin(:,i_bin)==1,input_data.pulse_bin_idx==i_bin),2)'/input_data.bin_size;
    end
    
    % get synaptic effects and modulate firing rate based on 
    syn_input_data = [];
    syn_input_data.is_act_bin = is_act_bin; 
    syn_input_data.trans_act_func = input_data.trans_act_func;
    syn_input_data.locs = input_data.locs;
    syn_input_data.corr_table = input_data.corr_table;
    syn_input_data.block_size = input_data.block_size;
    syn_input_data.PD = input_data.PD;
    
    FR_syn = getSynapticEffects(syn_input_data)';
    
    FR_act = FR_act + FR_syn;
    
    % find FR_stim as FR_act - FR_base
    FR_stim = FR_act - input_data.FR;
    
    % get (x,y) vel based on decoder;
    act_vel = FR_act*input_data.dec;
    base_vel = input_data.FR*input_data.dec;
    stim_vel = FR_stim*input_data.dec;
    
    % output x,y vel and other stuff
    output_data.stim_vel = stim_vel;
    output_data.act_vel = act_vel;
    output_data.base_vel = base_vel;
    
    output_data.FR_act = FR_act;
    output_data.FR_stim = FR_stim;
    output_data.FR_syn = FR_syn;
    
    output_data.is_act = is_act;
    output_data.prob_act = prob_act;
    if(store_pattern)
        output_data.can_be_act = new_can_be_act;
    else
    	output_data.can_be_act = all_can_be_act;
    end
end