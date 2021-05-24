function [FR_syn] = getSynapticEffects(input_data)
    % gets synaptic effect on firing rate. Intention is to add to baseline
    % firing rate.
    
    FR_syn = zeros(size(input_data.is_act_bin));
    switch(input_data.trans_act_func)
        case 'mexican_hat'
            % for each activated neuron, apply mexican hat function to
            % surrounding neurons. Use sigma = 2.0*input_data.block_size
            sigma_val = 2.0*input_data.block_size;
            FR_fact = 1;
            
            activated_neurons = find(any(input_data.is_act_bin,2));
           
            dist_mat = sqrt(sum((reshape(input_data.locs,size(input_data.locs,1),1,size(input_data.locs,2)) - ...
                reshape(input_data.locs(activated_neurons,:),1,numel(activated_neurons),size(input_data.locs,2))).^2,3));
            dist_mat(dist_mat==0) = 100000;

            mexh_vals = getMexicanHat(sigma_val,dist_mat);
            FR_syn = FR_syn + (FR_fact*mexh_vals)*input_data.is_act_bin(activated_neurons,:);
            
        case 'corr_project'
            conn_space_constant = 0.25; % mm
            tuning_space_constant = pi/2; % radians
            activated_neurons = find(any(input_data.is_act_bin,2));
            FR_fact = 1.0;
            
            prob_samp = rand(size(input_data.locs,1),numel(activated_neurons),2);
                        
            PD_diff = angleDiff(input_data.PD, input_data.PD(activated_neurons)',1,0); % use radians, don't preserve sign
                        
            dist_mat = sqrt(sum((reshape(input_data.locs,size(input_data.locs,1),1,size(input_data.locs,2)) - ...
                reshape(input_data.locs(activated_neurons,:),1,numel(activated_neurons),size(input_data.locs,2))).^2,3));
            dist_mat(dist_mat==0) = 100000;
            
            p_conn_dist = exp(-dist_mat/conn_space_constant);
            p_conn_exc = exp(-PD_diff/tuning_space_constant);
            p_conn_inhib = mean(p_conn_exc);
            
            is_exc = squeeze(prob_samp(:,:,1)) < p_conn_dist.*p_conn_exc;
            is_inhib = squeeze(prob_samp(:,:,2)) < p_conn_dist.*p_conn_inhib;

            FR_syn = FR_syn + FR_fact*(is_exc - is_inhib)*input_data.is_act_bin(activated_neurons,:);
            
        case 'none'
            
        case ''
            
            
        otherwise
            error('unknown activation function');
    end





end