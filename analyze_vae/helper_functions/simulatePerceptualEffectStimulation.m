function [output_data] = simulatePerceptualEffectStimulation(input_data)

    % set random number generator. This is about 2x faster than the default
    % generator.
    rng('shuffle','threefry');


    % setup variables to output
    data_len = 2*floor(input_data.move_len/input_data.td(1).bin_size);
    
    % place array if desired
    neuron_mask = ones(size(input_data.PD));
    if(input_data.use_array)
        neuron_mask(:) = 0;
        % update neuron_mask to use only neurons that array "records" from
        % pick a random location and array orientation (angle)
        loc_start = input_data.locs(datasample(1:1:size(input_data.locs,1),1),:);
        array_angle = rand()*pi/2;
        
        % get grid, rotate around center and then translate
        [temp_x,temp_y] = meshgrid(0:500,0:500);
        array_locs = [reshape(temp_x,[],1), reshape(temp_y,[],1)];
        array_locs = array_locs*[cos(array_angle) -sin(array_angle); sin(array_angle), cos(array_angle)];
       
        array_locs = array_locs - median(array_locs);
        array_locs(array_locs(:,1) < -1 | array_locs(:,1) > 45,:) = [];
        array_locs(array_locs(:,2) < -1 | array_locs(:,2) > 45,:) = [];
        array_locs = input_data.elec_spacing*array_locs;
        % get closest neurons to array locations
        
        for i_elec = 1:size(array_locs,1)
            neuron_dist = sqrt(sum((array_locs(i_elec,:) - input_data.locs).^2,2));
            [min_dist,min_idx] = min(neuron_dist);
            if(min_dist < input_data.block_size)
                neuron_mask(min_idx) = 1;
            end
        end
        
    end
    
    run_data = [];
    for i_run = 1:input_data.n_runs
        % initialize data matrices
        run_data(i_run).no_stim_pred = zeros(input_data.n_test, 1);
        run_data(i_run).no_stim_class = zeros(size(run_data(i_run).no_stim_pred));
        run_data(i_run).no_stim_data = zeros(input_data.n_test, data_len);
        run_data(i_run).ang = zeros(input_data.n_test,1);
        
        n_stim_conds = input_data.n_locs*numel(input_data.amps_test);
        run_data(i_run).stim_pred = zeros(input_data.n_test,n_stim_conds);
        run_data(i_run).stim_data = zeros(input_data.n_test,n_stim_conds,2);
        run_data(i_run).stim_amp = zeros(1, n_stim_conds);
        run_data(i_run).stim_idx = zeros(input_data.n_stim_elec, n_stim_conds);
        
        % choose target axis for this run (0-359 degrees)
        tgt_axis = rand()*359;
        
        % choose stim location based on target axis

        idx_stim_poss = find(angleDiff(rad2deg(input_data.PD),tgt_axis,0,0) <= input_data.PD_tol);

        if(input_data.use_array)
            idx_stim_poss = intersect(idx_stim_poss, find(neuron_mask));
        end
    
        idx_stim = zeros(input_data.n_locs, input_data.n_stim_elec);
        
        for i = 1:input_data.n_locs
            idx_stim(i,:) = datasample(idx_stim_poss, input_data.n_stim_elec);
        end
        % for each target axis, train a new classifier
        input_data.tgt_axis = tgt_axis; % in degrees!
        [class_output_data] = trainMovementDirectionClassifier(input_data);
        
        % get no stim predictions
        [temp_val, temp_posterior] = predict(class_output_data.class_mdl, class_output_data.train_test_data.test_data_in);
        run_data(i_run).mdl = class_output_data.class_mdl;
        run_data(i_run).train_test_data = class_output_data.train_test_data;
        
        % store no stim predictions and data for this run
        run_data(i_run).no_stim_pred = temp_val; % store posterior for class 1 (in dir of target axis)
        run_data(i_run).no_stim_class = class_output_data.train_test_data.test_data_out;
        run_data(i_run).no_stim_data= class_output_data.train_test_data.test_data_in;
        run_data(i_run).ang = class_output_data.train_test_data.test_ang;
        
        % get which bin in the stim train each pulse is in
        stim_input = [];        
        stim_input.dec = input_data.dec;
        stim_input.act_func = input_data.act_func;
        stim_input.locs = input_data.locs;
        stim_timing = 0:1/input_data.freq:input_data.move_len;
        bin_edges = [0:1:ceil(input_data.move_len/input_data.td.bin_size)]*input_data.td.bin_size;
        [~,~,stim_input.pulse_bin_idx] = histcounts(stim_timing, bin_edges);
        stim_input.n_pulses = numel(stim_timing);
        stim_input.bin_size = input_data.td(1).bin_size;

        % for each amplitude and stim location, stimulate during each movement and classify
        % result
        stim_cond_counter = 1;
        for i_stim = 1:size(idx_stim,1)
            stim_input.stim_loc = input_data.locs(idx_stim(i_stim,:),:);
            for i_amp = 1:numel(input_data.amps_test)
                stim_input.amp=input_data.amps_test(i_amp);
                stim_input.can_be_act = [];
                for i_move = 1:numel(class_output_data.train_test_data.test_data_out)
                    start_idx = class_output_data.train_test_data.test_idx_start(i_move);
                    end_idx = start_idx + data_len/2 - 1;
                    stim_input.FR = input_data.td.VAE_firing_rates(start_idx:end_idx,:);
                    
                    stim_effect = getStimEffect(stim_input);
                    
                    % get activation pattern if i_move = 1;
                    if(i_move==1)
                        stim_input.can_be_act = stim_effect.can_be_act;
                    end
                    
                    dec_vel = stim_effect.FR_act*input_data.dec + input_data.bias;
                    mean_dec_vel = mean(dec_vel);
                    dec_vel = reshape(dec_vel,1,numel(dec_vel));

                    [curr_stim_val] = predict(class_output_data.class_mdl, mean_dec_vel);
                    
                    run_data(i_run).stim_pred(i_move, stim_cond_counter) = curr_stim_val;
                    run_data(i_run).stim_data(i_move, stim_cond_counter,:) = mean_dec_vel;
                    
                end
                run_data(i_run).stim_amp(1,stim_cond_counter) = input_data.amps_test(i_amp);
                run_data(i_run).stim_idx(:,stim_cond_counter) = idx_stim(i_stim,:)';
                
                stim_cond_counter = stim_cond_counter + 1;
            end
        end
        
        % store useful data for this run
        run_data(i_run).tgt_axis = tgt_axis;
        
        
    end

    % package outputs
    output_data = run_data;
    
    
end