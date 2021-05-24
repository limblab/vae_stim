function [ is_act, can_be_act, prob_act ] = activateNeurons( input_data )
% returns whether each neuron is activated for each pulse
% based on location, amplitude and
% activation function

% input_data contains: 
    % act_func : string picking the activation function
    % locs : location of neurons
    % stim_loc : location of stimulation
    % amp : stimulation amplitude
    % n_pulses : number of pulses. 
    
    is_act = []; prob_act = [];
    dist = sqrt(sum((input_data.locs - input_data.stim_loc).^2,2));
    
    if(~isfield(input_data,'can_be_act') || isempty(input_data.can_be_act))
        can_be_act = ones(size(dist)); % some act functions overwrite this variable.
        get_pattern = 1;
    else
        can_be_act = input_data.can_be_act;
        get_pattern = 0;
    end
    
    switch(input_data.direct_act_func)
        case 'circular'
            % pick radius of activation based on I = k*r^2 (r is radius, k is uA/mm)
            k = 1292;
            r = sqrt(input_data.amp/k);
            
            is_act = dist < r & can_be_act;
            is_act = repmat(is_act, 1, input_data.n_pulses);
        case 'exp_decay'
            % get space constant from linear interpolation
            amp_list = [15,30,50,100];
            space_constant_list = [100,250,325,500]/1000; % in mm
            space_constant = interp1(amp_list,space_constant_list,input_data.amp);
            
            prob_act = exp(-dist/space_constant);
            is_act = rand(numel(prob_act),input_data.n_pulses) < prob_act & can_be_act;
        case 'model_based' % sets pattern for each pulse
            % load space constants if needed. We actually are hard coding
            % them to speed up this process
%             load(['D:\Joseph\stimModel' filesep 'ModelSpreadFits_noSyn_diam2_15_30_50_100uA_logisticFit']);
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is inputted as mm, needs to be in um when fitting logistic curves
            
            is_act = rand(numel(prob_act),input_data.n_pulses+1) < prob_act & can_be_act;
        case 'model_based_set_pattern' % sets pattern once for each train
            % get hard coded model parameters
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves
            
            if(get_pattern)
                can_be_act = rand(numel(prob_act),1) < prob_act;
            end
            is_act = ones(numel(prob_act),input_data.n_pulses) & can_be_act;
        case 'model_based_floor' % sets pattern once for each pulse, prevents neurons with low probabilities from being activated
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves
            prob_act(prob_act < 0.25) = 0; % floor probability
            
            is_act = rand(numel(prob_act),input_data.n_pulses) < prob_act & can_be_act;
            
        case 'model_based_circle_corr'
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves

            % get correlation circle radius based on amplitude
            r_vals = [50,150,250,520]./1000; %mm 
            amp_vals = [15,30,50,100];
            corr_r = interp1(amp_vals,r_vals,input_data.amp,'linear','extrap');
            
            stim_loc_idx = getStimLocationIdx(input_data.stim_loc, input_data.locs);
            corr_loc_idx = find(sqrt(sum((input_data.locs - input_data.stim_loc).^2,2)) <= corr_r);
            
            corr_vals = mean(input_data.corr_table(:,corr_loc_idx),2);
            % adjust prob_act based on correlation; only for long range
            % connections; use circle to get correlation, increase radius
            % with amp based on model data
            
            % scale corr_vals so that mean is 0,
            corr_vals = corr_vals-mean(corr_vals);
            
            max_fact = 0.5; % scales effect of correlation on prob_act
            corr_fact = max_fact*(2*dist);
            corr_fact(dist > 0.5) = max_fact;
            
            corr_prob = (corr_fact.*corr_vals)+1; % want corr=[1,0,-1] -> prob=[1+corr_fact,1, 1-corr_factor]
                        
            prob_act = prob_act.*corr_prob;

            prob_act(prob_act>1) = 1; prob_act(prob_act < 0) = 0;
            
            is_act = rand(numel(prob_act),input_data.n_pulses) < prob_act & can_be_act;
        otherwise
            error('unknown activation function');
    end

end

