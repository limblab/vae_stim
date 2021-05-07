function [ is_act, can_be_act ] = activateNeurons( input_data )
% returns whether each neuron is activated for each pulse
% based on location, amplitude and
% activation function

% input_data contains: 
    % act_func : string picking the activation function
    % locs : location of neurons
    % stim_loc : location of stimulation
    % amp : stimulation amplitude
    % n_pulses : number of pulses. 
    
    is_act = [];
    dist = sqrt(sum((input_data.locs - input_data.stim_loc).^2,2));
    
    if(~isfield(input_data,'can_be_act') || isempty(input_data.can_be_act))
        can_be_act = ones(size(dist)); % some act functions overwrite this variable.
        get_pattern = 1;
    else
        can_be_act = input_data.can_be_act;
        get_pattern = 0;
    end
    
    switch(input_data.act_func)
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
        case 'model_based'
            % load space constants
%             load(['D:\Joseph\stimModel' filesep 'ModelSpreadFits_noSyn_diam2_15_30_50_100uA_logisticFit']);
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves
            
            is_act = rand(numel(prob_act),input_data.n_pulses+1) < prob_act & can_be_act;
        case 'model_based_act'
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves
            
            if(get_pattern)
                can_be_act = rand(numel(prob_act),1) < prob_act;
            end
            is_act = ones(numel(prob_act),input_data.n_pulses+1) & can_be_act;
        case 'model_based_floor'
            [a_val,b_val] = getModelBasedActivationParameters(input_data.amp);
            
            prob_act = 1-1./(1+exp(-a_val*(dist*1000 - b_val))); % dist is in mm, was in um when fitting logistic curves
            prob_act(prob_act < 0.05) = 0; % floor probability
            
            is_act = rand(numel(prob_act),input_data.n_pulses+1) < prob_act & can_be_act;
            
        otherwise
            error('unknown activation function');
    end

end

