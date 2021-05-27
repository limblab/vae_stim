function [ output_data ] = buildDecoderDropout( input_data )
% this function builds a decoder using dropout reguralization
% input data contains:
% lr : learning rate
% num_iters : number of iterations
% dropout_rate : percentage of neurons to drop for each iteration
% fr : firing rates to train with
% hand_vel : output data to predict

    if(size(input_data.fr,1) > size(input_data.fr,2))
        warning('Transposed FR because neurons should be along column');
        input_data.fr = input_data.fr';
    end

    if(isfield(input_data,'dec') && isfield(input_data,'bias'))
        dec = input_data.dec';
        bias = input_data.bias';
        dec = dec/(1-input_data.dropout_rate);
        warning('loaded dec and bias before training');
    else
        dec = rand(size(input_data.hand_vel,2),size(input_data.fr,1))-0.5;
        bias = zeros(size(input_data.hand_vel,2),1);
    end

    vaf_list_drop = zeros(input_data.num_iters,size(dec,1));
    vaf_list_mdl = zeros(input_data.num_iters,size(dec,1));
    n_neurons = size(input_data.fr,1);
    
    for i_iter = 1:input_data.num_iters
        % dropout inputs
        keep_mask = zeros(n_neurons,1);
        keep_mask(datasample(1:1:n_neurons,ceil(n_neurons*(1-input_data.dropout_rate)))) = 1;
        x = (keep_mask.*input_data.fr);
        
        vel_pred = dec*x + bias;

        d_dec = -2*x*(input_data.hand_vel-vel_pred')/length(dec);
        d_bias = mean(-2*(input_data.hand_vel-vel_pred'))';

        bias = bias - input_data.lr*d_bias;
        dec = dec - input_data.lr*d_dec';

        if(mod(i_iter,100)==0)
            vaf_list_drop(i_iter,:) = compute_vaf(input_data.hand_vel,vel_pred');
            vaf_list_mdl(i_iter,:) = compute_vaf(input_data.hand_vel,((dec*input_data.fr)*(1-input_data.dropout_rate) + bias)');
            disp([i_iter/input_data.num_iters, vaf_list_drop(i_iter,:), vaf_list_mdl(i_iter,:)])
        end
    end

    % adjust dec to deal with dropout
    dec = dec*(1-input_data.dropout_rate);
    
    % package outputs
    output_data.dec = dec';
    output_data.bias = bias';
    output_data.vaf_list_drop = vaf_list_drop;
    output_data.vaf_list_mdl = vaf_list_mdl;
    
    

end

