function [output_data] = getMovementTrainingTestingData(input_data)
% returns training and testing data for training a classifier. 
% input_data contains : 
%   td : trial data
%   in_sig : field to use as movement data
%   move_len : length of movement
%   min_sig : minimum value of signal
%   max_sig : maximum value of signal
%   n_train : number of training trials
%   n_test : number of testing trials
%   tgt_axis : In degrees, classify movements as this direction or 180 degrees away.

    n_bins = floor(input_data.move_len/input_data.td(1).bin_size);

    signal = input_data.td.(input_data.in_sig);
    
    % make set of possible signals by shifting data
    signal = repmat(reshape(signal,size(signal,1),1,size(signal,2)),1,n_bins,1);
    idx_start = [1:1:size(signal,1)]';
    for i_bin = 2:n_bins
        signal(:,i_bin,:) = circshift(signal(:,1,:),-1*(i_bin-1));
    end
    
    % make each entry in signal unique
    signal = signal(n_bins+1:n_bins:end-n_bins,:,:);
    idx_start = idx_start(n_bins+1:n_bins:end-n_bins,1);
    
    % then take mean and remove any below or above speed bounds. 
    signal_mean = mean(sqrt(sum(signal.^2,2)),3);
    keep_mask = signal_mean > input_data.min_sig & signal_mean < input_data.max_sig;
    % remove any directions that are too close to 0 and 180 (since these
    % are uninteresting).
    signal_vel = squeeze(mean(signal,2));
    signal_dir = rad2deg(atan2(signal_vel(:,2),signal_vel(:,1)));
    keep_mask = keep_mask & angleDiff(signal_dir,input_data.tgt_axis,0,0) > 35 & angleDiff(signal_dir,input_data.tgt_axis,0,0) < 145;
    
    
    signal = signal(keep_mask == 1,:,:);
    idx_start = idx_start(keep_mask==1,1);

    % sample training and testing data set
    if(input_data.n_test+input_data.n_train > numel(idx_start))
        train_prop = input_data.n_train/(input_data.n_test+input_data.n_train);
        input_data.n_test = floor(numel(idx_start)*(1-train_prop));
        input_data.n_train = floor(numel(idx_start)*train_prop);
    end
    
    sample_idx = datasample(1:1:numel(idx_start),input_data.n_test+input_data.n_train,'Replace',false);
    
    sample_data = signal(sample_idx,:,:);
    mean_sample_vel = squeeze(mean(sample_data,2));
    sample_dir = rad2deg(atan2(mean_sample_vel(:,2),mean_sample_vel(:,1)));
    sample_class = angleDiff(sample_dir, input_data.tgt_axis,0,0) < 90;
    
    sample_data = reshape(sample_data,size(sample_data,1),size(sample_data,2)*size(sample_data,3));

    train_data = mean_sample_vel(1:input_data.n_train,:);
    train_class = sample_class(1:input_data.n_train);
    train_idx_start = idx_start(sample_idx(1:input_data.n_train),:,:);
    train_ang = sample_dir(1:input_data.n_train);
    
    test_data = mean_sample_vel(input_data.n_train+1:end,:);
    test_class = sample_class(input_data.n_train+1:end);
    test_idx_start = idx_start(sample_idx(input_data.n_train+1:end),:,:);
    test_ang = sample_dir(input_data.n_train+1:end);
    
    % package outputs
    output_data.train_data_in = train_data;
    output_data.train_data_out = train_class;
    output_data.train_idx_start = train_idx_start;
    output_data.train_ang = train_ang;
    
    output_data.test_data_in = test_data;
    output_data.test_data_out = test_class;
    output_data.test_idx_start = test_idx_start;
    output_data.test_ang = test_ang;
end