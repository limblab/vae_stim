function [output_data] = trainMovementDirectionClassifier(input_data)
% trains a classifier to determine which direction a movement was in
% input_data contains:
%   move_len : length of movement in s
%   td : trial data containing data
%   in_sig : field of trial_data to use as 'movement' data
%   tgt_axis : In degrees, classify movements as this direction or 180 degrees away.
%   min_sig : minimum velocity for movements
%   max_sig : maximum velocity for movements
%   n_train : number of training movements
%   n_test : number of test movements



    % get training and testing data
    [data] = getMovementTrainingTestingData(input_data);
    
    
    % train classifier on training and testing data
    switch input_data.classifier_name
        case 'svm'
            class_mdl = fitcsvm(data.train_data_in, data.train_data_out);
        case 'bayes'
            class_mdl = fitcnb(data.train_data_in, data.train_data_out);
        case 'linear'
            class_mdl = fitclinear(data.train_data_in, data.train_data_out);
            % no model, just look at target axis and mean stim vel
        otherwise
            error('classifier name not recognized');
    end
    
    % test classifier
    [test_out,post_prob] = predict(class_mdl, data.test_data_in);
    test_acc = sum(test_out == data.test_data_out)/numel(test_out);
    
    
    % package outputs
    output_data.train_test_data=data;
    output_data.test_acc = test_acc;
    output_data.class_mdl = class_mdl;
    output_data.test_posterior = post_prob;

end