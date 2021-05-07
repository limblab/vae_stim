function [a_val,b_val] = getModelBasedActivationParameters(amp)
%             load(['D:\Joseph\stimModel' filesep 'ModelSpreadFits_noSyn_diam2_15_30_50_100uA_logisticFit']);
    % data from mat file listed above
    a_data = [0.00849,0.007527,0.006955,0.005851];
    b_data = [180.4, 291.9,389.6,537.1];
    amp_list = [15,30,50,100];

    % interpolate parameters for the given amplitude
    a_val = interp1(amp_list,a_data,amp,'linear','extrap');
    b_val = interp1(amp_list,b_data,amp,'linear','extrap');
end

