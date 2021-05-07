function fitGPFA(binned)
%function fitGPFA
%       inputs:
%            binned data structure with gpfaConfig filled with needed
%            parameters
%       Outputs:
%           First n factors of data set specified in config file. Saved in
%           gpfaData section of data structure
%       Config Structure:
%
%           windows : start and end times of windows to perform
%           dimensionality reduction on (eg. [targetAppearsTrial1,
%           goCueTrial1;targetAppearsTrial2, goCueTrial2....]
%           
%           dimension: dimension of reduced dataset
%
%           segLength: Lenght of smaller segments to cut trials into. Makes
%           computation of factors faster for FA and GPFA if trials are of
%           equal length
%
%           trials: trial numbers corresponding to windows
%
%           kernSD: smoothing kernel standard deviation. Larger value acts
%           as a larger low pass filter on trajectories

    dat = binned.dimRedHelper();
    kernSD = 10;%GPFA optimizes smoothing kernel so ignore the parameter in binned.dimReductionConfig.kernSD, and pass a dummy variable
    runIdx =102;
    xDim = binned.dimReductionConfig.dimension;
    result = neuralTraj(runIdx,dat, 'method', 'gpfa', 'xDim', xDim, 'kernSDList', kernSD, 'segLength', binned.dimReductionConfig.segLength);
    [result.estParamsPP, result.seqTrainPP] = postprocess(result, 'kernSD', kernSD);
    result.method = 'gpfa';
    gpfaData = result;
    set(binned,'gpfaData', gpfaData);
    opData = binned.dimReductionConfig;
    evntData=loggingListenerEventData('fitGpfa',opData);
    notify(binned,'ranGPFAFit',evntData)
end

