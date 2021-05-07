function fitGPFA(binned)
%fitGPFA
    %this is a method function of the binnedData class and should be saved
    %in the @binnedData folder with the class definition and other methods
    %files
    %
    %bd.fitGpfa uses the configuration in the bd.gpfaConfig field to compute
    %GPFA analysis for each unit and stores the result in the
    %bd.gpfa field
    
    %get our list of units
    if isempty(binned.gpfaConfig.units)
        %find all our units and make a cell array containing the whole list
        unitMask=~cellfun(@(x)isempty(strfind(x,'CH')),binned.bins.Properties.VariableNames) & ~cellfun(@(x)isempty(strfind(x,'ID')),binned.bins.Properties.VariableNames);
        uList=binned.bins.Properties.VariableNames(unitMask);
    else
        %use the list the user supplied
        uList=binned.gpfaConfig.units;
        if ~iscellstring(uList)
            error('fitGPFA:unitListNotCellString','the list of units in binnedData.gpfaConfig.units must be a cell array of strings, where each string is the name of a unit column in binnedData.bins')
        end
    end
    if isempty(binned.pdConfig.windows)
        rowMask=true(size(binned.bins.t));
    else
    end
    if binned.gpfaConfig.dimension == 0
        error('fitGPFA:dimensionalityNotSet','set the desired dimensionality in gpfaConfig')
    else
        xDim = gpfaConfig.dimension;
    end
    if binned.gpfaConfig.segLength ==0
        warning('Not setting segLength may increase the runtime of the function. Cut trials to same length for speed')
    end
    if isempty(binned.gpfaConfig.trialNums)
        trial
    else
        
    end
    segLength = binned.gpfaConfig.segLength;
    kernSD = 1;
    result = neuralTraj(runIdx, dat, 'method', 'gpfa', 'xDim', xDim, 'kernSDList', kernSD, 'segLength', segLength);
    [estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
    set(binned, 'gpfaData', seqTrain);
    % check the method and compute GPFA
   

end

