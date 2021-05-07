function fitFA(binned)
%function fitFA
%       inputs:
%            binned data structure with faConfig filled with needed
%            parameters
%       Outputs:
%           First n factors of data set specified in config file. Saved in
%           faData section of data structure
%       Config Structure:
%
%           windows : start and end times of windows to perform
%           dimensionality reduction on (eg. [targetAppearsTrial1,
%           goCueTrial1;targetAppearsTrial2, goCueTrial2....]
%           
%           which : list of binned data column numbers to include in the 
%           dimensionality reduction
%
%           dimension: dimension of reduced dataset

    data=binned.data{windows2mask(binned.data.t,binned.dimReductionConfig.windows),binned.dimReductionConfig.which};
    if binned.dimReductionConfig.rootTransform
        data=sqrt(data);
    end
    if binned.dimReductionConfig.useTrialTime
        times=nan(size(binned.data.t));
        for i=1:size(binned.dimReductionConfig.windows,1)
            idxList=find(binned.data.t>=binned.dimReductionConfig.windows(i,1) & binned.data.t <=binned.dimReductionConfig.windows(i,2));
            timeList=binned.data.t(idxList)-binned.data.t(idxList(1));
            times(idxList)=timeList;
        end
        data=[times(~isnan(times));data];
    end
    [faData.lambda,faData.PSI,faData.T,faData.stats,faData.F]=factoran(data,binned.dimReductionConfig.dimension);

    set(binned,'faData', faData);
    opData = binned.dimReductionConfig;
    evntData=loggingListenerEventData('fitFA',opData);
    notify(binned,'ranFAFit',evntData)
end

