function fitPPCA(binned)
%function fitPPCA
%       inputs:
%            binned data structure with pcaConfig filled with needed
%            parameters:
%            rootTransform:  A boolean flag to root transform firing rate
%               data so that it is more normally distributed
%            useTrialTime:   A boolean flag to include an additional
%               feature vector that is the time within the window. This
%               will produce features correlated with time and may obscure
%               target effects
%           windows : start and end times of windows to perform
%               dimensionality reduction on (eg. [targetAppearsTrial1,
%               goCueTrial1;targetAppearsTrial2, goCueTrial2....]
%           which : list of binned data column numbers to include in the 
%               dimensionality reduction
%           dimension: dimension of reduced dataset
%
%       Outputs:
%           puts data into ppcaData, and sets ranPPCAFit event
%           



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

    [ppcaData.coeff,ppcaData.score,ppcaData.latent,ppcaData.mu,ppcaData.istropicVariance,ppcaData.stats]=ppca(data,binned.dimReductionConfig.dimension);

    set(binned,'ppcaData', ppcaData);
    opData = binned.dimReductionConfig;
    evntData=loggingListenerEventData('fitPPCA',opData);
    notify(binned,'ranPPCAFit',evntData)
end

