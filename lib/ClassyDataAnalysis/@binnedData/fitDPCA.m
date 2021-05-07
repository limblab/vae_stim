function fitDPCA(binned,windowClasses)
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
%           puts data into dpcaData, and sets ranDPCAFit event

    
    if size(windowClasses,1)~=size(binned.dimReductionConfig.windows,1)
        error('fitDPCA:badWindowClasses','windowClasses must have 1 entry per window in binnedData.dimReductionConfig.windows')
    end

    %get data into a local var:
    data=binned.data{windows2mask(binned.data.t,binned.dimReductionConfig.windows),binned.dimReductionConfig.which};
    for i=1:size(binned.dimReductionConfig.windows,1)
        idxList=find(binned.data.t>=binned.dimReductionConfig.windows(i,1) & binned.data.t <=binned.dimReductionConfig.windows(i,2));
        tmpData=binned.data{idxList,binned.dimReductionConfig.which};
        if binned.dimReductionConfig.useTrialTime
            t=binned.data.t(idxList)-binned.data.t(idxList(1));
        else
            t=zeros(numel(idxList,1));
        end
        if binned.dimReductionConfig.rootTransform
            data{i}=[t sqrt(tmpData)];
        else
            data{i}=[t tmpData];
        end
        if ~exist('timePts','var')
            timePts=numel(idxList);
        else
            timePts=min([numel(idxList),timePts]);
        end
    end
   
    %convert data&windowClasses&time into a multidimensional structure as
    %expected by the dpca function:
    %data needs to be an N d matrix where:
    %D1 is the numer of neurons
    %D2:N-2: is number of stimuli for a single stimulus type(e.g. targets or active/passive/delay or task)
    %DN-1 is number of time points
    %DN is number of trial repetitions
    numClasses=size(windowClasses,2);
    classSizes=nan(numClasses,1);
    for i=1:numClasses
        classList{i}=unique(windowClasses{:,i});
        classSizes(i)=numel(classList{i});
    end
    %           (num units      ,classSizes,timePts, num trials)
    dpcaData=nan([size(data,2)-1,classSizes,timePts,size(windowClasses,1)]);
    for i=1:size(windowClasses,1)
        classIdxVec=nan(numClasses,1);
        for j=1:numClasses
            classIdxVec(j)=find(classList{j}==windowClasses{i,j},1,'first');
        end
        for j=1:size(data,2)-1
            dpcaData(j,classIdxVec,:,i)=reshape(data{i}(1:timePts,j+1),[ones(1,numel(classIdxVec)),timePts,1]);
        end
    end
    %define parameters including interactions:
    %note: windowClasses does not include the time. Number of trials is 
    %rows of window classes, and number of classes are columns of 
    %windowClasses
    paramList={};
    dpcaData.marginalizationNameList={};
    timeDim=size(windowClasses,2)+2;
    classNames=windowClasses.properties.VariableNames;
    if size(windowClasses,2)>2
        warning('fitDPCA:ignoring2ndOrderInteractions','fitDPCA only checks the first order interactions. If you want to include higher order interactions you will need to update the method')
    end
    for i=1:size(windowClasses,2)
        paramList{end+1}={i,[i timeDim]};
        dpcaData.marginalizationNameList{end+1}=classNames{i};
        for j=i+1:size(windowClasses,2)-1
            %add the first order interactions:
            paramList{end+1}={[i j],[i j timeDim]};
            dpcaData.marginalizationNameList{end+1}=[classNames{i},'/',classNames{j},' interaction'];
        end
    end
    dpcaData.paramList=paramList;
    %get trial averaged data:
    avgData=nanmean(data,ndims(data));
    dpcaData.avgData=avgData;
    %dpca without regularization:
    [dpcaData.W,dpcaData.V,dpcaData.whichMarg]=dpca(avgData,binned.dimReductionConfig.dimension,binned.dimReductionConfig.dimension,'combinedParams',paramList);
    dpcaData.varExplained=dpca_explainedVariance(avgData,dpcaData.W,dpcaData.V,'combinedParams',paramList);

    %get the latent variables:
    %start by reshaping the average data into a 2d matrix with all
    %conditions concatenated:
    avgConcatenatedData=avgData(:,:)';
    %mean subtract:
    avgCenteredData=bsxfun(@minus,avgConcatenatedData,nanmean(avgConcatenatedData));
    %project centered average data onto latent space:
    avgDataDimensionality=siz(avgData);
    dpcaData.latentAvg=reshape(avgCenteredData*dpcaData.W,[size(dpcaData.W,2),avgDataDimensionality(2:end)]);
    %project single trial data onto latent space:
    concatenatedData=data(:,:)';
    centeredData=bsxfun(@minus,concatenatedData,nanmean(concatenatedData));
    dataDimensionality=size(data);
    projData=centeredData*dpcaData.W;
    dpcaData.latentSingle=reshape(projData',[size(projData,2),dataDimensionality(2:end)]);
        
    set(binned,'ppcaData', ppcaData);
    opData = binned.dimReductionConfig;
    evntData=loggingListenerEventData('fitPPCA',opData);
    notify(binned,'ranPPCAFit',evntData)


end