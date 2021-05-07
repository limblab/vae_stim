function fitPCA(binned,varargin)
%function fitPCA
%       inputs:
%            binned data structure with pcaConfig filled with needed
%            parameters
%            rootTransform:  A boolean flag to root transform firing rate
%               data so that it is more normally distributed
%            useTrialTime:   A boolean flag to include an additional
%               feature vector that is the time within the window. This
%               will produce features correlated with time and may obscure
%               target effects
%            Optional: Key-value pair specifying additional operations
%            {'MachensFloor',classVector}:  flags binned.fitPCA to try and 
%               estimate the noise floor for the PCA data. classVector must
%               be a vector with length equal to the number of windows in
%               binned.dimReductionConfig. classVector should contain a
%               single class (e.g. perturbation number, target direction
%               etc.) for each window. The class MUST be an integer value
%               If this flag is used, the pcData structure will have an 
%               additional field 'MachensFloor'containing two vectors, the 
%               cumulative sum noise vector, and a flag vector indicating 
%               whether the eigenvalue for the PC's in pcData.latent 
%               exceeds the noise floor.
%       Outputs:
%           puts data into pcaData, and sets ranPCAFit event
%           
%       binned.dimReductionConfig Structure must have:
%
%           windows : start and end times of windows to perform
%           dimensionality reduction on (eg. [targetAppearsTrial1,
%           goCueTrial1;targetAppearsTrial2, goCueTrial2....]
%           
%           which : list of binned data column numbers to include in the 
%           dimensionality reduction
%
%           dimension: dimension of reduced dataset
    

    %parse varargin:
    if ~isempty(varargin)
        if mod(numel(varargin),2)~=0
            error('fitPCA:oddNumberInputs','optional inputs must be key-value pairs')
        end
        for i=1:2:numel(varargin)
            switch varargin{i}
                case 'MachensFloor'
                    doMachensFloor=true;
                    classList=varargin{i+1};
                case 'VAF'
                    doVAF=true;
                    VAFThreshold=varargin{i+1};
                otherwise
                    error('fitPCA:unrecognizedInputKey',['did not recognize the input key: ',varargin{i}])
            end
        end
    end
    if ~exist('doMachensFloor','var')
        doMachensFloor=false;
    end
    if ~exist('doVAF','var')
        doVAF=false;
    end
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
    
    [pcaData.coeff,pcaData.score,pcaData.latent,pcaData.tSquared,pcaData.explained,pcaData.mu]=pca(data);
    
    if doMachensFloor
        classes=unique(round(classList));
        for i=1:numel(classes)
            disp(['estimating noise for class# ',num2str(i)])
            classWindows{i}=binned.dimReductionConfig.windows(classList==classes(i),:);
            machensData=binned.data{windows2mask(binned.data.t,classWindows{i}),binned.dimReductionConfig.which};
            if binned.dimReductionConfig.rootTransform
                machensData=sqrt(machensData);
            end
            noiseFloor(i,:)=getNoiseEst(machensData,'nBoot',100);
        end
        noiseFloor=cumsum(max(noiseFloor,[],1))';
        pcaData.MachensFloor.noise=noiseFloor;
        pcaData.MachensFloor.goodPC=pcaData.latent>pcaData.MachensFloor.noise;
    end
    if doVAF
        pcaData.VAF.goodPC=cumsum(pcaData.explained)<=VAFThreshold;
    end
    set(binned,'pcaData', pcaData);
    opData = binned.dimReductionConfig;
    evntData=loggingListenerEventData('fitPCA',opData);
    notify(binned,'ranPCAFit',evntData)
end

