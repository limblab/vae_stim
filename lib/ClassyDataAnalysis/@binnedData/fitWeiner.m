function fitWeiner(binned)
    %this is a method of the binnedData class and must be stored in the
    %@binnedData folder
    %
    %fitWeiner fits a weiner filter to the data using the configuration in
    %the binnedData.weinerConfig property, and returns output data in the
    %binnedData.weinerData property. This method also notifys the
    %'ranWeinerFit' event so that listeners can take follow up actions,
    %like the experiment class copying the fit-data over to the
    %experiment.analysis structure.
    %
    %this comment section is a stub to explain what the config options do
    %
    
    %the following are legacy variables. In principle fitWeiner can be
    %updated to support these if desired.
    numLags=1;
    numSides=1;
    fs=1;
    %get full data we are working with:
    if isempty(binned.weinerConfig.windows)
        dataMask=true(size(binned.data,1),1);
    else
        dataMask=windows2mask(binned.data.t,binned.weinerConfig.windows);
    end
    if isempty(binned.weinerConfig.outputList)
        error('fitWeiner:noOutput','weinerConfig must have a list of outputs to fit data to. Currently binnedData.weinerConfig.outputList is empty')
    else
        outputList=binned.weinerConfig.outputList;
    end
    if isempty(binned.weinerConfig.inputList)
        %find all our units and make a cell array containing the whole list
        unitMask=~cellfun(@(x)isempty(strfind(x,'CH')),binned.data.Properties.VariableNames) & ~cellfun(@(x)isempty(strfind(x,'ID')),binned.data.Properties.VariableNames);
        inputList=binned.data.Properties.VariableNames(unitMask);
    else
        inputList=binned.weinerConfig.inputList;
    end
    outputMask=list2tableMask(binned.data,outputList);
    inputMask=list2tableMask(binned.data,inputList);
    inputData=binned.data(dataMask,inputMask);
    outputData=binned.data(dataMask,outputMask);
    %set up folds:
    numPts=floor(size(inputData,1)/binned.weinerConfig.numFolds);
    reserveMask=false(size(inputData,1),binned.weinerConfig.numFolds);
    for i=1:binned.weinerConfig.numFolds
        %get a mask for the data points that haven't been assigned a fold
        %yet:
        tempMask=~any(reserveMask,2);
        %generate a mask for a new fold from the unassigned points:
        subMask=false(sum(tempMask),1);
        temp=1:numel(subMask);
        subMask(datasample(temp,numPts))=true;
        %insert the new fold into the main mask array:
        reserveMask(tempMask,i)=subMask;
    end
    
    %loop through folds
    wData=struct('model',[],...
                    'poly',[],...
                    'R2',[],...
                    'VAF',[],...
                    'MSE',[],...
                    'foldMask',[],...
                    'reserveMask',[]);
    for i=1:binned.weinerConfig.numFolds
        disp(['working on fold ',num2str(i),' of ',num2str(binned.weinerConfig.numFolds)])
        %get weiner model from training data
        [wData(i).model.weights,wData(i).model.VAF,wData(i).model.MCC]=filMIMO4(inputData{~reserveMask(:,i),:},outputData{~reserveMask(:,i),:},numLags,numSides,fs);
        %get polynomial:
        
        
        %predict outputs in reserve data:
        [predictedData,~,truncatedOutputData]=predMIMO4(   inputData{reserveMask(:,i),:},...
                                                                wData(i).model.weights,...
                                                                numSides,fs,outputData{reserveMask(:,i),:});
        % 
        %compute R2, mse and vaf from fitted reserve data
        wData(i).R2=CalculateR2(truncatedOutputData,predictedData);
        wData(i).VAF=[1 - sum( (predictedData-truncatedOutputData).^2 ) ./ ...
                        sum( (truncatedOutputData - repmat(mean(truncatedOutputData),size(truncatedOutputData,1),1)).^2 )]';
        wData(i).MSE=[mean((predictedData-truncatedOutputData).^2)]';  
        %find the mask for the fold and the reserve on the original binned
        %data:
        idxList=sort(find(dataMask));
        foldMask=false(size(binned.data,1),1);
        foldMask(idxList(~reserveMask(i,:)))=true;
        wData(i).foldMask=foldMask;
        
        wData(i).reserveMask=false(size(binned.data,1),1);
        wData(i).reserveMask(idxList(reserveMask(i,:)))=true;
        wData(i).reserveMask=reserveMask;
    end
    
    %update the binnedData.weinderData field with our new fit
    set(binned,'weinerData',wData)
    %notify listeners that a new weiner fit was just made
    evntData=loggingListenerEventData('fitWeiner',binned.weinerConfig);
    notify(binned,'ranWeinerFit',evntData)
    
end

