function getUNTTaskTable(cds,times)
    %this is a method of the commonDataStructure class and must be saved in
    %the @commonDataStructure folder with the other method definitions
    %
    %produces a trials table with the following information:
    %startTime      -time of trial start
    %ctrOnTime      -time the center target appeared
    %tgtOnTime      -time the ourter target appeared
    %goCueTime      -time of the go cue
    %endTime        -time the trial ended
    %result         -result of the trial (RAFI)
    %tgtDir         -direction of the true target in degrees
    %cuePrior       -kappe for the prior of the cue
    %tgtPrior       -kappa for the prior of the actual targets
    

    centerOnTime    = cds.words.ts(cds.words.word==hex2dec('30'));
    otOnTime        = cds.words.ts(cds.words.word==hex2dec('40'));
    goCueTime       = cds.words.ts(cds.words.word==hex2dec('31'));
    %preallocate vectors:
    numTrials=numel(times.number);
    tgtDirList=nan(numTrials,1);
    tgtKappaList=nan(numTrials,1);
    cueKappaList=nan(numTrials,1);
    cueSliceNum=nan(numTrials,1);
    cueSliceLocs=cell(numTrials,1);
    goTime=nan(numTrials,1);
    OTTime=nan(numTrials,1);
    ctrOnTime=nan(numTrials,1);
    % For each trial complete code
    for trial=1:numTrials
        %find the databurst for this trial
        idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
        %get target and prior info from databurst
        if ~isempty(idxDB) 
            tgtDirList(trial) = mod(180*bytes2float(cds.databursts.db(idxDB,10:13))/pi,360);
            tgtKappaList(trial) = bytes2float(cds.databursts.db(idxDB,14:17));
            cueKappaList(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
            cueSliceNum(trial) = bytes2float(cds.databursts.db(idxDB,22:25));
            cueSliceLocs{trial} = mod(tgtDirList(trial)+180*bytes2float(cds.databursts.db(idxDB,26:65))'/pi,360);
            if cueKappaList(trial) > 100000 
                cueKappaList(trial) = NaN;
            end
        else
            tgtDirList(trial) = NaN;
            tgtKappaList(trial) = NaN;
            cueKappaList(trial) = NaN;
        end
        % get the timestamp for the go cue
        gT = goCueTime(find(goCueTime<times.endTime(trial) & goCueTime>times.startTime(trial),1,'first'));
        if isempty(gT)
            goTime(trial)=NaN;
        else
            goTime(trial)=gT;
        end
        % get the timestamp for the outer target appearance 
        OTT = otOnTime(find(otOnTime<times.endTime(trial) & otOnTime>times.startTime(trial),1,'first'));
        if isempty(OTT)
            OTTime(trial)=NaN;
        else
            OTTime(trial)=OTT;
        end
        %get the timestamp for center target appearance:
        cOT = centerOnTime(find(centerOnTime<times.endTime(trial) & centerOnTime>times.startTime(trial),1,'first'));
        if isempty(cOT)
            ctrOnTime(trial)=NaN;
        else
            ctrOnTime(trial)=cOT;
        end
    end

    for trial = 1:length(cueSliceNum)
        if (cueSliceNum(trial)>0 && cueSliceNum(trial)<=20)
            cueSliceLocs{trial} = cueSliceLocs{trial}(1:cueSliceNum(trial));
        end
    end

    % Deal with weird prior databursts
    checkPrior = @(burst) burst < (.99*10e-5) | burst > (1e5+1) | isnan(burst);
    badBursts = find(checkPrior(tgtKappaList));
    goodBursts = find(~checkPrior(tgtKappaList));
    for i = 1:length(badBursts)
        bb = badBursts(i);
        ind_dists = abs(goodBursts - bb);
        replacer_ind = goodBursts(find(ind_dists==min(ind_dists),1,'first'));
        replacer = tgtKappaList(replacer_ind);
        tgtKappaList(bb)= replacer;
    end

    trialsTable=table(roundTime(ctrOnTime,.001),roundTime(OTTime,.001),roundTime(goTime,.001),...
                        tgtDirList,cueSliceLocs,cueKappaList,tgtKappaList,...
                        'VariableNames',{'ctrOnTime','tgtOnTime','goCueTime','tgtDir','cueDir','cueKappa','tgtKappa'});
    trialsTable.Properties.VariableUnits={'s','s','s','Deg','Deg','AU','AU'};
    trialsTable.Properties.VariableDescriptions={'center target onset time','outer target onset time','go cue time',...
                                                    'actual target direction','cue directions','kappa of the von mises function for the visual cue',...
                                                    'kappa for the von mises function for the actual target locations'};
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the UNT task';
    set(cds,'trials',trialsTable)
    evntData=loggingListenerEventData('getUNTTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end