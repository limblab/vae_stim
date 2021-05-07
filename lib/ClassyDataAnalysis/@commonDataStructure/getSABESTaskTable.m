function getSABESTaskTable(cds,times)
    %this is a method of the commonDataStructure class and must be saved in
    %the @commonDataStructure folder with the other method definitions
    %
    %produces a trials table with the following information:
    %startTime      -time of trial start
    %ctrOnTime      -time the center target appeared
    %tgtOnTime      -time the ourter target appeared
    %goCueTime      -time of the go cue
    %otHoldTime     -time that the outer hold starts
    %endTime        -time the trial ended
    %result         -result of the trial (RAFI)
    %tgtDir         -direction of the true target in degrees
    %cursShift      -cursor offset from hand
    %tgtShift       -visual shift

    centerOnTime    = cds.words.ts(cds.words.word==hex2dec('30'));
    otOnTime        = cds.words.ts(cds.words.word==hex2dec('40'));
    goCueTime       = cds.words.ts(cds.words.word==hex2dec('31'));
    otHoldTime      = cds.words.ts(cds.words.word==hex2dec('A1'));
    
    %preallocate vectors:
    numTrials=numel(times.number);
    tgtDirList=nan(numTrials,1);
    cursShiftList=nan(numTrials,1);
    tgtShiftList=nan(numTrials,1);
    holdTime=nan(numTrials,1);
    goTime=nan(numTrials,1);
    OTTime=nan(numTrials,1);
    ctrOnTime=nan(numTrials,1);
    % For each trial complete code
    for trial=1:numTrials
        %find the databurst for this trial
        idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
        %get target and prior info from databurst
        if ~isempty(idxDB) 
            tgtDirList(trial) = 180*bytes2float(cds.databursts.db(idxDB,10:13))/pi;
            cursShiftList(trial) = bytes2float(cds.databursts.db(idxDB,14:17));
            tgtShiftList(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
        else
            tgtDirList(trial) = NaN;
            cursShiftList(trial) = NaN;
            tgtShiftList(trial) = NaN;
        end
        % get the timestamp for the outer hold
        oHT = otHoldTime(find(otHoldTime<times.endTime(trial) & otHoldTime>times.startTime(trial),1,'first'));
        if isempty(oHT)
            holdTime(trial)=NaN;
        else
            holdTime(trial)=oHT;
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
        cOT = find(centerOnTime<times.endTime(trial) & centerOnTime>times.startTime(trial),1,'first');
        if isempty(cOT)
            ctrOnTime(trial)=NaN;
        else
            ctrOnTime(trial)=cOT;
        end
    end

    trialsTable=table(roundTime(ctrOnTime,.001),roundTime(OTTime,.001),roundTime(goTime,.001),...
                        roundTime(holdTime,.001),tgtDirList,cursShiftList,tgtShiftList,...
                        'VariableNames',{'ctrOnTime','OTTime','goTime','holdTime','tgtDir','cursShift','tgtShift'});
    trialsTable.Properties.VariableUnits={'s','s','s','s','Deg','cm','cm'};
    trialsTable.Properties.VariableDescriptions={'center target onset time','outer target onset time','go cue time',...
                                                 'outer target hold time','target angle','cursor shift','visual shift'};
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the center-out Sabes task';
    set(cds,'trials',trialsTable)
    evntData=loggingListenerEventData('getSABESTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end