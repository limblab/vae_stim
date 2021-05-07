function getOORTaskTable(cds,times)
    %this is a method of the commonDataStructure class and must be saved in
    %the @commonDataStructure folder with the other method definitions
    %
    %produces a trials table with the following information:
    %     startTargOnTime   - start target onset time
    %     startTargHold     - start target hold time
    %     goCueTime         - go cue time
    %     endTargHoldTime   - outer target hold time
    %     tgtDir           - intended angle of movement
    %     forceDir          - intended angle of force

    startTargOnTime = cds.words.ts(cds.words.word==hex2dec('30')); %WORD_CT_ON
    startTargHold   = cds.words.ts(cds.words.word==hex2dec('A0')); %WORD_CENTER_TARGET_HOLD
    goCueTime       = cds.words.ts(cds.words.word==hex2dec('31')); %WORD_GO_CUE
    endTargHoldTime = cds.words.ts(cds.words.word==hex2dec('A1')); %WORD_OUTER_TARGET_HOLD
    
    timetrial = @(word_time,trialnum) word_time(find(word_time<times.endTime(trialnum) & word_time>times.startTime(trialnum),1,'first'));
    
    %preallocate vectors:
    numTrials=numel(times.number);    
    [startx,starty,endx,endy,forceDir] = deal(nan(numTrials,1));
    [CO_t,CH_t,GC_t,OH_t] = deal(nan(numTrials,1));
    % For each trial complete code
    for trial=1:numTrials
        %find the databurst for this trial
        idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
        %get target and prior info from databurst
        if ~isempty(idxDB) 
            startx(trial)   = bytes2float(cds.databursts.db(idxDB,10:13));
            starty(trial)   = bytes2float(cds.databursts.db(idxDB,14:17));
            endx(trial)     = bytes2float(cds.databursts.db(idxDB,18:21));
            endy(trial)     = bytes2float(cds.databursts.db(idxDB,22:25));
            forceDir(trial) = 180*bytes2float(cds.databursts.db(idxDB,26:29))/pi;
        end
        % get the timestamp for the start target On
        COT = timetrial(startTargOnTime,trial);
        if isempty(COT)
            CO_t(trial)=NaN;
        else
            CO_t(trial)=COT;
        end
        % get the timestamp for the start target Hold
        CHT = timetrial(startTargHold,trial);
        if isempty(CHT)
            CH_t(trial)=NaN;
        else
            CH_t(trial)=CHT;
        end
        % get the timestamp for the Go cue
        GCT = timetrial(goCueTime,trial);
        if isempty(GCT)
            GC_t(trial)=NaN;
        else
            GC_t(trial)=GCT;
        end
        % get the timestamp for the end target Hold
        OHT = timetrial(endTargHoldTime,trial);
        if isempty(OHT)
            OH_t(trial)=NaN;
        else
            OH_t(trial)=OHT;
        end
        
    end

    % calculate intended movement direction
    tgtDir = mod(atan2d(endy-starty,endx-startx),360);

    trialsTable=table(roundTime(CO_t,.001),roundTime(CH_t,.001),...
                      roundTime(GC_t,.001),roundTime(OH_t,.001),...
                      round(tgtDir),round(forceDir),...
                      'VariableNames',{'startTargOnTime','startTargHoldTime',...
                                       'goCueTime','endTargHoldTime',...
                                       'tgtDir','forceDir'});
    trialsTable.Properties.VariableUnits={'s','s','s','s','Deg','Deg'};
    trialsTable.Properties.VariableDescriptions={'start target onset time',...
                                                 'start target hold time',...
                                                 'go cue time',...
                                                 'end target hold time',...
                                                 'end target direction',...
                                                 'intended force direction'};
                                           
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the Out-out task';
    set(cds,'trials',trialsTable)
    evntData=loggingListenerEventData('getOORTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end
