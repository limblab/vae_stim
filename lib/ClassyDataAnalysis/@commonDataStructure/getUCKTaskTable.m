function getUCKTaskTable(cds,times)
    %this is a method of the commonDataStructure class and must be saved in
    %the @commonDataStructure folder with the other method definitions
    %
    %produces a trials table with the following information:
    %     ctrTgtOnTime - center target onset time
    %     ctrHold      - center target hold time
    %     tgtOnTime    - outer target onset time
    %     memDelayTime - memory delay onset time
    %     targCueTime  - target cue onset time
    %     goCueTime    - go cue time
    %     tgtHoldTime  - outer target hold time
    %     tgtDir_cue1  - angle of target 1
    %     tgtDir_cue2  - angle of target 2
    %     cue1color    - color of target 1
    %     cue2color    - color of target 2
    %     tgtDir       - angle of correct target
    %     cue1rate     - rate of target 1
    %     cue2rate     - rate of target 2
    %     numTgt       - number of targets on screen (1 or 2)

    centerOnTime    = cds.words.ts(cds.words.word==hex2dec('30'));
    centerholdTime  = cds.words.ts(cds.words.word==hex2dec('A0')); %WORD_CENTER_TARGET_HOLD
    otOnTime        = cds.words.ts(cds.words.word==hex2dec('40'));
    memdelayTime    = cds.words.ts(cds.words.word==hex2dec('81')); %WORD_CT_MEM_DELAY
    targcueTime     = cds.words.ts(cds.words.word==hex2dec('82')); %WORD_CT_TARGCUE_ON
    goCueTime       = cds.words.ts(cds.words.word==hex2dec('31'));
    otHoldTime      = cds.words.ts(cds.words.word==hex2dec('A1'));
    
    timetrial = @(word_time,trialnum) word_time(find(word_time<times.endTime(trialnum) & word_time>times.startTime(trialnum),1,'first'));
    
    %preallocate vectors:
    numTrials=numel(times.number);    
    [cue1ang,cue2ang,color1,color2,trueang,rate1,rate2,numtargs] = deal(nan(numTrials,1));
    [CO_t,CH_t,OT_t,MD_t,TC_t,GC_t,OH_t] = deal(nan(numTrials,1));
    % For each trial complete code
    for trial=1:numTrials
        %find the databurst for this trial
        idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
        %get target and prior info from databurst
        if ~isempty(idxDB) 
            cue1ang(trial)  = 180*bytes2float(cds.databursts.db(idxDB,10:13))/pi;
            cue2ang(trial)  = 180*bytes2float(cds.databursts.db(idxDB,14:17))/pi;
            color1(trial)   = bytes2float(cds.databursts.db(idxDB,18:21));
            color2(trial)   = bytes2float(cds.databursts.db(idxDB,22:25));
            trueang(trial)  = 180*bytes2float(cds.databursts.db(idxDB,26:29))/pi;
            rate1(trial)    = bytes2float(cds.databursts.db(idxDB,30:33));
            rate2(trial)    = bytes2float(cds.databursts.db(idxDB,34:37));
            if size(cds.databursts.db,2)==37
                r1 = rate1(trial); r2 = rate2(trial);
                if (abs(r1) > 10 || abs(r2) > 10)&& trial ~= 1
                    r1 = rate1(trial-1); r2 = rate2(trial-1); 
                end
                co1 = abs(round(r1*10e6) - round(r1*10e1)*10e4)>1;
                co2 = abs(round(r2*10e6) - round(r2*10e1)*10e4)>1;
                numtargs(trial) = 1+(mean([co1 co2])==0);
            else
                numtargs(trial) = bytes2float(cds.databursts.db(idxDB,38:41));
            end
        end
        % get the timestamp for the center On
        COT = timetrial(centerOnTime,trial);
        if isempty(COT)
            CO_t(trial)=NaN;
        else
            CO_t(trial)=COT;
        end
        % get the timestamp for the center Hold
        CHT = timetrial(centerholdTime,trial);
        if isempty(CHT)
            CH_t(trial)=NaN;
        else
            CH_t(trial)=CHT;
        end
        % get the timestamp for the outer target On
        OTT = timetrial(otOnTime,trial);
        if isempty(OTT)
            OT_t(trial)=NaN;
        else
            OT_t(trial)=OTT;
        end
        % get the timestamp for the memory delay period
        MDT = timetrial(memdelayTime,trial);
        if isempty(MDT)
            MD_t(trial)=NaN;
        else
            MD_t(trial)=MDT;
        end
        % get the timestamp for the target Cue time
        TCT = timetrial(targcueTime,trial);
        if isempty(TCT)
            TC_t(trial)=NaN;
        else
            TC_t(trial)=TCT;
        end
        % get the timestamp for the Go cue
        GCT = timetrial(goCueTime,trial);
        if isempty(GCT)
            GC_t(trial)=NaN;
        else
            GC_t(trial)=GCT;
        end
        % get the timestamp for the outer Hold
        OHT = timetrial(otHoldTime,trial);
        if isempty(OHT)
            OH_t(trial)=NaN;
        else
            OH_t(trial)=OHT;
        end
        
    end

    trialsTable=table(roundTime(CO_t,.001),roundTime(CH_t,.001),roundTime(OT_t,.001),...
                      roundTime(MD_t,.001),roundTime(TC_t,.001),roundTime(GC_t,.001),...
                      roundTime(OH_t,.001),cue1ang,cue2ang,color1,color2,trueang,rate1,rate2,numtargs,...
                      'VariableNames',{'ctrTgtOnTime','ctrHold','tgtOnTime','memDelayTime','targCueTime',...
                                       'goCueTime','tgtHoldTime','tgtDir_cue1','tgtDir_cue2','cue1color','cue2color',...
                                       'tgtDir','cue1rate','cue2rate','numTgt'});
    trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s','Deg','Deg','color','color','Deg','none','none','none'};
    trialsTable.Properties.VariableDescriptions={'center target onset time',...
                                                 'center target hold time',...
                                                 'outer target(s) onset time',...
                                                 'memory delay onset time (targets disappear)',...
                                                 'target cue time (indication of correct target)',...
                                                 'go cue time',...
                                                 'outer target hold time',...
                                                 'target 1 angle',...
                                                 'target 2 angle',...
                                                 'target 1 color',...
                                                 'target 2 color',...
                                                 'correct target angle',...
                                                 'target 1 rate',...
                                                 'target 2 rate',...
                                                 'number of targets on trial'};
                                           
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the center-out Cisek task';
    set(cds,'trials',trialsTable)
    evntData=loggingListenerEventData('getUCKTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end