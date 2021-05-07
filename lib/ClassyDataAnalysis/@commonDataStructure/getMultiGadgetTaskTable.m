function getMultiGadgetTaskTable(cds,times)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %cds.getMultiGadgetTaskTable(times)
    %getMultiGadgetTaskTable returns no value, instead it populates the trials field
    %of the cds assuming the task is a multi gadget task. Takes a single
    %input:times, which is a table with 4 columns: number, startTime,
    %endTime, and result. These times define the start and stop of trials
    %as indicated by the state words for trial start and trial end. the
    %result code will be a character 'R':reward 'A':abort 'F':fail
    %'I':incomplete.
    
    % Isolate the individual word timestamps
    wordTouchPad = hex2dec('30');
    wordGoCue = hex2dec('31');
    wordCatch = hex2dec('32');
    wordGadgetOn = hex2dec('40');
    wordReach = hex2dec('70');
    
    % touchPad times
    TouchPadWords = cds.words.ts(cds.words.word == wordTouchPad);
    
    %go cue times
    GoCueWords = cds.words.ts(cds.words.word == wordGoCue);
    
    % catch times
    CatchWords = cds.words.ts(cds.words.word == wordCatch);
    
    % get when the gadget turns on, and which gadget # it is
    GadgetOnWords = cds.words.ts(bitand(hex2dec('f0'),cds.words.word) == wordGadgetOn);
    GadgetCodes = cds.words.word(bitand(hex2dec('f0'),cds.words.word) == wordGadgetOn);
    
    % Reach time and what target we're going for
    ReachWords = cds.words.ts(bitand(hex2dec('f0'),cds.words.word) == wordReach);
    ReachCodes = cds.words.word(bitand(hex2dec('f0'),cds.words.word) == wordReach);
    
   
    % useful stuff -- number of trials, burst size, and NaN filled lists
    % for each of the important things we want
    burst_size = size(cds.databursts.db,2);
    numTrials = numel(times.number);
    
    TouchPadTimeList = nan(numTrials,1);
    GoCueTimeList = nan(numTrials,1);
    CatchFlag = nan(numTrials,1);
    GadgetTimeList = nan(numTrials,1);
    GadgetNumber = nan(numTrials,1);
    TargetCorners = nan(numTrials,4);
    TargetCenters = nan(numTrials,2);
    TargetDir = nan(numTrials,1);
    
    
    for trial = 1:numTrials
        
        % touchpad timing
        idxTP = find(TouchPadWords > times.startTime(trial) & TouchPadWords < times.endTime(trial),1,'first');
        if isempty(idxTP)
            TPTime = NaN;
        else
            TPTime = TouchPadWords(idxTP);
        end
        
        
        % targets
        idxTar = find(ReachWords > times.startTime(trial) & ReachWords < times.endTime(trial),1,'first');
        if isempty(idxTar)
            ReachTime = NaN;
            ReachDir = NaN;
        else
            ReachTime = ReachWords(idxTar);
            ReachDir = bitand(hex2dec('0f'), ReachCodes(idxTar));
        end
        
        % databursts
        dbidx = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
    
        % target location
        targetLoc = cds.databursts.db(dbidx, burst_size-15:end);
        targetLoc = bytes2float(targetLoc, 'little')';  
        if isempty(targetLoc)
            warning('databurst(%d) is corrupted, no target info',dbidx);
            targetLoc = [-1 -1 -1 -1];
        end
        
        % catch trials
        idxCatch = find(CatchWords > times.startTime(trial) & CatchWords < times.endTime(trial), 1, 'first');
        if isempty(idxCatch)
            isCatch = 0;
        else
            isCatch = 1;
        end
        
        % Go cue ts
        idxGo = find(GoCueWords > times.startTime(trial) & GoCueWords < times.endTime(trial), 1, 'first');
        if isempty(idxGo)
            if ~isCatch
                goCue = nan;
            else
                goCue = GoCueWords(idxCatch);
            end
        else
            goCue = GoCueWords(idxGo);
        end
        
        % gadget on time and which gadget - this obviously won't matter for
        % the ball drop task. We should probably figure out a way to keep
        % track of which task is located where.
        idxG = find(GadgetOnWords > times.startTime(trial) & GadgetOnWords < times.endTime(trial), 1, 'first');
        if isempty(idxG)
            if ~isCatch
                gadgetCue = nan;
                gNumber = nan;
            else
                gadgetCue = GadgetOnWords(idxCatch);
                gNumber = nan;
            end
        else
            gadgetCue = GadgetOnWords(idxG);
            gNumber = GadgetCodes(idxG);
        end
        
        
        % build up the arrays
        TouchPadTimeList(trial) = TPTime;
        GoCueTimeList(trial) = goCue;
        CatchFlag(trial) = isCatch;
        GadgetTimeList(trial) = gadgetCue;
        GadgetNumber(trial) = gNumber;
        TargetCorners(trial,:) = targetLoc;
        TargetCenters(trial,:)=[targetLoc(1)+targetLoc(3),targetLoc(2)+targetLoc(4)]/2; %center coordinates of outer target 
        TargetDir(trial)=atan2d(TargetCenters(trial,2),TargetCenters(trial,1)); %Direction (in degrees) of outer target

        
        
    end
    
    
    % compile everything into the trial table
    trialsTable = table(TouchPadTimeList,GoCueTimeList,CatchFlag,GadgetTimeList,...
        GadgetNumber,TargetCorners,TargetCenters,TargetDir,...
        'VariableNames',{'touchPadTime','goCueTime','catchFlag','gadgetOnTime','gadgetNumber',...
        'tgtCorners','tgtCenter','tgtDir'});
    
    trialsTable.Properties.VariableUnits = {'s','s','int','s','int','int','int','deg'};
    trialsTable.Properties.VariableDescriptions = {...
        'monkey touches touchpad time',...
        'go cue time',...
        'was this a catch trial? T/F',...
        'gadget on time',...
        'gadget location identification',...
        'x-y pairs for upper left and lower right target corners',...
        'x-y pairs for target center',...
        'target direction (in degrees)'};
    trialsTable = [times,trialsTable];
    trialsTable.Properties.Description = 'Trial table for the multi_gadget tasks';

    % add it to the cds
    set(cds,'trials',trialsTable)
    cds.addOperation('getMultiGadgetTaskTable',mfilename('fullpath'));


end
    
    
    
    
    
    