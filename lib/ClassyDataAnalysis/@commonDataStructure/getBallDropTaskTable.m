function getBallDropTaskTable(cds,times)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %cds.getBallDropTaskTable(times)
    %getBallDropTaskTable returns no value, instead it populates the trials field
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
    wordPickup = hex2dec('90');
    
    % touchPad times
    TouchPadWords = cds.words.ts(cds.words.word == wordTouchPad);
    
    %go cue times
    GoCueWords = cds.words.ts(cds.words.word == wordGoCue);
    
    % catch times
    CatchWords = cds.words.ts(cds.words.word == wordCatch);
    
    % pickup -- for the ball drop task, friends
    PickupWords = cds.words.ts(cds.words.word == wordPickup);
    
    % useful stuff -- number of trials
    numTrials = numel(times.number);
    
    TouchPadTimeList = nan(numTrials,1);
    GoCueTimeList = nan(numTrials,1);
    CatchFlag = nan(numTrials,1);
    PickupTimeList = nan(numTrials,1);
    
    
    for trial = 1:numTrials
        
        % touchpad timing
        idxTP = find(TouchPadWords > times.startTime(trial) & TouchPadWords < times.endTime(trial),1,'first');
        if isempty(idxTP)
            TPTime = NaN;
        else
            TPTime = TouchPadWords(idxTP);
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
        
        
        % Pickup times - for the ball drop task
        idxPU = find(PickupWords > times.startTime(trial) & PickupWords < times.endTime(trial), 1, 'first');
        if isempty(idxPU)
            PUTime = nan;
        else
            PUTime = PickupWords(idxPU);
        end
        
        
        % build up the arrays
        TouchPadTimeList(trial) = TPTime;
        GoCueTimeList(trial) = goCue;
        CatchFlag(trial) = isCatch;
        PickupTimeList(trial) = PUTime;
        
        
    end
    
    
    % compile everything into the trial table
    trialsTable = table(TouchPadTimeList,GoCueTimeList,CatchFlag,PickupTimeList,...
        'VariableNames',{'touchPadTime','goCueTime','catchFlag','pickupTime'});
    
    trialsTable.Properties.VariableUnits = {'s','s','int','s'};
    trialsTable.Properties.VariableDescriptions = {...
        'monkey touches touchpad time',...
        'go cue time',...
        'was this a catch trial? T/F',...
        'pickup time (ball drop task)'};
    trialsTable = [times,trialsTable];
    trialsTable.Properties.Description = 'Trial table for the multi_gadget tasks';

    % add it to the cds
    set(cds,'trials',trialsTable)
    cds.addOperation('getBallDropTaskTable',mfilename('fullpath'));


end
    
    
    
    
    
    