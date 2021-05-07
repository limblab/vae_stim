function getTrialTable(cds,opts)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %makes a simple table with trial data that is common to any task. 
    %output is a dataset with the following columns:
    %trial_number   -allows later subtables to maintain trial order
    %start_time     -start of trial
    %go_time        -Time of first go cue, ignores multiple cues as you'd
    %                   get in RW
    %end_time       -end of trial
    %trial_result   -Numeric code: 0=Reward,1=Abort,2=Fail,3=Incomplete
    %
    %assumes that cds.words exists and is non-empty
    %
    %if there is a trial start word and no end word before the next trial
    %start, that trial start will be ignored. Also ignores the first 1sec of
    %data to avoid problems associated with the missing 1sec of kinematic
    %data
    
    if isempty(cds.words) || isempty(cds.databursts)
        %this file has no trial info
        cds.getGenericTaskTable()
        return
    end
    
    wordStart = hex2dec('10');
    
    
    startTime =  cds.words.ts( bitand(hex2dec('f0'),cds.words.word) == wordStart &  cds.words.ts>1.000);
    numTrials = length(startTime);

    wordEnd = hex2dec('20');
    endTime =  cds.words.ts( bitand(hex2dec('f0'), cds.words.word) == wordEnd);
    endCodes =  cds.words.word( bitand(hex2dec('f0'), cds.words.word) == wordEnd);
    
    % old CO task goes straight from bump state to pretrial with no end code
    if strcmpi(opts.task,'CO')
        bumpWordBase = hex2dec('50');
        bumpTimes = cds.words.ts(cds.words.word >= (bumpWordBase) & cds.words.word <= (bumpWordBase+14))';
        bumpCodes = cds.words.word(cds.words.word >= (bumpWordBase) & cds.words.word <= (bumpWordBase+14))';
    end
    
    %Check for and remove corrupted endCodes
    corrupt_idx = mod(endCodes,32) > 3;
    if any(corrupt_idx)
        warning('getTrialTable:corruptEndCode','Corrupt trial result codes found, removing trials')
        endTime = endTime(~corrupt_idx);
        endCodes = endCodes(~corrupt_idx);
    end
    
    %preallocate with -1
    stopTime=nan(size(startTime));
    trialResult=cell(size(stopTime));
    resultCodes='RAFI';
    for ind = 1:numTrials-1
        % Find the end of the trial
        if ind==numTrials
            next_trial_start = inf;
        else
            next_trial_start = startTime(ind+1);
        end
        trial_end_idx = find(endTime > startTime(ind) & endTime < next_trial_start, 1, 'first');
        
        if isempty(trial_end_idx)
            stopTime(ind) = nan;
            trialResult(ind) = {'-'};
        else
            stopTime(ind) = endTime(trial_end_idx);
            trialResult(ind) = {resultCodes(mod(endCodes(trial_end_idx),32)+1)}; %0 is reward, 1 is abort, 2 is fail, and 3 is incomplete (incomplete should never happen)
        end
        
        % CO task goes straight from bump state to pretrial with no end code
        if isnan(stopTime(ind)) && strcmpi(opts.task,'CO')
            % look for bumps
            trial_bump_idx = find(bumpTimes > startTime(ind) & bumpTimes < next_trial_start, 1, 'first');
            if ~isempty(trial_bump_idx)
                stopTime(ind) = bumpTimes(trial_bump_idx);
                trialResult(ind) = {'R'};
            end
        end
    end
    mask=~isnan(stopTime);
    if sum(mask)==0
        %didn't find any trials
        return
    elseif sum(mask)==1
        %matlab has an error if you try to generate a single row table when
        %one of the variables is a character. Handle this by making a cell
        %array and calling cell2table:
        times=cell2table({[1:sum(mask)]',roundTime(startTime(mask),.001),roundTime(stopTime(mask),.001),char(trialResult(mask))},'VariableNames',{'number','startTime','endTime','result'});
    else
        times=table([1:sum(mask)]',roundTime(startTime(mask),.001),roundTime(stopTime(mask),.001),char(trialResult(mask)),'VariableNames',{'number','startTime','endTime','result'});
    end
    
    
    %specific task table code will add operations, so add the operation
    %for this file here, before we run the task specific code:
    if ~strcmpi(opts.task,'Unknown')
        %try to get trial data specific to the task
        switch opts.task
            case 'RW' %Labs standard random walk task for the robot
                try
                    cds.getRWTaskTable(times);
                end
            case 'CO' %labs standard center out task for the robot
                cds.getCOTaskTable(times);
            case 'CObump'
                cds.getCObumpTaskTable(times);
            case 'COactpas' % aliased with CObump
                cds.getCObumpTaskTable(times);
            case 'WF' %wrist flexion task
                cds.getWFTaskTable(times);
            case 'multi_gadget'
                cds.getMultiGadgetTaskTable(times);
            case 'ball_drop'
                cds.getBallDropTaskTable(times); 
            case 'BD' %Tucker's psychophysics bump direction task
%                 error('getTrialTable:taskNotImplemented','the code to create a trial table for the psychophysics task is not implemented. Please help by implementing it! ')
                cds.getBDTaskTable(times);
            case 'UNT' %Brian Dekleva's uncertainty task
                cds.getUNTTaskTable(times);
            case 'RP' %Ricardo's resist perturbations task
                error('getTrialTable:taskNotImplemented','the code to create a trial table for the resist perturbations task is not implemented. Please help by implementing it! ')
                
            case 'DCO' %Ricardo's dynamic center out task
                error('getTrialTable:taskNotImplemented','the code to create a trial table for the dynamic center out task is not implemented. Please help by implementing it! ')
                
            case 'SABES' % Brian Dekleva's center out sabes task
                cds.getSABESTaskTable(times);
            
            case 'UCK' % Brian Dekleva's 2-target Cisek task
                cds.getUCKTaskTable(times);
                
            case 'OOR' % Raeed's Out-out reach task
                cds.getOORTaskTable(times);

            case 'TRT' % Raeed's two workspace random target task
                cds.getTRTTaskTable(times);
            case 'RT' % reaction time task
                cds.getReactionTimeTaskTable(times);
            case 'RR'
                cds.getRingReportingTaskTable(times);
            case 'AFC'
                cds.get2AFCTaskTable(times);
            otherwise
                warning('getTrialTable:UnknownTask','The task for this data file was not set. Trial table will contain only trial start,stop and result')
                set(cds,'trials',times)
        end
    else
        %cds.setField('trials',times)
        set(cds,'trials',times)
    end
    evntData=loggingListenerEventData('getTrialTable',opts.task);
    notify(cds,'ranOperation',evntData)
end
