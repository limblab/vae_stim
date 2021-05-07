function getBDTaskTable(cds,times)
    % THIS IS FOR THE 2AFC task
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %computes the trial variables for the 2AFC task and composes the trial
    %table in the cds using the task variables and the generic trial times
    %passed in from the calling function. This is intended to be called by 
    %the getTrialTable method of the cds class, rather than directly by a
    %user
    
    %get our word timing for changes in the state machine:
    % Isolate the individual word timestamps
    bumpWordBase = hex2dec('50');
    bumpMask=cds.words.word >= (bumpWordBase) & cds.words.word <= (bumpWordBase+5);
    bumpTimes = cds.words.ts(bumpMask)';
    bumpCodes = cds.words.word(bumpMask)';

    wordOTOn = hex2dec('40');
    otMask=bitand(hex2dec('f0'),cds.words.word) == wordOTOn;
    otOnTimes = cds.words.ts( otMask);
    otOnCodes = cds.words.word( otMask);
    
    wordGo = hex2dec('31');
    goCueTime = cds.words.ts(cds.words.word == wordGo);
    
    wordStim=hex2dec('60');
    stimMask=bitand(hex2dec('f0'),cds.words.word) == wordStim;
    stimTimes=cds.words.ts( stimMask );
    stimCodeList=cds.words.word( stimMask );
    
    %preallocate our trial variables:
    numTrials=numel(times.number);

    
    bumpAngle = nan(numTrials,1);
    
    tgtOnTime=nan(numTrials,1);
    bumpTimeList=nan(numTrials,1);
    goCueList=nan(numTrials,1);
    ctrHold=nan(numTrials,1);
    delayHold = nan(numTrials,1);
    moveTime = nan(numTrials,1);
    
    bumpDelay = nan(numTrials,1);
    bumpHold = nan(numTrials,1);
    intertrialTime = nan(numTrials,1);
    penaltyTime = nan(numTrials,1);
    tgtSize = nan(numTrials,1);
    bigTgtSize = nan(numTrials,1);
    tgtRadius = nan(numTrials,1);
    tgtAngle = nan(numTrials,1);
    
    hideCursor = nan(numTrials,1);
    abortDuringBump = nan(numTrials,1);
    cue1BumpPeak = nan(numTrials,1);
    cue1BumpRise = nan(numTrials,1);
    cue1BumpMag = nan(numTrials,1);
    cue1BumpDir = nan(numTrials,1);
    
    cue2BumpPeak = nan(numTrials,1);
    cue2BumpRise = nan(numTrials,1);
    cue2BumpMag = nan(numTrials,1);
    cue2BumpDir = nan(numTrials,1);
    
    cue1IsStim = nan(numTrials,1);
    cue2IsStim = nan(numTrials,1);
    cue1StimCode = nan(numTrials,1);
    cue2StimCode = nan(numTrials,1);
    isSameCue = nan(numTrials,1);
    useCue1 = nan(numTrials,1);
    cue1First = nan(numTrials,1);
    otHold = nan(numTrials,1);
    periodDuration = nan(numTrials,1);
    interperiodDuration = nan(numTrials,1);
    redoTrial = nan(numTrials,1);
    sameTargetRight = nan(numTrials,1);
    isTrainingTrial = nan(numTrials,1);
    
    
    %get the databurst version:
    dbVersion=cds.databursts.db(1,2);
    skipList=[];
    
    switch dbVersion
        case 0
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
        case 1    
            % loop thorugh our trials and build our list vectors:
            
            for trial = 1:numTrials
                %find and parse the current databurst:
                idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
                if isempty(idxDB)
                    skipList=[skipList,trial];
                    continue
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
                %from mastercon code to ensure matching when extracting data from
                %databurst:
               % * Version 1 (0x01)
                    %  * ----------------
                    %  * byte  0:		uchar		=> number of bytes to be transmitted
                    %  * byte  1:		uchar		=> version number (in this case 0)
                    %  * byte  2-4:	uchar		=> task code 'A' 'F' 'C'
                    %  * bytes 5-6:	uchar       => version code
                    %  * byte  7-8:	uchar		=> version code (micro)
                    %  * byte 9-12: float       => center hold time
                    %  * byte 13-16: float      => delay hold time
                    %  * byte 17-20: float      => movement time
                    %  * byte 21-24: float      => bump delay time
                    %  * byte 25-28: float      => bump hold time
                    %  * byte 29-32: float      => intertrial time
                    %  * byte 33-36: float      => penalty time
                    %  * byte 37-40: float      => target size
                    %  * byte 41-44: float      => big target size
                    %  * byte 45-48: float      => target distance
                    %  * byte 49-52: float      => target angle
                    %  * byte 53: uchar         => hide cursor
                    %  * byte 54: uchar         => abort during bump
                    %  * byte 55-58: float      => cue 1 bump hold duration
                    %  * byte 59-62: float      => cue 1 bump rise time
                    %  * byte 63-66: float      => cue 1 bump peak magnitude
                    %  * byte 67-70: float      => cue 1 bump direction
                    %  * byte 71-74: float      => cue 2 bump hold duration
                    %  * byte 75-78: float      => cue 2 bump rise time
                    %  * byte 79-82: float      => cue 2 bump peak magnitude
                    %  * byte 83-86: float      => cue 2 bump direction
                    %  * byte 87: uchar         => cue 1 is stim
                    %  * byte 88: uchar         => cue 2 is stim
                    %  * byte 89-92: float      => cue 1 stim code
                    %  * byte 93-96: float      => cue 2 stim code
                    %  * byte 97: uchar         => is same cue
                    %  * byte 98: uchar         => use cue 1
                    %  * byte 99: uchar         => cue 1 first
                    %  * byte 100-103: float    => outer target hold
                    %  * byte 104-107: float    => period duration
                    %  * byte 108-111: float    => interperiod duration
                    %  * byte 112: uchar        => redo trial
                    %  * byte 113: uchar        => same target right 
                    %  * byte 114: uchar        => training trial  
                    %  */
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                delayHold(trial) = bytes2float(cds.databursts.db(idxDB,14:17));
                moveTime(trial) = bytes2float(cds.databursts.db(idxDB,18:21));
                
                bumpDelay(trial) = bytes2float(cds.databursts.db(idxDB,22:25));
                bumpHold(trial) = bytes2float(cds.databursts.db(idxDB,26:29));
                
                intertrialTime(trial)=bytes2float(cds.databursts.db(idxDB,30:33));
                penaltyTime(trial)=bytes2float(cds.databursts.db(idxDB,34:37));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,38:41));
                bigTgtSize(trial) = bytes2float(cds.databursts.db(idxDB,42:45));
                tgtRadius(trial) = bytes2float(cds.databursts.db(idxDB,46:49));
                tgtAngle(trial) = bytes2float(cds.databursts.db(idxDB,50:53));
                
                hideCursor(trial) = cds.databursts.db(idxDB,54);
                abortDuringBump(trial) = cds.databursts.db(idxDB,55);
                cue1BumpPeak(trial) = bytes2float(cds.databursts.db(idxDB,56:59));
                cue1BumpRise(trial) = bytes2float(cds.databursts.db(idxDB,60:63));
                cue1BumpMag(trial) = bytes2float(cds.databursts.db(idxDB,64:67));  
                cue1BumpDir(trial) = bytes2float(cds.databursts.db(idxDB,68:71));
                
                cue2BumpPeak(trial) = bytes2float(cds.databursts.db(idxDB,72:75));
                cue2BumpRise(trial) = bytes2float(cds.databursts.db(idxDB,76:79));
                cue2BumpMag(trial) = bytes2float(cds.databursts.db(idxDB,80:83));  
                cue2BumpDir(trial) = bytes2float(cds.databursts.db(idxDB,84:87));
                
                cue1IsStim(trial) = cds.databursts.db(idxDB,88);
                cue2IsStim(trial) = cds.databursts.db(idxDB,89);
                
                cue1StimCode(trial) = bytes2float(cds.databursts.db(idxDB,90:93));
                cue2StimCode(trial) = bytes2float(cds.databursts.db(idxDB,94:97));
                isSameCue(trial) = cds.databursts.db(idxDB,98);
                useCue1(trial) = cds.databursts.db(idxDB,99);
                cue1First(trial) = cds.databursts.db(idxDB,100);
                
                otHold(trial) = bytes2float(cds.databursts.db(idxDB,101:104));
                periodDuration(trial) = bytes2float(cds.databursts.db(idxDB,105:108));
                interperiodDuration(trial) = bytes2float(cds.databursts.db(idxDB,109:112));
                redoTrial(trial) = cds.databursts.db(idxDB,113);
                sameTargetRight(trial) = cds.databursts.db(idxDB,114);
                isTrainingTrial(trial) = cds.databursts.db(idxDB,115);
                
                %now get things that rely only on words and word timing:
                idxOT=find(otOnTimes>times.startTime(trial) & otOnTimes < times.endTime(trial),1,'first');
                if isempty(idxOT)
                    tgtOnTime(trial)=nan;
                    %tgtID(trial)=nan; %target ID has no meaning in this version of the databurst
                else
                    tgtOnTime(trial)=otOnTimes(idxOT);
                    %tgtID(trial)=otOnCodes(idxOT); %target ID has no meaning in this version of the databurst
                end

                % Bump code and time
                idxBump = find(bumpTimes > times.startTime(trial) & bumpTimes < times.endTime(trial), 1, 'first');
                if isempty(idxBump)
                    bumpTimeList(trial) = nan;
                    %bumpList(trial) = nan;%bump ID has no meaning in this version of the databurst
                    bumpAngle(trial)=nan;
                else
                    bumpTimeList(trial) = bumpTimes(idxBump);
                    %bumpList(trial) = bitand(hex2dec('0f'),bumpCodes(idxBump));%bump ID has no meaning in this version of the databurst
                end

                % Go cue
                idxGo = find(goCueTime > times.startTime(trial) & goCueTime < times.endTime(trial), 1, 'first');
                if isempty(idxGo)
                    goCueList(trial) = nan;
                else
                    goCueList(trial) = goCueTime(idxGo);
                end

                %Stim code
                idx = find(stimTimes > times.startTime(trial) & stimTimes < times.endTime(trial),1,'first');
                if isempty(idx)
                    stimCode(trial) = nan;
                else
                    stimCode(trial) = bitand(hex2dec('0f'),stimCodeList(idx));%hex2dec('0f') is a bitwise mask for the trailing bit of the word
                end
            end

            %build table:
            trialsTable=table(ctrHold,tgtOnTime,goCueList,intertrialTime,penaltyTime,bumpDelay,bumpHold,...
                                tgtSize,bigTgtSize,tgtAngle,tgtRadius,hideCursor,abortDuringBump,cue1BumpPeak,cue1BumpRise,...
                                cue1BumpMag,cue1BumpDir,cue2BumpPeak,cue2BumpRise,cue2BumpMag,cue2BumpDir,...
                                cue1IsStim,cue2IsStim,cue1StimCode,cue2StimCode,isSameCue,useCue1,cue1First,otHold,...
                                periodDuration,interperiodDuration,redoTrial,sameTargetRight,isTrainingTrial,...
                                'VariableNames',{'ctrHold','tgtOnTime','goCueTime','intertrialPeriod','penaltyPeriod','bumpDelay','bumpHold'...
                                'tgtSize','bigTgtSize','tgtDir','tgtRadius','hideCursor','abortDuringBump','cue1BumpPeak','cue1BumpRise',...
                                'cue1BumpMag','cue1BumpDir','cue2BumpPeak','cue2BumpRise','cue2BumpMag','cue2BumpDir',...
                                'cue1IsStim','cue2IsStim','cue1StimCode','cue2StimCode','isSameCue','useCue1','cue1First','otHold',...
                                'periodDuration','interperiodDuration','redoTrial','sameTargetRight','isTrainingTrial'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','cm','deg','cm','bool','bool','s','s',...
                                                    'N','deg','s','s','N','deg',...
                                                    'bool','bool','int','int','bool','bool','bool','s',...
                                                    's','s','bool','bool','bool'};
%             trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','go cue time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
%                                                             'size of targets','angle of outer target','x-y position of outer target','target distance from center','were targets on during bump','min tgt angle','max tgt angle','only the correct target was shown','pct of trials that only show correct target',...
%                                                             'time of bump onset','would we abort during bumps','did we have a center hold bump',...
%                                                             'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
%                                                             'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
%                                                             'was there stimulation','how often did stim happen','code in the stim word',...
%                                                             'did the cursor recenter after bump','is the correct tgt the one in the tgt direction'};
%             
%       
        

        otherwise
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
    end
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the CObump task';
    %sanitize trial table by masking off corrupt databursts with nan's:
    mask= ( trialsTable.ctrHold<0           | trialsTable.ctrHold>10000 | ...
            trialsTable.intertrialPeriod<0  | trialsTable.intertrialPeriod>10000 |...
            trialsTable.penaltyPeriod<0     | trialsTable.penaltyPeriod>10000 |...
            trialsTable.tgtSize<.000001);
        
    mask(skipList)=1;
    idx=find(mask);
    for j=5:size(trialsTable,2)
        if ~isempty(find(strcmp({'goCueTime','tgtOnTime','bumpTime','tgtID','bumpID'},trialsTable.Properties.VariableNames{j}),1))
            %skip things that are based on the words, not the databurst
            continue
        end
        if islogical(trialsTable{1,j})
            trialsTable{idx,j}=false;
        else
            trialsTable{idx,j}=nan(size(trialsTable{1,j}));
        end
    end
    
    set(cds,'trials',trialsTable)
    evntData=loggingListenerEventData('getCOTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end

