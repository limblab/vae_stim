function getReactionTimeTaskTable(cds,times)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %computes the trial variables for the CO task and composes the trial
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

    tgtSize=nan(numTrials,1);
    tgtRadius = nan(numTrials,1);
    tgtAngle=nan(numTrials,1);
    randomTargets = nan(numTrials,1);
    showTgtDuringBump = nan(numTrials,1);
    
    bumpTrial = nan(numTrials,1);
    bumpMagnitude=nan(numTrials,1);
    bumpAngle=nan(numTrials,1);
    bumpFloor = nan(numTrials,1);
    bumpCeiling = nan(numTrials,1);
    bumpStep = nan(numTrials,1);
    bumpRisePeriod=nan(numTrials,1);
    bumpDuration = nan(numTrials,1);
    
    stimTrial=false(numTrials,1);
    stimCode=nan(numTrials,1);
    
    isTrainingTrial=false(numTrials,1);
    
    recenterCursor=false(numTrials,1);
    hideCursor = nan(numTrials,1);
    
    intertrialPeriod=nan(numTrials,1);
    penaltyPeriod=nan(numTrials,1);
    bumpHold=nan(numTrials,1);
    ctrHold=nan(numTrials,1);
    bumpDelay=nan(numTrials,1);
    
    tgtOnTime = nan(numTrials,1);
    bumpTimeList = nan(numTrials,1);
    goCueList = nan(numTrials,1);
    
    abortDuringBump = nan(numTrials,1);
    forceReaction = nan(numTrials,1);
    
    bumpStaircaseIdx = nan(numTrials,1);
    bumpStaircaseValue = nan(numTrials,1);
    stimStaircaseIdx = nan(numTrials,1);
    
    isVisualTrial = nan(numTrials,1);
    
    %get the databurst version:
    dbVersion=cds.databursts.db(1,2);
    skipList=[];
    switch dbVersion
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
                %       * Version 1 (0x01)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case one)
                %  * byte  2-3:	uchar		=> task code ('FC')
                %  * bytes 4-5:	uchar       => version code
                %  * byte  6-7:	uchar		=> version code (micro)
                %  *
                %  * bytes 8-11:  float		=> target angle
                %  * byte  12:	uchar           => random target flag
                %  * bytes 13-16: float		=> target radius
                %  * bytes 17-20: float		=> target size
                %  * byte  21:	uchar		=> show target during bump
                %  *
                %  * byte  22:                => bump trial flag
                %  * bytes 23-26: float		=> bump direction
                %  * bytes 27-30: float       => bump magnitude
                %  * bytes 31-34: float		=> bump floor (minimum force(N) bump can take)
                %  * bytes 35-38:	float		=> bump ceiling (maximum force(N) bump can take)
                %  * bytes 39-42:	float		=> bump step
                %  * bytes 43-46: float		=> bump duration
                %  * bytes 47-50: float		=> bump ramp
                %  *
                %  * byte  51:	uchar		=> stim trial flag
                %  * bytes 52:    uchar       => stim code
                %  *
                %  * byte  53:    uchar       => training trial flag
                %  *
                %  * byte  54:	uchar		=> recenter cursor flag
                %  * byte  55:    uchar       => hide cursor during bump
                %  *
                %  * bytes 56-59: float		=> intertrial time
                %  * bytes 60-63: float		=> penalty time
                %  * bytes 64-67: float		=> bump hold time
                %  * bytes 68-71: float		=> center hold time
                %  * bytes 72-75: float		=> bump delay time
                %  * byte 76:	uchar		=> abort during bump
                %  * byte 77:	uchar		=> force reaction
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                randomTargets(trial)=cds.databursts.db(idxDB,14);
                tgtRadius(trial)=bytes2float(cds.databursts.db(idxDB,15:18));
                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                showTgtDuringBump(trial) = cds.databursts.db(idxDB,23);

                bumpTrial(trial) = cds.databursts.db(idxDB,24);
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,25:28));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,29:32));
                bumpFloor(trial) = bytes2float(cds.databursts.db(idxDB,33:36));
                bumpCeiling(trial) = bytes2float(cds.databursts.db(idxDB,37:40));
                bumpStep(trial) = bytes2float(cds.databursts.db(idxDB,41:44));
                bumpDuration(trial) = bytes2float(cds.databursts.db(idxDB,45:48));
                bumpRisePeriod(trial) = bytes2float(cds.databursts.db(idxDB,49:52));
                
                stimTrial(trial)= cds.databursts.db(idxDB,53);
                stimCode(trial) = cds.databursts.db(idxDB,54);
                
                isTrainingTrial(trial)=cds.databursts.db(idxDB,55);
                
                recenterCursor(trial)=cds.databursts.db(idxDB,56);
                hideCursor(trial)=cds.databursts.db(idxDB,57);

                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,58:61));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,70:73));
                
                abortDuringBump(trial) = cds.databursts.db(idxDB,78);
                forceReaction(trial) = cds.databursts.db(idxDB,79);
                
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
            trialsTable=table(bumpHold,tgtOnTime,goCueList,intertrialPeriod,penaltyPeriod,ctrHold,bumpDuration,...
                                tgtSize,tgtAngle,tgtRadius,...
                                isTrainingTrial,...
                                bumpTimeList,abortDuringBump,bumpDuration,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimCode,... 
                                recenterCursor,forceReaction,hideCursor,...
                                bumpStep,bumpCeiling,bumpFloor,...
                                'VariableNames',{'ctrHold','tgtOnTime','goCueTime','intertrialPeriod','penaltyPeriod',...
                                'bumpDelay','bumpHoldTime','tgtSize','tgtDir','tgtDistance',...
                                'isTrainingTrial',...
                                'bumpTime','abortDuringBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimCode',...
                                'recenterCursor','forceReaction','hideCursor',...
                                'bumpStep','bumpCeiling','bumpFloor'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm',...
                                                    'bool',...
                                                    's','bool','s','s','N','deg'...
                                                    'bool','int',...
                                                    'bool','bool','bool',...
                                                    'int','N','N'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','go cue time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
                                                            'size of targets','angle of outer target','distance to outer target from center',...
                                                            'only the correct target was shown',...
                                                            'time of bump onset','would we abort during bumps','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','code in the stim word',...
                                                            'did the cursor recenter after bump','did we force reaction time','did we hide the cursor',...
                                                            'step number of the bump staircase','staircase ceiling force','staircase floor force'};
            
        case 2
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
                %   * Version 2 (0x02) -- supports multiple staircases, more information is output
                %  *  so that the staircases can be tracked (which staircase was selected and where that
                %  *  staircase currently is)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case one)
                %  * byte  2-3:	uchar		=> task code ('FC')
                %  * bytes 4-5:	uchar       => version code
                %  * byte  6-7:	uchar		=> version code (micro)
                %  *
                %  * bytes 8-11:  float		=> target angle
                %  * byte  12:	uchar           => random target flag
                %  * bytes 13-16: float		=> target radius
                %  * bytes 17-20: float		=> target size
                %  * byte  21:	uchar		=> show target during bump
                %  *
                %  * byte  22:                => bump trial flag
                %  * bytes 23-26: float		=> bump direction
                %  * bytes 27-30: float       => bump magnitude
                %  * bytes 31-34: float		=> bump floor (minimum force(N) bump can take)
                %  * bytes 35-38:	float		=> bump ceiling (maximum force(N) bump can take)
                %  * bytes 39-42:	float		=> bump step
                %  * bytes 43-46: float		=> bump duration
                %  * bytes 47-50: float		=> bump ramp
                %  *
                %  * byte  51:	uchar		=> stim trial flag
                %  * bytes 52:    uchar       => stim code
                %  *
                %  * byte  53:    uchar       => training trial flag
                %  *
                %  * byte  54:	uchar		=> recenter cursor flag
                %  * byte  55:    uchar       => hide cursor during bump
                %  *
                %  * bytes 56-59: float		=> intertrial time
                %  * bytes 60-63: float		=> penalty time
                %  * bytes 64-67: float		=> bump hold time
                %  * bytes 68-71: float		=> center hold time
                %  * bytes 72-75: float		=> bump delay time
                %  * byte 76:	uchar		=> abort during bump
                %  * byte 77:	uchar		=> force reaction
                %  * bytes 78-81: float   => bump staircase idx
                %  * bytes 82-85: float   => current bump staircase value
                %  */
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                randomTargets(trial)=cds.databursts.db(idxDB,14);
                tgtRadius(trial)=bytes2float(cds.databursts.db(15:18));
                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                showTgtDuringBump(trial) = cds.databursts.db(idxDB,23);

                bumpTrial(trial) = cds.databursts.db(idxDB,24);
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,25:28));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,29:32));
                bumpFloor(trial) = bytes2float(cds.databursts.db(idxDB,33:36));
                bumpCeiling(trial) = bytes2float(cds.databursts.db(idxDB,37:40));
                bumpStep(trial) = bytes2float(cds.databursts.db(idxDB,41:44));
                bumpDuration(trial) = bytes2float(cds.databursts.db(idxDB,45:48));
                bumpRisePeriod(trial) = bytes2float(cds.databursts.db(idxDB,49:52));
                
                stimTrial(trial)= cds.databursts.db(idxDB,53);
                stimCode(trial) = cds.databursts.db(idxDB,54);
                
                isTrainingTrial(trial)=cds.databursts.db(idxDB,55);
                
                recenterCursor(trial)=cds.databursts.db(idxDB,56);
                hideCursor(trial)=cds.databursts.db(idxDB,57);

                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,58:61));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,70:73));
                
                abortDuringBump(trial) = cds.databursts.db(idxDB,78);
                forceReaction(trial) = cds.databursts.db(idxDB,79);
                
                bumpStaircaseIdx(trial) = bytes2float(cds.databursts.db(idxDB,80:83));
                bumpStaircaseValue(trial) = bytes2float(cds.databursts.db(idxDB,84:87));
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
            trialsTable=table(bumpHold,tgtOnTime,intertrialPeriod,penaltyPeriod,ctrHold,bumpDuration,...
                                tgtSize,tgtAngle,tgtRadius,...
                                isTrainingTrial,...
                                bumpTrial,bumpTimeList,abortDuringBump,bumpDuration,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimCode,... 
                                recenterCursor,forceReaction,hideCursor,...
                                bumpStep,bumpCeiling,bumpFloor,bumpStaircaseIdx,bumpStaircaseValue,...
                                'VariableNames',{'ctrHold','tgtOnTime','intertrialPeriod','penaltyPeriod',...
                                'bumpDelay','bumpHoldTime','tgtSize','tgtDir','tgtDistance',...
                                'isTrainingTrial',...
                                'isBumpTrial','bumpTime','abortDuringBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimCode',...
                                'recenterCursor','forceReaction','hideCursor',...
                                'bumpStep','bumpCeiling','bumpFloor','bumpStaircaseIdx','bumpStaircaseValue'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s',...
                                                    'cm','deg','cm',...
                                                    'bool',...
                                                    'bool','s','bool','s','s','N','deg'...
                                                    'bool','int',...
                                                    'bool','bool','bool',...
                                                    'int','N','N','int','int'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
                                                            'size of targets','angle of outer target','distance to outer target from center',...
                                                            'only the correct target was shown',...
                                                            'is a bump trial','time of bump onset','would we abort during bumps','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','code in the stim word',...
                                                            'did the cursor recenter after bump','did we force reaction time','did we hide the cursor',...
                                                            'step number of the bump staircase','staircase ceiling force','staircase floor force','index of staircase','current value of that staircase'};
          
         case 3
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
                %   * Version 3 (0x03) -- supports multiple stim staircases, more information is output
                %  *  so that the staircases can be tracked (which staircase was selected and where that
                %  *  staircase currently is)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case one)
                %  * byte  2-3:	uchar		=> task code ('FC')
                %  * bytes 4-5:	uchar       => version code
                %  * byte  6-7:	uchar		=> version code (micro)
                %  *
                %  * bytes 8-11:  float		=> target angle
                %  * byte  12:	uchar           => random target flag
                %  * bytes 13-16: float		=> target radius
                %  * bytes 17-20: float		=> target size
                %  * byte  21:	uchar		=> show target during bump
                %  *
                %  * byte  22:                => bump trial flag
                %  * bytes 23-26: float		=> bump direction
                %  * bytes 27-30: float       => bump magnitude
                %  * bytes 31-34: float		=> bump floor (minimum force(N) bump can take)
                %  * bytes 35-38:	float		=> bump ceiling (maximum force(N) bump can take)
                %  * bytes 39-42:	float		=> bump step
                %  * bytes 43-46: float		=> bump duration
                %  * bytes 47-50: float		=> bump ramp
                %  *
                %  * byte  51:	uchar		=> stim trial flag
                %  * bytes 52:    uchar       => stim code
                %  *
                %  * byte  53:    uchar       => training trial flag
                %  *
                %  * byte  54:	uchar		=> recenter cursor flag
                %  * byte  55:    uchar       => hide cursor during bump
                %  *
                %  * bytes 56-59: float		=> intertrial time
                %  * bytes 60-63: float		=> penalty time
                %  * bytes 64-67: float		=> bump hold time
                %  * bytes 68-71: float		=> center hold time
                %  * bytes 72-75: float		=> bump delay time
                %  * byte 76:	uchar		=> abort during bump
                %  * byte 77:	uchar		=> force reaction
                %  * bytes 78-81: float   => bump staircase idx
                %  * bytes 82-85: float   => current bump staircase value
                %  * bytes 86-89: float   => stim staircase idx
                %  */
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                randomTargets(trial)=cds.databursts.db(idxDB,14);
                tgtRadius(trial)=bytes2float(cds.databursts.db(15:18));
                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                showTgtDuringBump(trial) = cds.databursts.db(idxDB,23);

                bumpTrial(trial) = cds.databursts.db(idxDB,24);
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,25:28));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,29:32));
                bumpFloor(trial) = bytes2float(cds.databursts.db(idxDB,33:36));
                bumpCeiling(trial) = bytes2float(cds.databursts.db(idxDB,37:40));
                bumpStep(trial) = bytes2float(cds.databursts.db(idxDB,41:44));
                bumpDuration(trial) = bytes2float(cds.databursts.db(idxDB,45:48));
                bumpRisePeriod(trial) = bytes2float(cds.databursts.db(idxDB,49:52));
                
                stimTrial(trial)= cds.databursts.db(idxDB,53);
                stimCode(trial) = cds.databursts.db(idxDB,54);
                
                isTrainingTrial(trial)=cds.databursts.db(idxDB,55);
                
                recenterCursor(trial)=cds.databursts.db(idxDB,56);
                hideCursor(trial)=cds.databursts.db(idxDB,57);

                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,58:61));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,70:73));
                
                abortDuringBump(trial) = cds.databursts.db(idxDB,78);
                forceReaction(trial) = cds.databursts.db(idxDB,79);
                
                bumpStaircaseIdx(trial) = bytes2float(cds.databursts.db(idxDB,80:83));
                bumpStaircaseValue(trial) = bytes2float(cds.databursts.db(idxDB,84:87));
                stimStaircaseIdx(trial) = bytes2float(cds.databursts.db(idxDB,88:91));
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
            trialsTable=table(bumpHold,tgtOnTime,intertrialPeriod,penaltyPeriod,ctrHold,bumpDuration,...
                                tgtSize,tgtAngle,tgtRadius,...
                                isTrainingTrial,...
                                bumpTrial,bumpTimeList,abortDuringBump,bumpDuration,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimCode,... 
                                recenterCursor,forceReaction,hideCursor,...
                                bumpStep,bumpCeiling,bumpFloor,bumpStaircaseIdx,bumpStaircaseValue,stimStaircaseIdx,...
                                'VariableNames',{'ctrHold','tgtOnTime','intertrialPeriod','penaltyPeriod',...
                                'bumpDelay','bumpHoldTime','tgtSize','tgtDir','tgtDistance',...
                                'isTrainingTrial',...
                                'isBumpTrial','bumpTime','abortDuringBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimCode',...
                                'recenterCursor','forceReaction','hideCursor',...
                                'bumpStep','bumpCeiling','bumpFloor','bumpStaircaseIdx','bumpStaircaseValue','stimStaircaseIdx'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s',...
                                                    'cm','deg','cm',...
                                                    'bool',...
                                                    'bool','s','bool','s','s','N','deg'...
                                                    'bool','int',...
                                                    'bool','bool','bool',...
                                                    'int','N','N','int','int','int'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
                                                            'size of targets','angle of outer target','distance to outer target from center',...
                                                            'only the correct target was shown',...
                                                            'is a bump trial','time of bump onset','would we abort during bumps','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','code in the stim word',...
                                                            'did the cursor recenter after bump','did we force reaction time','did we hide the cursor',...
                                                            'step number of the bump staircase','staircase ceiling force','staircase floor force','index of bump staircase','current value of bump staircase','index of stim staircase'};
          
           case 4
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
%                  /* Version 4 (0x04) -- supports multiple stim staircases and bump staircases, more information is output
%                  *  so that the staircases can be tracked (which staircase was selected and where that
%                  *  staircase currently is)
%                  * ----------------
%                  * byte  0:		uchar		=> number of bytes to be transmitted
%                  * byte  1:		uchar		=> version number (in this case one)
%                  * byte  2:		uchar		=> task number
%                  * byte  3-4:	uchar		=> task code ('FC')
%                  * bytes 5-6:	uchar       => version code
%                  * byte  7-8:	uchar		=> version code (micro)
%                  *
%                  * bytes 9-12:  float		=> target angle
%                  * byte  13:	uchar           => random target flag
%                  * bytes 14-17: float		=> target radius
%                  * bytes 18-21: float		=> target size
%                  * byte  22:	uchar		=> show target during bump
%                  *
%                  * byte  23:                => bump trial flag
%                  * bytes 24-27: float		=> bump direction
%                  * bytes 28-31: float       => bump magnitude
%                  * bytes 32-35: float		=> bump floor (minimum force(N) bump can take)
%                  * bytes 36-39:	float		=> bump ceiling (maximum force(N) bump can take)
%                  * bytes 40-43:	float		=> bump step
%                  * bytes 44-47: float		=> bump duration
%                  * bytes 48-51: float		=> bump ramp
%                  *
%                  * byte  52:	uchar		=> stim trial flag
%                  * bytes 53:    uchar       => stim code
%                  *
%                  * byte  54:    uchar       => training trial flag
%                  *
%                  * byte  55:	uchar		=> recenter cursor flag
%                  * byte  56:    uchar       => hide cursor during bump
%                  *
%                  * bytes 57-60: float		=> intertrial time
%                  * bytes 61-64: float		=> penalty time
%                  * bytes 65-68: float		=> bump hold time
%                  * bytes 69-72: float		=> center hold time
%                  * bytes 73-76: float		=> bump delay time
%                  * byte 77:	uchar		=> abort during bump
%                  * byte 78:	uchar		=> force reaction
% 
%                  * bytes 79-82: float   => bump staircase idx
%                  * bytes 83-86: float   => current bump staircase value
% 
%                  * bytes 87-90: float	=> current stim staircase idx
%                  * byte 91: uchar       => visual trial
%                  * byte 92: uchar       => lock cursor during cue
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                randomTargets(trial)=cds.databursts.db(idxDB,14);
                tgtRadius(trial)=bytes2float(cds.databursts.db(15:18));
                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                showTgtDuringBump(trial) = cds.databursts.db(idxDB,23);

                bumpTrial(trial) = cds.databursts.db(idxDB,24);
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,25:28));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,29:32));
                bumpFloor(trial) = bytes2float(cds.databursts.db(idxDB,33:36));
                bumpCeiling(trial) = bytes2float(cds.databursts.db(idxDB,37:40));
                bumpStep(trial) = bytes2float(cds.databursts.db(idxDB,41:44));
                bumpDuration(trial) = bytes2float(cds.databursts.db(idxDB,45:48));
                bumpRisePeriod(trial) = bytes2float(cds.databursts.db(idxDB,49:52));
                
                stimTrial(trial)= cds.databursts.db(idxDB,53);
                stimCode(trial) = cds.databursts.db(idxDB,54);
                
                isTrainingTrial(trial)=cds.databursts.db(idxDB,55);
                
                recenterCursor(trial)=cds.databursts.db(idxDB,56);
                hideCursor(trial)=cds.databursts.db(idxDB,57);

                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,58:61));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,70:73));
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,74:77));
                
                abortDuringBump(trial) = cds.databursts.db(idxDB,78);
                forceReaction(trial) = cds.databursts.db(idxDB,79);
                
                bumpStaircaseIdx(trial) = bytes2float(cds.databursts.db(idxDB,80:83));
                bumpStaircaseValue(trial) = bytes2float(cds.databursts.db(idxDB,84:87));
                stimStaircaseIdx(trial) = bytes2float(cds.databursts.db(idxDB,88:91));
                
                isVisualTrial(trial) = cds.databursts.db(idxDB,92);
                
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
            trialsTable=table(bumpHold,tgtOnTime,goCueList,intertrialPeriod,penaltyPeriod,ctrHold,bumpDelay,...
                                tgtSize,tgtAngle,tgtRadius,...
                                isTrainingTrial,...
                                bumpTrial,bumpTimeList,abortDuringBump,bumpDuration,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimCode,... 
                                recenterCursor,forceReaction,hideCursor,...
                                bumpStep,bumpCeiling,bumpFloor,bumpStaircaseIdx,bumpStaircaseValue,stimStaircaseIdx,isVisualTrial,...
                                'VariableNames',{'bumpHold','tgtOnTime','goCueTime','intertrialPeriod','penaltyPeriod',...
                                'ctrHold','bumpHoldTime','tgtSize','tgtDir','tgtDistance',...
                                'isTrainingTrial',...
                                'isBumpTrial','bumpTime','abortDuringBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimCode',...
                                'recenterCursor','forceReaction','hideCursor',...
                                'bumpStep','bumpCeiling','bumpFloor','bumpStaircaseIdx','bumpStaircaseValue','stimStaircaseIdx','isVisualTrial'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm',...
                                                    'bool',...
                                                    'bool','s','bool','s','s','N','deg'...
                                                    'bool','int',...
                                                    'bool','bool','bool',...
                                                    'int','N','N','int','int','int','bool'};
            trialsTable.Properties.VariableDescriptions={'bump hold time','outer target onset time','state movement','intertrial time','penalty time','ctr tgt hold time','time after entering ctr tgt that cue happens',...
                                                            'size of targets','angle of outer target','distance to outer target from center',...
                                                            'only the correct target was shown',...
                                                            'is a bump trial','time of bump onset','would we abort during bumps','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','code in the stim word',...
                                                            'did the cursor recenter after bump','did we force reaction time','did we hide the cursor',...
                                                            'step number of the bump staircase','staircase ceiling force','staircase floor force','index of bump staircase','current value of bump staircase','index of stim staircase','is this a visual trial'};
          
                   
        otherwise
            error('getReactionTimeTaskTable:unrecognizedDBVersion',['the trial table code for RT is not implemented for databursts with version#:',num2str(dbVersion)])
    end
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the RT task';
    %sanitize trial table by masking off corrupt databursts with nan's:
    mask= ( trialsTable.ctrHold<0           | trialsTable.ctrHold>10000 | ...
            trialsTable.intertrialPeriod<0  | trialsTable.intertrialPeriod>10000 |...
            trialsTable.penaltyPeriod<0     | trialsTable.penaltyPeriod>10000 |...
            trialsTable.bumpHoldPeriod<0     | trialsTable.bumpHoldPeriod>10000 |...
            trialsTable.bumpRisePeriod<0     | trialsTable.bumpRisePeriod>10000 |...
            trialsTable.bumpMagnitude<-100     | trialsTable.bumpMagnitude>100 |...
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