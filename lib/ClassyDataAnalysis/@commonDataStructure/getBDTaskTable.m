function getBDTaskTable(cds,times)
    % THIS IS FOR THE BUMP DIRECTION TASK, NOT THE BALL DROP TASK. DESPITE
    % THE FACT THAT BALL DROP IS LABELED AS BD IN THE MASTERCON BEHAVIOR
    % CODE. SORRY, BUT BLAME SOMEONE ELSE FOR NAMING THIS.
    % 
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
    tgtOnTime=nan(numTrials,1);
    bumpTimeList=nan(numTrials,1);
    goCueList=nan(numTrials,1);
    ctrHold=nan(numTrials,1);

    bumpDelay=nan(numTrials,1);
    bumpHold=nan(numTrials,1);
    intertrialPeriod=nan(numTrials,1);
    penaltyPeriod=nan(numTrials,1);

    tgtSize=nan(numTrials,1);
    tgtAngle=nan(numTrials,1);
    tgtCtr=nan(numTrials,2);

    ctrHoldBump=false(numTrials,1);
    delayBump=false(numTrials,1);
    moveBump=false(numTrials,1);
    bumpHoldPeriod=nan(numTrials,1);
    bumpRisePeriod=nan(numTrials,1);
    bumpMagnitude=nan(numTrials,1);
    bumpAngle=nan(numTrials,1);
    tgtRadius=nan(numTrials,1);
    tgtDuringBump=false(numTrials,1);
    stimTrialFreq=nan(numTrials,1);
    isPrimaryTarget=false(numTrials,1);
    randomTargets=nan(numTrials,1);
    tgtDirFloor=nan(numTrials,1);
    tgtDirCeil=nan(numTrials,1);
    bumpDirFloor=nan(numTrials,1);
    bumpDirCeil=nan(numTrials,1);
    bumpDirStep=nan(numTrials,1);
    isTrainingTrial=false(numTrials,1);
    trainingTrialFreq=nan(numTrials,1);
    recenterCursor=false(numTrials,1);
    abortDuringBump=false(numTrials,1);
    stimTrial=false(numTrials,1);
    stimCode=nan(numTrials,1);
    
    numTargets = nan(numTrials,1);
    correctAngle = nan(numTrials,1);
    angleTolerance = nan(numTrials,1);
    %get the databurst version:
    
    numTargets = nan(numTrials,1);
    correctAngle = nan(numTrials,1);
    angleTolerance = nan(numTrials,1);
    dbVersion=cds.databursts.db(1,2);
    skipList=[];
    switch dbVersion
        case 0
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
        case 1
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
        case 2
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
        case 3
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
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
                % 2         db->addByte(DATABURST_VERSION);
                % 3         db->addByte('2');
                % 4         db->addByte('B');
                % 5         db->addByte('C');
                % 6         db->addByte(BEHAVIOR_VERSION_MAJOR);
                % 7         db->addByte(BEHAVIOR_VERSION_MINOR);
                % 8         db->addByte((BEHAVIOR_VERSION_MICRO & 0xFF00) >> 8);
                % 9         db->addByte(BEHAVIOR_VERSION_MICRO & 0x00FF);
                % 10:13 	db->addFloat((float)this->tgt_angle);
                % 14:17 	db->addFloat((float)this->bump_dir);
                % 18        db->addByte((byte)this->params->use_random_targets);
                % 19:22 	db->addFloat((float)this->params->target_floor);
                % 23:26 	db->addFloat((float)this->params->target_ceiling);
                % 27:30 	db->addFloat((float)this->bumpmag_local);
                % 31:34 	db->addFloat((float)this->params->bump_duration);
                % 35:38 	db->addFloat((float)this->params->bump_ramp);
                % 39:42 	db->addFloat((float)this->params->bump_floor);
                % 43:46 	db->addFloat((float)this->params->bump_ceiling);
                % 47        db->addByte((byte)this->stim_trial);
                % 48        db->addByte((byte)this->training_trial);
                % 49:52 	db->addFloat((float)this->params->training_frequency);
                % 53:56 	db->addFloat((float)this->params->stim_prob);
                % 57        db->addByte((byte)this->params->recenter_cursor);
                % 58:61 	db->addFloat((float)this->params->target_radius);
                % 62:65 	db->addFloat((float)this->params->target_size);
                % 66:69 	db->addFloat((float)this->params->intertrial_time);
                % 70:73 	db->addFloat((float)this->params->penalty_time);
                % 74:77 	db->addFloat((float)this->params->bump_hold_time);
                % 78:81 	db->addFloat((float)this->params->ct_hold_time);
                % 82:85 	db->addFloat((float)this->params->bump_delay_time);
                % 86        db->addByte((byte)this->params->show_target_during_bump);
                % 87:90 	db->addFloat((float)this->params->bump_incr);
                % 91        db->addByte((byte)this->is_primary_target);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,78:81));
                
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,82:85));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,74:77));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,70:73));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                tgtRadius(trial)=bytes2float(cds.databursts.db(58:61));
                tgtCtr(trial,:)=tgtRadius(trial)*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,31:34));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,35:38));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,27:30));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                tgtDuringBump(trial)=cds.databursts.db(36);
                ctrHoldBump(trial)=~tgtDuringBump(trial) && bumpMagnitude(trial)>0;
                delayBump(trial)=tgtDuringBump(trial) && bumpMagnitude(trial)>0;
                moveBump(trial)=false;
                
                stimTrial(trial)=cds.databursts.db(idxDB,47);
                stimTrialFreq(trial)=bytes2float(cds.databursts.db(idxDB,53:56));
                isPrimaryTarget(trial)=cds.databursts.db(idxDB,91);
                randomTargets(trial)=cds.databursts.db(idxDB,18);
                tgtDirFloor(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                tgtDirCeil(trial)=bytes2float(cds.databursts.db(idxDB,23:26));
                bumpDirFloor(trial)=bytes2float(cds.databursts.db(idxDB,39:42));
                bumpDirCeil(trial)=bytes2float(cds.databursts.db(idxDB,43:46));
                bumpDirStep(trial)=bytes2float(cds.databursts.db(idxDB,87:90));
                isTrainingTrial(trial)=cds.databursts.db(idxDB,48);
                trainingTrialFreq(trial)=bytes2float(cds.databursts.db(idxDB,49:52));
                recenterCursor(trial)=cds.databursts.db(idxDB,57);
                abortDuringBump(trial)=true;
                
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
            trialsTable=table(ctrHold,tgtOnTime,goCueList,intertrialPeriod,penaltyPeriod,bumpDelay,bumpHold,...
                                tgtSize,tgtAngle,round(tgtCtr,4),tgtRadius,tgtDuringBump,tgtDirFloor,tgtDirCeil,isTrainingTrial,trainingTrialFreq,...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimTrialFreq,stimCode,... 
                                recenterCursor,isPrimaryTarget,...
                                'VariableNames',{'ctrHold','tgtOnTime','goCueTime','intertrialPeriod','penaltyPeriod','bumpDelay','bumpHold'...
                                'tgtSize','tgtDir','tgtCtr','tgtRadius','tgtDuringBump','tgtDirFloor','tgtDirCeil','isTrainingTrial','trainingTrialFreq'...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimTrialFreq','stimCode',...
                                'recenterCursor','isPrimaryTgt'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm','cm','bool','deg','deg','bool','pct',...
                                                    's','bool','bool','bool','bool','s','s','N','deg',...
                                                    'bool','pct','int',...
                                                    'bool','bool'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','go cue time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
                                                            'size of targets','angle of outer target','x-y position of outer target','target distance from center','were targets on during bump','min tgt angle','max tgt angle','only the correct target was shown','pct of trials that only show correct target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                            'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','how often did stim happen','code in the stim word',...
                                                            'did the cursor recenter after bump','is the correct tgt the one in the tgt direction'};
            

        case 5
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
                % 2         db->addByte(DATABURST_VERSION);
                % 3         db->addByte('2');
                % 4         db->addByte('B');
                % 5         db->addByte('C');
                % 6         db->addByte(BEHAVIOR_VERSION_MAJOR);
                % 7         db->addByte(BEHAVIOR_VERSION_MINOR);
                % 8         db->addByte((BEHAVIOR_VERSION_MICRO & 0xFF00) >> 8);
                % 9         db->addByte(BEHAVIOR_VERSION_MICRO & 0x00FF);
                % 10:13 	db->addFloat((float)this->tgt_angle);
                % 14:17 	db->addFloat((float)this->bump_dir);
                % 18        db->addByte((byte)this->params->use_random_targets);
                % 19:22 	db->addFloat((float)this->params->target_floor);
                % 23:26 	db->addFloat((float)this->params->target_ceiling);
                % 27:30 	db->addFloat((float)this->bumpmag_local);
                % 31:34 	db->addFloat((float)this->params->bump_duration);
                % 35:38 	db->addFloat((float)this->params->bump_ramp);
                % 39:42 	db->addFloat((float)this->params->bump_floor);
                % 43:46 	db->addFloat((float)this->params->bump_ceiling);
                % 47        db->addByte((byte)this->stim_trial);
                % 48        db->addByte((byte)this->training_trial);
                % 49:52 	db->addFloat((float)this->params->training_frequency);
                % 53:56 	db->addFloat((float)this->params->stim_prob);
                % 57        db->addByte((byte)this->params->recenter_cursor);
                % 58:61 	db->addFloat((float)this->params->target_radius);
                % 62:65 	db->addFloat((float)this->params->target_size);
                % 66:69 	db->addFloat((float)this->params->intertrial_time);
                % 70:73 	db->addFloat((float)this->params->penalty_time);
                % 74:77 	db->addFloat((float)this->params->bump_hold_time);
                % 78:81 	db->addFloat((float)this->params->ct_hold_time);
                % 82:85 	db->addFloat((float)this->params->bump_delay_time);
                % 86        db->addByte((byte)this->params->show_target_during_bump);
                % 87:90 	db->addFloat((float)this->params->bump_incr);
                % 91        db->addByte((byte)this->is_primary_target);
                % 92:95     float num targets
                % 96:99     float correct target angle
                % 100:103   float angle tolerance
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,78:81));

                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,82:85));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,74:77));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,66:69));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,70:73));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,62:65));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                tgtRadius(trial)=bytes2float(cds.databursts.db(58:61));
                tgtCtr(trial,:)=tgtRadius(trial)*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,31:34));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,35:38));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,27:30));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                tgtDuringBump(trial)=cds.databursts.db(36);
                ctrHoldBump(trial)=~tgtDuringBump(trial) && bumpMagnitude(trial)>0;
                delayBump(trial)=tgtDuringBump(trial) && bumpMagnitude(trial)>0;
                moveBump(trial)=false;

                stimTrial(trial)=cds.databursts.db(idxDB,47);
                stimTrialFreq(trial)=bytes2float(cds.databursts.db(idxDB,53:56));
                isPrimaryTarget(trial)=cds.databursts.db(idxDB,91);
                randomTargets(trial)=cds.databursts.db(idxDB,18);
                tgtDirFloor(trial)=bytes2float(cds.databursts.db(idxDB,19:22));
                tgtDirCeil(trial)=bytes2float(cds.databursts.db(idxDB,23:26));
                bumpDirFloor(trial)=bytes2float(cds.databursts.db(idxDB,39:42));
                bumpDirCeil(trial)=bytes2float(cds.databursts.db(idxDB,43:46));
                bumpDirStep(trial)=bytes2float(cds.databursts.db(idxDB,87:90));
                isTrainingTrial(trial)=cds.databursts.db(idxDB,48);
                trainingTrialFreq(trial)=bytes2float(cds.databursts.db(idxDB,49:52));
                recenterCursor(trial)=cds.databursts.db(idxDB,57);
                abortDuringBump(trial)=true;

                numTargets(trial) = bytes2float(cds.databursts.db(idxDB,92:95));
                correctAngle(trial) = bytes2float(cds.databursts.db(idxDB, 96:99));
                angleTolerance(trial) = bytes2float(cds.databursts.db(idxDB, 100:103));

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
            trialsTable=table(ctrHold,tgtOnTime,goCueList,intertrialPeriod,penaltyPeriod,bumpDelay,bumpHold,...
                                tgtSize,tgtAngle,round(tgtCtr,4),tgtRadius,tgtDuringBump,tgtDirFloor,tgtDirCeil,isTrainingTrial,trainingTrialFreq,...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTrial,stimTrialFreq,stimCode,... 
                                recenterCursor,isPrimaryTarget,...
                                numTargets,correctAngle,angleTolerance,...
                                'VariableNames',{'ctrHold','tgtOnTime','goCueTime','intertrialPeriod','penaltyPeriod','bumpDelay','bumpHold'...
                                'tgtSize','tgtDir','tgtCtr','tgtRadius','tgtDuringBump','tgtDirFloor','tgtDirCeil','isTrainingTrial','trainingTrialFreq'...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'isStimTrial','stimTrialFreq','stimCode',...
                                'recenterCursor','isPrimaryTgt',...
                                'numTargets','correctAngle','angleTolerance'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm','cm','bool','deg','deg','bool','pct',...
                                                    's','bool','bool','bool','bool','s','s','N','deg',...
                                                    'bool','pct','int',...
                                                    'bool','bool',...
                                                    'int','deg','deg'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','go cue time','intertrial time','penalty time','time after entering ctr tgt that bump happens','time after bump onset before go cue',...
                                                            'size of targets','angle of outer target','x-y position of outer target','target distance from center','were targets on during bump','min tgt angle','max tgt angle','only the correct target was shown','pct of trials that only show correct target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                            'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                            'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                            'was there stimulation','how often did stim happen','code in the stim word',...
                                                            'did the cursor recenter after bump','is the correct tgt the one in the tgt direction',...
                                                            'number of targets used','correct reach angle','tolerance for reward'};

        otherwise
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for BD is not implemented for databursts with version#:',num2str(dbVersion)])
    end
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the CObump task';
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