function getRingReportingTaskTable(cds,times)
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
    

    wordOTHold = hex2dec('A1');
    wordOTHoldMask = cds.words.word == wordOTHold;
    otHoldTimes=cds.words.ts(wordOTHoldMask);
    
    %preallocate our trial variables:
    numTrials=numel(times.number);
    tgtOnTime=nan(numTrials,1);
    bumpTimeList=nan(numTrials,1);
    goCueList=nan(numTrials,1);
    otHoldTime = nan(numTrials,1);
    
    ctrHold=nan(numTrials,1);
    otHold=nan(numTrials,1);
    delayHold=nan(numTrials,1);
    movePeriod=nan(numTrials,1);
    bumpDelay=nan(numTrials,1);
    bumpHold=nan(numTrials,1);
    intertrialPeriod=nan(numTrials,1);
    penaltyPeriod=nan(numTrials,1);

    tgtSize=nan(numTrials,1);
    bigTgtSize=nan(numTrials,1);
    tgtAngle=nan(numTrials,1);
    tgtRadius = nan(numTrials,1);
    tgtWidth = nan(numTrials,1);
    
    hideCursorDuringBump=false(numTrials,1);
    hideCursorDuringMovement = false(numTrials,1);

    abortDuringBump=false(numTrials,1);
    
    bumpHoldPeriod=nan(numTrials,1);
    bumpRisePeriod=nan(numTrials,1);
    bumpMagnitude=nan(numTrials,1);
    bumpAngle=nan(numTrials,1);
    doBump=nan(numTrials,1);
    
    stimTrial=false(numTrials,1);
    stimCode=nan(numTrials,1);
    stimDuringBump=false(numTrials,1);
    stimInsteadOfBump=false(numTrials,1);
    stimDelay=nan(numTrials,1);
    stimTimeList=nan(numTrials,1);
        
    catchTrial=nan(numTrials,1);
    
    showOuterRing=false(numTrials,1);
    showOuterTarget = false(numTrials,1);
    useSquareTargets = false(numTrials,1);
    
    %get the databurst version:
    dbVersion=cds.databursts.db(1,2);
    skipList=[];
    switch dbVersion
        case 0
            error('getRingReportingTaskTable:unrecognizedDBVersion',['the trial table code for Ring reporting is not implemented for databursts with version#:',num2str(dbVersion)])
        case 1
            % * Version 1 (0x01) 
            %  * ----------------
            %  * Created by modifying version 6 of CObump

            %  * byte  0:		uchar		=> number of bytes to be transmitted
            %  * byte  1:		uchar		=> version number (in this case 1)
            %  * byte  2-4:		uchar		=> task code 'R' 'R' 'B'
            %  * bytes 5-6:		uchar       => version code
            %  * byte  7-8:		uchar		=> version code (micro)

            %  * bytes 9-12:	float		=> center hold time
            %  * bytes 13-16:	float		=> delay time
            %  * bytes 17-20:	float		=> move time
            %  * bytes 21-24:	float		=> bump delay time
            %  * bytes 25-28:	float		=> bump hold time
            %  * bytes 29-32:	float		=> intertrial time
            %  * bytes 33-36:	float		=> penalty time

            %  * bytes 37-40:	float		=> target size
            %  * bytes 41-44:   float       => big target size
            %  * bytes 45-48:   float       => outer target hold time
            %  * bytes 49-52:	float		=> target angle
            %  * bytes 53-56:   float       => movement length (target radius)
            %  * bytes 57-60:   float       => target width (OT_size)

            %  * byte 61:		uchar		=> hide cursor during bump
            %  * byte 62:       uchar       => hide cursor during movement
            %  * byte 63:		uchar		=> abort during bumps

            %  * bytes 64-67:	float		=> bump hold at peak
            %  * bytes 68-71:	float		=> bump rise time
            %  * bytes 72-75:	float		=> bump magnitude
            %  * bytes 76-79:	float		=> bump direction
            %  * byte 80:       uchar       => do bump 

            %  * byte 81:		uchar		=> stim trial
            %  * byte 82:		uchar		=> stim during bump
            %  * byte 83:		uchar		=> stim instead of bump
            %  * bytes 84-87:   float       => stim delay
            %  * byte 88:       uchar        => catch trial

            %  * byte 89:       uchar       => show ring
            %  * byte 90:       uchar       => show outer target
            %  * byte 91:       uchar       => use square targets
            %  */
            
            % loop thorugh our trials and build our list vectors:
            for trial = 1:numTrials
                %find and parse the current databurst:
                idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
                if isempty(idxDB)
                    skipList=[skipList,trial];
                    continue
                end

                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                delayHold(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                movePeriod(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,22:25));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,26:29));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,30:33));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,34:37));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,38:41));
                bigTgtSize(trial)=bytes2float(cds.databursts.db(idxDB,42:45));
                otHold(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,50:53));
                tgtRadius(trial)=bytes2float(cds.databursts.db(idxDB,54:57));
                tgtWidth(trial)=bytes2float(cds.databursts.db(idxDB,58:61));
                
                hideCursorDuringBump(trial) = cds.databursts.db(idxDB,62);
                hideCursorDuringMovement(trial) = cds.databursts.db(idxDB,63);
                abortDuringBump(trial) = cds.databursts.db(idxDB,64);
                
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,65:68));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,69:72));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,73:76));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,77:80));
                doBump(trial)=cds.databursts.db(idxDB,81);
                
                stimTrial(trial)=cds.databursts.db(idxDB,82);
                stimDuringBump(trial)=cds.databursts.db(idxDB,83);
                stimInsteadOfBump(trial)=cds.databursts.db(idxDB,84);
                stimDelay(trial)=bytes2float(cds.databursts.db(idxDB,85:88));
                
                catchTrial(trial)=cds.databursts.db(idxDB,89);
                
                showOuterRing(trial) = cds.databursts.db(idxDB,90);
                showOuterTarget(trial)= cds.databursts.db(idxDB,91);
                useSquareTargets(trial)= cds.databursts.db(idxDB,92);
                
                %now get things that rely only on words and word timing:
                idxOT=find(otOnTimes>times.startTime(trial) & otOnTimes < times.endTime(trial),1,'first');
                if isempty(idxOT)
                    tgtOnTime(trial)=nan;
                else
                    tgtOnTime(trial)=otOnTimes(idxOT);
                end

                % outer target hold time
                idxOTHold=find(otHoldTimes>times.startTime(trial) & otHoldTimes < times.endTime(trial),1,'first');
                if isempty(idxOTHold)
                    otHoldTime(trial)=nan;
                else
                    otHoldTime(trial)=otHoldTimes(idxOTHold);
                end
                
                % Bump code and time
                idxBump = find(bumpTimes > times.startTime(trial) & bumpTimes < times.endTime(trial), 1, 'first');
                if isempty(idxBump)
                    bumpTimeList(trial) = nan;
                    bumpAngle(trial)=nan;
                else
                    bumpTimeList(trial) = bumpTimes(idxBump);
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
                    stimTimeList(trial)=stimTimes(idx);
                    stimCode(trial) = bitand(hex2dec('0f'),stimCodeList(idx));%hex2dec('0f') is a bitwise mask for the trailing bit of the word
                end
            end
            
            % convert bump direction into degrees
            bumpAngle = round(bumpAngle*180/pi);
            bumpAngle(bumpAngle < -180) = bumpAngle(bumpAngle < -180) + 360;
            bumpAngle(bumpAngle > 180) = bumpAngle(bumpAngle > 180) - 360;
            
            tgtAngle = round(tgtAngle*180/pi);
            tgtAngle(tgtAngle < -180) = tgtAngle(tgtAngle < -180) + 360;
            tgtAngle(tgtAngle > 180) = tgtAngle(tgtAngle > 180) - 360;
            %build table:
            trialsTable=table(ctrHold,otHold,tgtOnTime,otHoldTime,delayHold,goCueList,movePeriod,intertrialPeriod,penaltyPeriod,...
                                tgtSize,tgtAngle,tgtWidth,...
                                bumpTimeList,abortDuringBump,doBump,hideCursorDuringBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTimeList,stimCode,stimDuringBump,stimInsteadOfBump,stimDelay,catchTrial,...
                                tgtRadius,showOuterRing,showOuterTarget,useSquareTargets,hideCursorDuringMovement,...
                                'VariableNames',{'ctrHold','otHold','tgtOnTime','otHoldTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'tgtSize','tgtDir','tgtWidth',...
                                'bumpTime','abortDuringBump','doBump','hideCursorDuringBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'stimTime','stimCode','stimDuringBump','stimInsteadOfBump','stimDelay','catchTrial',...
                                'targetRadius','showRing','showOuterTarget','useSquareTargets','hideCursorDuringMovement'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s','s','s',...
                                                    'cm','deg','deg',...
                                                    's','bool','bool','bool','s','s','N','deg',...
                                                    's','int','bool','bool','s','bool',...
                                                    'cm','bool','bool','bool','bool'};

            trialsTable.Properties.VariableDescriptions={'center hold time','outer target hold time','outer target onset time','instructed delay time','time at outer target','go cue time','movement time','intertrial time','penalty time',...
                                                            'size of targets','angle of outer target','angle width of target',...
                                                            'time of bump onset','would we abort during bumps',...
                                                                'did we have a bump','was the cursor shown during the bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                                'time of stimulus on this trial','stim code issued in stim word','flag indicating the stimulus occurred concurrent with a bump',...
                                                                'flag indicating the simulus replaced a bump on this trial','delay after period start at which stimulus word was issued by xpc','flag indicating if catch trial (no bump and no stim)',...
                                                                'radius of the target from the center','flag indicating if outer ring is shown','flag indicating if outer target is shown','flag indicating if square targets were used instead of arc targets','was cursor shown during movement'};

        otherwise
            error('getRingReportingTaskTable:unrecognizedDBVersion',['the trial table code for Ring Reporting is not implemented for databursts with version#:',num2str(dbVersion)])
    end
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the CObump task';
    %sanitize trial table by masking off corrupt databursts with nan's:
    mask= ( trialsTable.ctrHold<0           | trialsTable.ctrHold>10000 | ...
            trialsTable.delayHold<0         | trialsTable.delayHold>10000 |...
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
