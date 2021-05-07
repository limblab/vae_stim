function getCObumpTaskTable(cds,times)
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
    vibTimesAll = cds.NEV.Data.Comments.TimeStampSec;
    vibStringAll = cds.NEV.Data.Comments.Text;
    if ~isempty(vibStringAll)
    vibStringOn = vibStringAll(vibStringAll(:,8)~= 'O',:);
    vibStringOff = vibStringAll(vibStringAll(:,8) == 'O',:);
    
    vibOnTime = vibTimesAll(vibStringAll(:, 8) ~= 'O');
    vibOffTime = vibTimesAll(vibStringAll(:, 8) == 'O');
    vibWindow = [vibOnTime', vibOffTime'];
    
%     vibNum = vibStringAll(:,
    
    %preallocate our trial variables:
    vibNumOn =str2num(vibStringOn(:,8:end));
    vibNumOff = str2num(vibStringOff(:,11:end));
    vibNum = [vibNumOn, vibNumOff];
    end
    numTrials=numel(times.number);
    tgtOnTime=nan(numTrials,1);
    tgtID=nan(numTrials,1);   bumpTimeList=nan(numTrials,1);
    bumpList=nan(numTrials,1);
    goCueList=nan(numTrials,1);
    ctrHold=nan(numTrials,1);
    otHold=nan(numTrials,1);
    delayHold=nan(numTrials,1);
    movePeriod=nan(numTrials,1);
    bumpDelay=nan(numTrials,1);
    bumpHold=nan(numTrials,1);
    intertrialPeriod=nan(numTrials,1);
    penaltyPeriod=nan(numTrials,1);

    tgtSize=nan(numTrials,1);
    tgtAngle=nan(numTrials,1);
    tgtCtr=nan(numTrials,2);

    hideCursor=false(numTrials,1);
    hideCursorMin=nan(numTrials,1);
    hideCursorMax=nan(numTrials,1);

    abortDuringBump=false(numTrials,1);
    ctrHoldBump=false(numTrials,1);
    delayBump=false(numTrials,1);
    moveBump=false(numTrials,1);
    bumpHoldPeriod=nan(numTrials,1);
    bumpRisePeriod=nan(numTrials,1);
    bumpMagnitude=nan(numTrials,1);
    bumpAngle=nan(numTrials,1);

    stimTrial=false(numTrials,1);
    stimCode=nan(numTrials,1);
    stimDuringBump=false(numTrials,1);
    stimInsteadOfBump=false(numTrials,1);
    stimDelay=nan(numTrials,1);
    stimTimeList=nan(numTrials,1);
    
    trialRedo=nan(numTrials,1);
    
    %get the databurst version:
    dbVersion=cds.databursts.db(1,2);
    skipList=[];
    
    switch dbVersion
        case 0
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for CObump is not implemented for databursts with version#:',num2str(dbVersion)])
        case 1
            % Databurst only has 30 bytes...?
            error('getCObumpTaskTable:unrecognizedDBVersion','the trial table code for CObump is not implemented for databursts with version#: %d',dbVersion)
                % * Version 1 (0x01)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case 0)
                %  * byte  2-4:	uchar		=> task code 'C' 'O' 'B'
                %  * bytes 5-6:	uchar       => version code
                %  * byte  7-8:	uchar		=> version code (micro)
                %  * bytes 9-12:  float		=> target angle
                %  * byte	 13:	uchar		=> random target flag
                %  * bytes 14-17:	float		=> target floor (minimum angle(deg) target can take in random target assignment)
                %  * bytes 18-21:	float		=> target ceiling (maximum angle(deg) target can take in random target assignment)
                %  * bytes 22-25:	float		=> target incriment(deg)
                %  * bytes 26-29: float		=> bump magnitude
                %  * bytes 30-33: float		=> bump direction
                %  * bytes 34-37: float		=> bump duration
                %  * bytes 38-41: float		=> bump ramp
                %  * byte  42:	uchar		=> stim trial flag
                %  * bytes 43-46: float		=> stimulation probability 
                %  * bytes 47-50: float		=> target radius
                %  * bytes 51-54: float		=> target size
                %  * bytes 55-58: float		=> intertrial time
                %  * bytes 59-62: float		=> penalty time
                %  * bytes 63-66: float		=> bump hold time
                %  * bytes 67-70: float		=> center hold time
                %  * bytes 71-74: float		=> bump delay time
                %  * byte  75:	uchar		=> flag for whether or not the cursor is hidden during movement
                %  * bytes 76-79: float		=> radius from center within which the cursor will be hidden
                
                % databurst stuff...
                % databurst seems to not match what the code says it should be...
                
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
            

            % Reorient bump angle to be relative to world coordinates instead of target
            bumpAngle = mod(bumpAngle + tgtAngle,360);
            
            %build table:
            trialsTable=table(tgtOnTime,goCueList,bumpTimeList,...
                                'VariableNames',{'tgtOnTime','goCueTime','bumpTime'});

            trialsTable.Properties.VariableUnits={'s','s','s'};
            trialsTable.Properties.VariableDescriptions={'outer target onset time','go cue time','time of bump onset'};
            
        case 2
            % loop thorugh our trials and build our list vectors:
            for trial = 1:numTrials
                %find and parse the current databurst:
                idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
                if isempty(idxDB)
                    skipList=[skipList,trial];
                    continue
                end
                % * Version 2 (0x02)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case 2)
                %  * byte  2-4:		uchar		=> task code 'C' 'O' 'B'
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
                %  * bytes 41-44:	float		=> target radius
                %  * bytes 45-48:	float		=> target angle

                %  * byte 49:		uchar		=> hide cursor
                %  * bytes 50-53:	float		=> hide radius min
                %  * bytes 54-57:	float		=> hide radius max

                %  * byte 58:		uchar		=> abort during bumps
                %  * bytes 59:      uchar		=> catch trial rate: THIS IS BUGGY-Casts the rate as a uchar, rather than a float
                %  * byte 60:		uchar		=> do center hold bump
                %  * byte 61:		uchar		=> do delay period bump
                %  * byte 62:		uchar		=> do move bump
                %  * bytes 63-66:	float		=> bump hold at peak
                %  * bytes 67-70:	float		=> bump rise time
                %  * bytes 71-74:	float		=> bump magnitude
                %  * bytes 75-78:	float		=> bump direction (relative to target angle)

                %  * byte 79:		uchar		=> stim trial
                %  * bytes 80-83:	float		=> stim trial rate
                %  */
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                delayHold(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                movePeriod(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,22:25));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,26:29));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,30:33));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,34:37));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,38:41));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtCtr(trial,:)=bytes2float(cds.databursts.db(idxDB,42:45))*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                hideCursor(trial)=cds.databursts.db(idxDB,50);
                hideCursorMin(trial)=bytes2float(cds.databursts.db(idxDB,51:54));
                hideCursorMax(trial)=bytes2float(cds.databursts.db(idxDB,55:58));

                abortDuringBump(trial)=cds.databursts.db(idxDB,59);
                ctrHoldBump(trial)=cds.databursts.db(idxDB,60);
                delayBump(trial)=cds.databursts.db(idxDB,61);
                moveBump(trial)=cds.databursts.db(idxDB,62);
                
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,63:66));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,67:70));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,71:74));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,75:78));
                
                stimTrial(trial)=cds.databursts.db(idxDB,79);


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

            % Reorient bump angle to be relative to world coordinates instead of target
            bumpAngle = mod(bumpAngle + tgtAngle,360);
            
            %build table:
            trialsTable=table(ctrHold,tgtOnTime,delayHold,goCueList,movePeriod,intertrialPeriod,penaltyPeriod,...
                                tgtSize,tgtAngle,round(tgtCtr,4),...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                'VariableNames',{'ctrHold','tgtOnTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'tgtSize','tgtDir','tgtCtr',...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm',...
                                                    's','bool','bool','bool','bool','s','s','N','deg'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','instructed delay time','go cue time','movement time','intertrial time','penalty time',...
                                                            'size of targets','angle of outer target','x-y position of outer target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                                'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump'};
            
        case 3
                        % loop thorugh our trials and build our list vectors:
            for trial = 1:numTrials
                %find and parse the current databurst:
                idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
                if isempty(idxDB)
                    skipList=[skipList,trial];
                    continue
                end
                % * Version 3 (0x03)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case 0)
                %  * byte  2-4:		uchar		=> task code 'C' 'O' 'B'
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
                %  * bytes 41-44:	float		=> target radius
                %  * bytes 45-48:	float		=> target angle

                %  * byte 49:		uchar		=> hide cursor
                %  * bytes 50-53:	float		=> hide radius min
                %  * bytes 54-57:	float		=> hide radius max

                %  * byte 58:		uchar		=> abort during bumps
                %  * byte 59:		uchar		=> do center hold bump
                %  * byte 60:		uchar		=> do delay period bump
                %  * byte 61:		uchar		=> do move bump
                %  * bytes 62-65:	float		=> bump hold at peak
                %  * bytes 66-69:	float		=> bump rise time
                %  * bytes 70-73:	float		=> bump magnitude
                %  * bytes 74-77:	float		=> bump direction

                %  * byte 78:		uchar		=> stim trial
                %  */
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                delayHold(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                movePeriod(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,22:25));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,26:29));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,30:33));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,34:37));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,38:41));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtCtr(trial)=bytes2float(cds.databursts.db(idxDB,42:45))*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                hideCursor(trial)=cds.databursts.db(idxDB,50);
                hideCursorMin(trial)=bytes2float(cds.databursts.db(idxDB,51:54));
                hideCursorMax(trial)=bytes2float(cds.databursts.db(idxDB,55:58));

                abortDuringBump(trial)=cds.databursts.db(idxDB,59);
                ctrHoldBump(trial)=cds.databursts.db(idxDB,60);
                delayBump(trial)=cds.databursts.db(idxDB,61);
                moveBump(trial)=cds.databursts.db(idxDB,62);
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,63:66));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,67:70));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,71:74));
                %bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,75:78));
                tmpAngle=bytes2float(cds.databursts.db(idxDB,75:78))-tgtAngle(trial);
                if tmpAngle>=360
                    tmpAngle=tmpAngle-360;
                end
                if tmpAngle<0
                    tmpAngle=tmpAngle+360;
                end
                bumpAngle(trial)=tmpAngle;
                stimTrial(trial)=cds.databursts.db(idxDB,79);


                %now get things that rely only on words and word timing:
                idxOT=find(otOnTimes>times.startTime(trial) & otOnTimes < times.endTime(trial),1,'first');
                if isempty(idxOT)
                    tgtOnTime(trial)=nan;
                    %tgtID(trial)=nan;%target ID has no meaning in this version of the databurst
                else
                    tgtOnTime(trial)=otOnTimes(idxOT);
                    %tgtID(trial)=otOnCodes(idxOT);%target ID has no meaning in this version of the databurst
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
                if ~isempty(idx)
                    stimCode(trial) = bitand(hex2dec('0f'),stimCode(idx));%hex2dec('0f') is a bitwise mask for the trailing bit of the word
                else
                    stimCode(trial) = nan;
                end
            end

            %build table:
            trialsTable=table(roundTime(ctrHold,.001),roundTime(tgtOnTime,.001),roundTime(delayHold,.001),roundTime(goCueList,.001),roundTime(movePeriod,.001),roundTime(intertrialPeriod,.001),roundTime(penaltyPeriod,.001),...
                                tgtSize,tgtAngle,round(tgtCtr,2),...
                                roundTime(bumpTimeList,.001),abortDuringBump,ctrHoldBump,delayBump,moveBump,roundTime(bumpHoldPeriod,.001),roundTime(bumpRisePeriod,.001),bumpMagnitude,bumpAngle,...
                                'VariableNames',{'ctrHold','tgtOnTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'tgtSize','tgtDir','tgtCtr',...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm',...
                                                    's','bool','bool','bool','bool','s','s','N','deg'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','instructed delay time','go cue time','movement time','intertrial time','penalty time',...
                                                            'size of targets','angle of outer target','x-y position of outer target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                                'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump'};
        
        case 4
            % loop thorugh our trials and build our list vectors:
            for trial = 1:numTrials
                %find and parse the current databurst:
                idxDB = find(cds.databursts.ts > times.startTime(trial) & cds.databursts.ts<times.endTime(trial), 1, 'first');
                if isempty(idxDB)
                    skipList=[skipList,trial];
                    continue
                end
                % * Version 4 (0x04)
                %  * ----------------
                %  * byte  0:		uchar		=> number of bytes to be transmitted
                %  * byte  1:		uchar		=> version number (in this case 0)
                %  * byte  2-4:		uchar		=> task code 'C' 'O' 'B'
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
                %  * bytes 41-44:	float		=> target radius
                %  * bytes 45-48:	float		=> target angle

                %  * byte 49:		uchar		=> hide cursor
                %  * bytes 50-53:	float		=> hide radius min
                %  * bytes 54-57:	float		=> hide radius max

                %  * byte 58:		uchar		=> abort during bumps
                %  * bytes 59:      uchar		=> catch trial rate: THIS IS BUGGY-Casts the rate as a uchar, rather than a float
                %  * byte 60:		uchar		=> do center hold bump
                %  * byte 61:		uchar		=> do delay period bump
                %  * byte 62:		uchar		=> do move bump
                %  * bytes 63-66:	float		=> bump hold at peak
                %  * bytes 67-70:	float		=> bump rise time
                %  * bytes 71-74:	float		=> bump magnitude
                %  * bytes 75-78:	float		=> bump direction (actual)

                %  * byte 79:		uchar		=> stim trial
                %  * bytes 80-83:	float		=> stim trial rate
                %  */
                ctrHold(trial)=bytes2float(cds.databursts.db(idxDB,10:13));
                delayHold(trial)=bytes2float(cds.databursts.db(idxDB,14:17));
                movePeriod(trial)=bytes2float(cds.databursts.db(idxDB,18:21));
                bumpDelay(trial)=bytes2float(cds.databursts.db(idxDB,22:25));
                bumpHold(trial)=bytes2float(cds.databursts.db(idxDB,26:29));
                intertrialPeriod(trial)=bytes2float(cds.databursts.db(idxDB,30:33));
                penaltyPeriod(trial)=bytes2float(cds.databursts.db(idxDB,34:37));

                tgtSize(trial)=bytes2float(cds.databursts.db(idxDB,38:41));
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtCtr(trial,:)=bytes2float(cds.databursts.db(idxDB,42:45))*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                hideCursor(trial)=cds.databursts.db(idxDB,50);
                hideCursorMin(trial)=bytes2float(cds.databursts.db(idxDB,51:54));
                hideCursorMax(trial)=bytes2float(cds.databursts.db(idxDB,55:58));

                abortDuringBump(trial)=cds.databursts.db(idxDB,59);
                ctrHoldBump(trial)=cds.databursts.db(idxDB,60);
                delayBump(trial)=cds.databursts.db(idxDB,61);
                moveBump(trial)=cds.databursts.db(idxDB,62);
                
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,63:66));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,67:70));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,71:74));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,75:78));
                
                stimTrial(trial)=cds.databursts.db(idxDB,79);


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
            
            % convert bump direction into degrees
            bumpAngle = round(bumpAngle*180/pi);
            
            %build table:
            trialsTable=table(ctrHold,tgtOnTime,delayHold,goCueList,movePeriod,intertrialPeriod,penaltyPeriod,...
                                tgtSize,tgtAngle,round(tgtCtr,4),...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                'VariableNames',{'ctrHold','tgtOnTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'tgtSize','tgtDir','tgtCtr',...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm',...
                                                    's','bool','bool','bool','bool','s','s','N','deg'};
            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','instructed delay time','go cue time','movement time','intertrial time','penalty time',...
                                                            'size of targets','angle of outer target','x-y position of outer target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                                'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump'};
        case 5
            % * Version 5 (0x05)
            %  * ----------------
            %  * byte  0:		uchar		=> number of bytes to be transmitted
            %  * byte  1:		uchar		=> version number (in this case 5)
            %  * byte  2-4:		uchar		=> task code 'C' 'O' 'B'
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
            %  * bytes 41-44:	float		=> target radius
            %  * bytes 45-48:	float		=> target angle

            %  * byte 49:		uchar		=> hide cursor
            %  * bytes 50-53:	float		=> hide radius min
            %  * bytes 54-57:	float		=> hide radius max

            %  * byte 58:		uchar		=> abort during bumps
            %  * byte 59:		uchar		=> do center hold bump
            %  * byte 60:		uchar		=> do delay period bump
            %  * byte 61:		uchar		=> do move bump
            %  * bytes 62-65:	float		=> bump hold at peak
            %  * bytes 66-69:	float		=> bump rise time
            %  * bytes 70-73:	float		=> bump magnitude
            %  * bytes 74-77:	float		=> bump direction

            %  * byte 78:		uchar		=> stim trial
            %  * byte 79:		uchar		=> stim during bump
            %  * byte 80:		uchar		=> stim instead of bump
            %  * bytes 81-84:	float		=> stim delay time

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
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtCtr(trial,:)=bytes2float(cds.databursts.db(idxDB,42:45))*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                hideCursor(trial)=cds.databursts.db(idxDB,50);
                hideCursorMin(trial)=bytes2float(cds.databursts.db(idxDB,51:54));
                hideCursorMax(trial)=bytes2float(cds.databursts.db(idxDB,55:58));

                abortDuringBump(trial)=cds.databursts.db(idxDB,59);
                ctrHoldBump(trial)=cds.databursts.db(idxDB,60);
                delayBump(trial)=cds.databursts.db(idxDB,61);
                moveBump(trial)=cds.databursts.db(idxDB,62);
                
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,63:66));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,67:70));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,71:74));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,75:78));
                
                stimTrial(trial)=cds.databursts.db(idxDB,79);

                stimDuringBump(trial)=cds.databursts.db(idxDB,80);
                stimInsteadOfBump(trial)=cds.databursts.db(idxDB,81);
%                 stimDelay(trial)=bytes2float(cds.databursts.db(idxDB,82:85));
                

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
                    stimTimeList(trial)=stimTimes(idx);
                    stimCode(trial) = bitand(hex2dec('0f'),stimCodeList(idx));%hex2dec('0f') is a bitwise mask for the trailing bit of the word
                end
                
                

            end
            
            % convert bump direction into degrees
            bumpAngle = round(bumpAngle*180/pi);
            
            %build table:
            trialsTable=table(ctrHold,tgtOnTime,delayHold,goCueList,movePeriod,intertrialPeriod,penaltyPeriod,...
                                tgtSize,tgtAngle,round(tgtCtr,4),...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTimeList,stimCode,stimDuringBump,stimInsteadOfBump,stimDelay,...
                                'VariableNames',{'ctrHold','tgtOnTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'tgtSize','tgtDir','tgtCtr',...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'stimTime','stimCode','stimDuringBump','stimInsteadOfBump','stimDelay'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s',...
                                                    'cm','deg','cm, cm',...
                                                    's','bool','bool','bool','bool','s','s','N','deg',...
                                                    's','int','bool','bool','s'};

            trialsTable.Properties.VariableDescriptions={'center hold time','outer target onset time','instructed delay time','go cue time','movement time','intertrial time','penalty time',...
                                                            'size of targets','angle of outer target','x-y position of outer target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                                'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                                'time of stimulus on this trial','stim code issued in stim word','flag indicating the stimulus occurred concurrent with a bump',...
                                                                'flag indicating the simulus replaced a bump on this trial','delay after period start at which stimulus word was issued by xpc'};

        case 6
            % * Version 6 (0x06)
            %  * ----------------
            %   Adding idiot mode and outer target hold
            %  * byte  0:		uchar		=> number of bytes to be transmitted
            %  * byte  1:		uchar		=> version number (in this case 6)
            %  * byte  2-4:		uchar		=> task code 'C' 'O' 'B'
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
            %  * bytes 41-44:	float		=> target radius
            %  * bytes 45-48:	float		=> target angle

            %  * byte 49:		uchar		=> hide cursor
            %  * bytes 50-53:	float		=> hide radius min
            %  * bytes 54-57:	float		=> hide radius max

            %  * byte 58:		uchar		=> abort during bumps
            %  * byte 59:		uchar		=> do center hold bump
            %  * byte 60:		uchar		=> do delay period bump
            %  * byte 61:		uchar		=> do move bump
            %  * bytes 62-65:	float		=> bump hold at peak
            %  * bytes 66-69:	float		=> bump rise time
            %  * bytes 70-73:	float		=> bump magnitude
            %  * bytes 74-77:	float		=> bump direction

            %  * byte 78:		uchar		=> stim trial
            %  * byte 79:		uchar		=> stim during bump
            %  * byte 80:		uchar		=> stim instead of bump
            %  * bytes 81-84:	float		=> stim delay time

            %  * bytes 85-88:   float       => outer target hold time
            %  * byte 89:       uchar       => redo trial (bool, true if trial is redone because of idiot mode)
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
                tgtAngle(trial)=bytes2float(cds.databursts.db(idxDB,46:49));
                tgtCtr(trial,:)=bytes2float(cds.databursts.db(idxDB,42:45))*[cos(tgtAngle(trial)*pi/180),sin(tgtAngle(trial)*pi/180)];

                hideCursor(trial)=cds.databursts.db(idxDB,50);
                hideCursorMin(trial)=bytes2float(cds.databursts.db(idxDB,51:54));
                hideCursorMax(trial)=bytes2float(cds.databursts.db(idxDB,55:58));

                abortDuringBump(trial)=cds.databursts.db(idxDB,59);
                ctrHoldBump(trial)=cds.databursts.db(idxDB,60);
                delayBump(trial)=cds.databursts.db(idxDB,61);
                moveBump(trial)=cds.databursts.db(idxDB,62);
                
                bumpHoldPeriod(trial)=bytes2float(cds.databursts.db(idxDB,63:66));
                bumpRisePeriod(trial)=bytes2float(cds.databursts.db(idxDB,67:70));
                bumpMagnitude(trial)=bytes2float(cds.databursts.db(idxDB,71:74));
                bumpAngle(trial)=bytes2float(cds.databursts.db(idxDB,75:78));
                
                stimTrial(trial)=cds.databursts.db(idxDB,79);

                stimDuringBump(trial)=cds.databursts.db(idxDB,80);
                stimInsteadOfBump(trial)=cds.databursts.db(idxDB,81);
                stimDelay(trial)=bytes2float(cds.databursts.db(idxDB,82:85));
                
                otHold(trial)=bytes2float(cds.databursts.db(idxDB,86:89));
                trialRedo(trial)=cds.databursts.db(idxDB,90);

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
                    stimTimeList(trial)=stimTimes(idx);
                    stimCode(trial) = bitand(hex2dec('0f'),stimCodeList(idx));%hex2dec('0f') is a bitwise mask for the trailing bit of the word
                end
                
%                 idxVibOn = find(vibOnTime > times.startTime(trial)-.4 & vibOnTime < times.endTime(trial),1);
%                 
%                 if ~isempty(idxVibOn)
%                     if vibOnTime(idxVibOn) < times.startTime(trial)
%                         vibOnTime(idxVibOn) = times.startTime(trial);
%                         cds.addProblem('Truncated vibration times at start of trial')
%                     end
%                     vibOnTimeList(trial)=vibOnTime(idxVibOn);
%                     vibNumOnList(trial) = vibNumOn(idxVibOn);
%                 else 
%                     vibOnTimeList(trial) = nan;
%                     vibNumOnList(trial) = nan;
%                 end
% 
%                 %% Have to put a little padding on this, because sometimes teh vibration goes a little past the end of the trial
%                 
%                 idxVibOff = find(vibOffTime > times.startTime(trial) & vibOffTime < times.endTime(trial)+.2,1);
%                                
%                 if ~isempty(idxVibOff)
%                     if vibOffTime(idxVibOff) > times.endTime(trial)
%                         vibOffTime(idxVibOff) = times.endTime(trial);
%                         cds.addProblem('Truncated vibration times at end of trial')
%                     end
%                     vibOffTimeList(trial)=vibOffTime(idxVibOff);
%                     vibNumOffList(trial) = vibNumOff(idxVibOff);
%                 else
%                     vibOffTimeList(trial) = nan;
%                     vibNumOffList(trial) = nan;
%                end
            end
            
            % convert bump direction into degrees
            bumpAngle = round(bumpAngle*180/pi);
            
            %build table:
            trialsTable=table(ctrHold,otHold,tgtOnTime,delayHold,goCueList,movePeriod,intertrialPeriod,penaltyPeriod,...
                                trialRedo,tgtSize,tgtAngle,round(tgtCtr,4),...
                                bumpTimeList,abortDuringBump,ctrHoldBump,delayBump,moveBump,bumpHoldPeriod,bumpRisePeriod,bumpMagnitude,bumpAngle,...
                                stimTimeList,stimCode,stimDuringBump,stimInsteadOfBump,stimDelay,... % vibOnTimeList', vibOffTimeList',vibNumOnList', vibNumOffList',...
                                'VariableNames',{'ctrHold','otHold','tgtOnTime','delayHold','goCueTime','movePeriod','intertrialPeriod','penaltyPeriod',...
                                'trialRedo','tgtSize','tgtDir','tgtCtr',...
                                'bumpTime','abortDuringBump','ctrHoldBump','delayBump','moveBump','bumpHoldPeriod','bumpRisePeriod','bumpMagnitude','bumpDir',...
                                'stimTime','stimCode','stimDuringBump','stimInsteadOfBump','stimDelay'});
                                %'vibOnTime', 'vibOffTime', 'chanVibOn', 'chanVibOff'});

            trialsTable.Properties.VariableUnits={'s','s','s','s','s','s','s','s',...
                                                    'bool','cm','deg','cm, cm',...
                                                    's','bool','bool','bool','bool','s','s','N','deg',...
                                                    's','int','bool','bool','s'}%, 's', 's', 'arb', 'arb'};

            trialsTable.Properties.VariableDescriptions={'center hold time','outer target hold time','outer target onset time','instructed delay time','go cue time','movement time','intertrial time','penalty time',...
                                                            'whether this trial is a redo','size of targets','angle of outer target','x-y position of outer target',...
                                                            'time of bump onset','would we abort during bumps','did we have a center hold bump',...
                                                                'did we have a delay period bump','did we have a movement period bump','the time the bump was held at peak amplitude',...
                                                                'the time the bump took to rise and fall from peak amplitude','magnitude of the bump','direction of the bump',...
                                                                'time of stimulus on this trial','stim code issued in stim word','flag indicating the stimulus occurred concurrent with a bump',...
                                                                'flag indicating the simulus replaced a bump on this trial','delay after period start at which stimulus word was issued by xpc'}
                                                                %'when vibration on comment was written', 'when vibration off comment was written', 'Which channels were vibd','Which channels have vib turn off'};

        otherwise
            error('getCObumpTaskTable:unrecognizedDBVersion',['the trial table code for CObump is not implemented for databursts with version#:',num2str(dbVersion)])
    end
    
    trialsTable=[times,trialsTable];
    trialsTable.Properties.Description='Trial table for the CObump task';
    try
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
    catch ME
        warning('getCObumpTaskTable:incompleteTrialTable','The trial table might be missing some information')
    end
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
