function varargout=getMoveStart(ex,varargin)
    %getMoveStart is a method of the experiment class.
    %ex.getMoveStart()
    %assumes that you want to find the initiation of movement between the
    %go cue and trial end. This may not work for trials with multiple
    %movements like Random Walk.
    %trialMoveTime=ex.getMoveStart(windows)
    %looks for movement starts in time ranges supplied by the windows
    %matrix. windows is a column matrix where the first column is the start
    %of a time window, and the second column is the end of a time window.
    %
    %get move start tries to find the peak movement speed within the trial
    %window. It then looks for a local minima prior to that peak speed that
    %meets the following criteria:
    %must be below 5% of the peak speed
    %must be the last minima before peak speed meeting the 5% criteria
    %
    %if no minima are found that meet the criteria, the absolute minima
    %between goCueTime (or the passed start of the time window) and peak
    %speed is used instead.
    %
    %In the case where no input is passed, getMoveStart assumes that you
    %want to append the result into the ex.trial.data table. In cases where
    %an arbitrary window is passed we cannot garauntee that the windows the
    %user selects have a 1-1 ralation to the trials. If the user passes
    %windows to getMoveStart, output will be passed ONLY to the moveTimes
    %variable.
    %
    %moveTimes is a single column vector containing the identified time of
    %motion onset for each time window
    
    if nargout>1
        error('getMoveStart:tooManyOutputs','getMoveStart only supports a single output')
    end
    
    if isempty(ex.kin.data)
        error('getMoveStart:missingKinmeaticData','there is no kinematic data in the experiment')
    elseif isempty(ex.trials.data)
        error('getMoveStart:noTrialData','there is no trial data in the experiment')
    end
    
    if numel(varargin)>0
        moveWindows=varargin{1};
        %set flag to put data directly into ex.trials.data:
        updateTrials=false;
    else
        %get the movement window for trials that got to the move phase and
        %weren't incomplete:
        if isempty(find(strcmp('goCueTime',ex.trials.data.Properties.VariableNames),1))
            error('getMoveStart:noGoCueTime','the trials table does not have goCueTimes')
        elseif isempty(find(strcmp('endTime',ex.trials.data.Properties.VariableNames),1))
            error('getMoveStart:noGoCueTime','the trials table does not have endTimes')
        end
        moveMask=~isnan(ex.trials.data.goCueTime(:,1));
        moveMask(strmatch('I',ex.trials.data.result,'exact'))=false;
        moveWindows=[ex.trials.data.goCueTime(moveMask,1),ex.trials.data.endTime(moveMask)];
        %set flag to put data directly into ex.trials.data:
        updateTrials=true;
    end
    
    %compute movement speed:
    speed=sqrt(ex.kin.data.vx.^2+ex.kin.data.vy.^2);
    
    moveTime=nan(size(moveWindows(:,1)));
    %loop through windows:
    for i=1:size(moveWindows,1)
        %get the index of the first point in the trial as the following
        %code will work only withing the trial and we need a reference back
        %to the whole timeseries:
        offset=find(ex.kin.data.t>moveWindows(i,1),1,'first');
        idxEnd=find(ex.kin.data.t>moveWindows(i,2),1,'first');
        if idxEnd==offset
            %this can happen if there is no abort when the monkey leaves
            %the center during a delay period, and is already in the target
            %at the go-cue. The go-cue then immediatley triggers the trial
            %end, in ~10ms, and there will be no kinematics between goCue
            %and trial end
            moveTime(i)=nan;
            continue
        end
        %get list of extrema in this window
        [peaks,ipeaks,valleys,ivalleys]=extrema(speed(offset:idxEnd));
        %get list of peaks sorted by when they happen in the window
        peakData=sortrows([peaks,ipeaks],2);
        %get list of valleys sorted by when they happen in the window
        valleyData=sortrows([valleys,ivalleys],2);
        %find which peak is our max speed
        [~,imax]=max(peakData(:,1));
        if peakData(imax,2)>1
            %get the first minima before the peak that is below 5% of the peak
            %amplitude
            candidates=valleyData(valleyData(:,1)<.05*peakData(imax,1),:);
            if isempty(candidates) || min(candidates(:,2))>peakData(imax,2)
                %just get the global minima between the window start and 
                %the peak speed:
                [~,imin]=min(speed(offset:offset+peakData(imax,2)));
            else
                %find the last candidate before peak speed:
                idxMin=find(candidates(:,2)<peakData(imax,2),1,'last');
                imin=candidates(idxMin,2);
            end
            moveTime(i)=ex.kin.data.t(offset+imin);
        else
            %peak speed was the beginning of the trial. I don't know how
            %the monkey finished the trial while going slower than he was
            %during the delay, but we are just gonna stick an NaN in there
            %and ignore it. Hopefully this never actually happens
            moveTime(i)=nan;
        end
        
    end
    %append new timing data to the trials table:
    moveTimes=nan(size(ex.trials.data.endTime));
    moveTimes(moveMask)=moveTime;
    if updateTrials
        mask=true(size(ex.trials.data,2),1);
        %if we already have a moveTimes column, set the mask to exclude it
        %so we don't get a conflict:
        idxMove=find(strcmp('moveTime',ex.trials.data.Properties.VariableNames),1);
        mask(idxMove)=false;
        %generate a new trials table with the new moveTimes
        trials=[ex.trials.data(:,mask),table(moveTimes,'VariableNames',{'moveTime'})];
        %set the ex.trials.data table to the new trials table
        ex.trials.appendTable(trials,'overWrite',true)
        evntData=loggingListenerEventData('getMoveStart',[]);
        notify(ex,'ranOperation',evntData)
    end
    if nargout==1
        varargout{1}=moveTimes;
    else
        varargout{1}=[];
    end
end