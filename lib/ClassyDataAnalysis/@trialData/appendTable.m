function appendTable(trials,data,varargin)
    %appendTable is a method of the trialData class. This method
    %takes a trials table from a cds object and either inserts it
    %into an empty trialData object, or appends it to the end of an
    %existing trialData object. 
    %
    %appendTable will look for columns with Time in the
    %VariableName property, e.g. startTime, endTime, goCueTime etc.
    %and will add an offset to these trials. The user may specify
    %the offset as a third argument to appendTable, or appendTable
    %will use the largest value in the stopTime column

    
    
    %get our state variables either from input or from defaults:
    if ~isempty(varargin)
        for i=1:2:length(varargin)
            if ~ischar(varargin{i}) || mod(length(varargin),2)>1
                error('appendTable:badKey','additional inputs to the appendTable method must be key-value pairs, with a string as the key')
            end
            switch varargin{i}
                case 'timeShift'
                    timeShift=varargin{i+1};
                case 'overWrite'
                    overWrite=varargin{i+1};
                otherwise
                    error('appendTable:badKeyString',['the key string: ',varargin{i}, 'is not recognized by appendTable'])
            end
        end
    end
    if ~exist('overWrite','var')
        overWrite=false;
    end
    %sanity check that existing trials and new trials have the same
    %columns:
    if ~isempty(trials.data) && ~overWrite && ~isempty(setdiff(trials.data.Properties.VariableNames,data.Properties.VariableNames)) 
        disp(['existing columns: ',strjoin(trials.data.Properties.VariableNames,', ')])
        disp(['new data columns: ',strjoin(data.Properties.VariableNames,', ')])
        error('appendTable:differentColumns','The existing trial data and the new data MUST have the same columns')
    end
    %if a time shift wasn't fed in, generate it:
    if ~exist('timeShift','var')
        if ~isempty(trials.data) && ~overWrite
            %if we are appending data to an existing table
            timeShift=max(trials.data.endTime);
        else
            %if no data exists, or we are overwriting existing data
            timeShift=0;
        end
    end   
        
    if ~isempty(timeShift) && timeShift>0
        %establish the mask that we use to select time columns
        mask=~cellfun('isempty',strfind(data.Properties.VariableNames,'Time'));
        data{:,mask}=data{:,mask}+timeShift;
    end
    if ~isempty(trials.data) && ~overWrite
        %incriment the trial number in data so that the new trials
        %start counting at the end of the old trials
        data.number=data.number+max(trials.data.number)+1;
    end
    if isempty(trials.data) || overWrite
        %just put the new trials in the field
        set(trials,'data',data)
    else
        %get the column index of time columns:
        set(trials,'data',[trials.data;data]);
    end
    trialInfo.numAdded=size(data,1);
    trialInfo.numTotal=size(trials.data,1);
    trialInfo.overwritten=overWrite;
    
    evntData=loggingListenerEventData('appendTable',trialInfo);
    notify(trials,'appended',evntData)
end