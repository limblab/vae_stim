function appendTable(tsd,data,varargin)
    %this is a method of the timeSeriesData class and should be saved in
    %the @timeSeriesData folder
    %
    %tsd.appendTable(data)
    %appends the table: 'data' to the tsd.data field. If tsd.data is empty,
    %then it will be populated with data. if tsd.data is populated already,
    %then the data input to appendTable must have the same columns as the
    %existing table in tsd.data.
    %tsd.appendTable(data,'key',value)
    %allows the user to define how appendTable operates using key-value
    %pairs. Currently defined keys are:
    %'timeShift':   Allows the user to pass a specific shift in the times
    %               of the new data. Shifts will be applied directly e.g.
    %               passing 1500 will shift the times of the new data by
    %               1500s. If no value is passed, the default shift is the
    %               last time in the existing data +1/sampleRate
    %'overWrite':   allows the user to overwrite the existing data rather
    %               than tack on to the end of data that already exists.
    %               The value for this key is a bool (e.g. true/false or
    %               1/0)

    %sanity check input
    if ~istable(data)
        error('appendTable:inputNotATable',['input to append table must be a table. Instead of a table, appendTable got a: ',class(data)])
    end
    if isempty(find(cell2mat({strcmp(data.Properties.VariableNames,'t')}),1))
        error('appendTable:notATimeSeries',['The input table does not have a column labeled t. It is possible this table is misconfigured, or is not time series data. columns in the input table are: ',strjoin(data.Properties.VariableNames,', ')])
    end
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
    if ~overWrite && ~isempty(tsd.data) && mode(diff(tsd.data.t))~=mode(diff(data.t))
        error('appendTable:differentSampleRates','the existing data and the new data are at different frequencies. Please decimate one of the two data sets')
    end
    if ~exist('timeShift','var')
        if ~isempty(tsd.data) && ~overWrite
            timeShift=max(tsd.data.t)+mode(diff(data.t));
        else
            timeShift=0;
        end
    end            
    %if we have a timeshift, apply it:
    if timeShift>0
        if isempty(tsd.data)
            warning('appendTable:shiftedNewData','applying a time shift to data that is being placed in an empty timeSeriesData.data field')
        elseif timeShift<max(tsd.data.t)
            error('appendTable:timeShiftTooSmall','when attempting to append new data, the specified time shift must be larger than the largest existing time')
        end
        %offset the times:
        data.t=data.t+timeShift-data.t(1);
        dt=mode(diff(data.t));
        if timeShift-max(tsd.data.t)>dt
            %build a block of nan values to fill the gap, so we get the correct
            %count when we use the mode timestep to estimate the number of
            %points in decimate data or other functions:
            plug=[(max(tsd.data.t)+dt):dt:timeShift,nan(round(1/dt),size(data,2)-1)];
            plug=array2table(plug,'VariabelNames',data.Properties.VariableNames);
            data=[plug;data];
        end
    end
    % put the data into the data field
    if isempty(tsd.data) || overWrite
        %just put the new dt in the field
        set(tsd,'data',data)
    else
        %get the column index of timestamp or time, whichever this
        %table is using:
        set(tsd,'data',[tsd.data;data]);
    end
    %log the appendTable event
    cfg.timeShift=timeShift;
    cfg.overWrite=overWrite;
    evntData=loggingListenerEventData('appendTable',cfg);
    notify(tsd,'appended',evntData)
end