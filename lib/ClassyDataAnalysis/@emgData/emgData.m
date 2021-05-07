classdef emgData < timeSeriesData
    %sub-class inheriting from timeSeriesData so that emg specific methods
    % may be added. See also the timeSeriesData
    % class definition for inherited properties and methods
    methods (Static = true)
        %constructor
        function emg=emgData()
        end
    end
    events
        rectified
        processedDefault
    end
    methods (Static = false)
        %general methods
        rectify(emg)
        processDefault(emg)
    end
end