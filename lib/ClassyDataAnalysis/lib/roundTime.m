function roundedTime=roundTime(timeData,varargin)
    %takes in a time vector and returns a vector with superfluous precision
    %rounded out. Passing timeData=[0.0011,0.0111,0.0211,0.0311...] would
    %result in roundedTime=[0.0000,0.0100,0.0200,0.0300...]. roundTime uses
    %the difference between sample points to identify the appropriate
    %resolution to round time to.
    %
    %roundTime occasionally has machine precision problems following from
    %the floating point operation at the arbitrary frequency. To compensate 
    %for these, a secondary rounding operation with decimal precision is
    %added. This limits the frequency output of roundTime to 100Mhz.
    
    %define the time step to round to:
    if isempty(varargin)
        % get the time-steps in the data. 
        deltas=diff(timeData);
        dt=mode(deltas);
        %use round to get rid of machine precision problems by rounding to 7
        %decimal places. skipping this can result in time vectors that look
        %like: [1.00999999...;1.019999999...;1.029999999...; etc]
        dt=round(dt,7);
        if ~isempty(find(deltas<0,1)) || dt==0
            %we have a collection of times, not a monotonic time vector just use
            %7 sig figs as an arbitrary precision
            dt=.0000001;
        end
    else
        dt=varargin{1};
    end
    %round timeData to the nearest multiple of dt 
    roundedTime=round(timeData/dt)*dt;
    %now handle stupid machine precision problems by rounding at some
    %stupidly high number of sig-figs so that we'd capture 100Mhz frequency,
    %but will get rid of machine precision errors:
    roundedTime=round(roundedTime,7);
end