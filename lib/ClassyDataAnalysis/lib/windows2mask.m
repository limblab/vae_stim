function mask=windows2mask(time,windows)
    %windows2mask gets a logical index vector by applying time windows to a
    %vector of timestamps.
    %mask=windows2mask(time,windows)
    %time is a column vector of timestamps. Time is not required to be
    %   sorted for this function
    %windows is a column matrix [tStart1,tEnd1; tStart2,tEnd2; tStart3,
    %   tEnd3; ... ]. The first column has the start time of each window,
    %   and the second column has the end time of each window.
    %the output of windows2mask is a single column vector of 0/1 logical
    %values that serves as a mask for the given windows. Mask will be 1
    %where the values of time fall within any of the given windows, and 0
    %otherwise.     
    
    if size(windows,2)~=2
        error('windows2mask:windowsNotColumnMatrix','the aray of windows must be a column matrix with the first column containing the start of each window, and the scond column containing the end of each window')
    end
    if isrow(time)
        error('windows2mask:timeNotColumnVector','the time input to windows2mask must be a column vector')
    end
    
    
    mask=false(size(time));
    for i=1:size(windows,1)
        
        mask(time>=windows(i,1) & time <=windows(i,2))=true;
    end
    return
end