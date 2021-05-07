function [dataD]=decimateData(data,fc)
    %[dataDecimated]=decimateData(data,fc)
    %decimateData takes in a time series column matrix 'data' where the 
    %first column is time, and each subsequent column is a column of data
    %samples. decimate data also takes a filterConfig object 'fc', which
    %specifies how to decimated the data, including the desired sample rate
    %and the filter specifications (see the filterConfig class for details)
    %
    %decimateData assumes that the samples are spaced evenly in time, and 
    %will not produce reliable results if time sampling is jittery. 
    %decimateData estimates the sampling rate of the original data using 
    %the mode of the inter-sample periods, upsamples to a common multiple
    %of the desired and current sample frequencies, filters using a
    %butterworth filter, and then decimates to the desired frequency. Note
    %that the filter cutoff in the filterConfig object should be less than
    %half the final sample frequency to avoid aliasing.
    
    
    %kin=kinematics/kinetics in column vector, first column must be time
    DF=1/fc.sampleRate;%DF= desired frequency of decimated signal
    SF=mode(diff(data(:,1)));%SF= sample frequency of data
    %upsample the signal:
    %dataD=interp1(data(:,1),data(:,2:end),t_upsamp);
    %upsample the data so that we can use simple decimation rather than
    %interpolation to downsample
    [p,q]=rat(DF/SF,.0001);
    dataD=upsample(data(:,2:end),q);%fills with zeros rather than interpolating, which is technically more correct. If q is 1 returns the original vector, if q is 2, inserts 1 zero between each pair of points etc.
    %lowpass filter using filter_config
    %convert our cutoff into a fraction of the upsampled frequency. SF is
    %in s, filter_config.cutoff is expcted in hz
    %butter builds a filter with a cutoff using 1/2 the specified frequency for some reason so we multiply our cutoff by 2
    cutoff=fc.cutoff*(SF/q)*2;
    [b, a] = butter(fc.poles, cutoff);%butterworth uses cutoff/(samplerate/2), or 2*cutoff/samplerate to specify cutoff
    dataD=filtfilt(b,a,dataD);
    dataD=downsample(dataD,p);
    
    % get time vector
    resamp_vec = ones(size(data,1),1);
    resamp_vec = downsample(upsample(resamp_vec,q),p);
    dataT=downsample(upsample(data(:,1),q),p);
    dataT=interp1(find(resamp_vec>0),dataT(resamp_vec>0),(1:length(dataT))');
    
    % get rid of extrapolated points
    extrap_idx = isnan(dataT);
    dataT(extrap_idx) = [];
    dataD(extrap_idx) = [];
    
    dataD=[dataT dataD];
end