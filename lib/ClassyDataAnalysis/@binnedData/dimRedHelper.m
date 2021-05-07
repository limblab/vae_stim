function [ dat ] = dimRedHelper( binned )
    %sanity check sampling rate
    if roundTime(mode(diff(binned.data.t)))~=.001
        error('dimRedHelper:badSampleRate','the core dimensionality reduction code uses 1ms bins with binary flags for the existance of spikes. This binnedData structure was built using data at a different sample rate. Please re-compute the binnedData with a 1ms bin size')
    end
    
    if isempty(binned.dimReductionConfig.units)
        %get a list of all the units:
       unitList=binned.getUnitNames();
    else
       unitList=binned.dimReductionConfig.units;
    end
    dat(i).trialId = binned.dimReductionConfig.trials(i);
    dat(i).spikes(:,:) = table2array(binned.data(find(binned.data.t>windows(i,1),1):find(binned.data.t>windows(i,2),1), 12:end))'./1000;
    
end

