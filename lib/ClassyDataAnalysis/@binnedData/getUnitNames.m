function [unitNames,varargout]=getUnitNames(binned)
    %helper method to get a cell array listing the unit names of all units
    %in the binnedData object. Returns a cell array with the column names
    %of firing rate data. optionally returns a boolean mask for those
    %columns of the binned.data table.
    unitMask=~cellfun(@(x)isempty(strfind(x,'CH')),binned.data.Properties.VariableNames) & ~cellfun(@(x)isempty(strfind(x,'ID')),binned.data.Properties.VariableNames);
    unitNames=binned.data.Properties.VariableNames(unitMask);
    if nargout>2
        error('getUnitNames:tooManyOutputs','getUnitNames provides up to 2 outputs')
    elseif nargout==2
        varargout{1}=unitMask;
    end
end