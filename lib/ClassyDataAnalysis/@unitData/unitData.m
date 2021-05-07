classdef unitData < matlab.mixin.SetGet
    properties(SetAccess = public,GetAccess = public, SetObservable = true)
        appendConfig
    end
    properties (SetAccess = protected,GetAccess = public, SetObservable = true)
        data%main data table
    end
    events
        appended
        removedSorting
    end
    methods (Static = true)
        %constructor
        function units=unitData()
            units.data=struct('chan',[],'ID',[],'array',{},'spikes',cell2table(cell(0,2),'VariableNames',{'ts','wave'}));
            units.appendConfig=struct('method','shape','threshold',.05,'SNRThreshold',2,'thresholdPoint',10,'default','unsorted');
        end
    end
    methods
        %setter methods
        function set.data(units,data)
            f=@(x) ~isa(x,'table');
            f2=@(x) size(x,2)~=2;
            f3=@(x) isempty(find(strcmp('ts',x.Properties.VariableNames),1));
            f4=@(x) isempty(find(strcmp('wave',x.Properties.VariableNames),1));
            
            if isempty(data) 
                units.data=data;
            elseif ~isstruct(data)
                error('unitData:badFormat','Units must be a struct')
            elseif ~isfield(data,'chan') ||  ~isnumeric([data(:).chan])
                error('unitData:badchanFormat','units must have a field called chan that contains a numeric array of channel numbers')
            elseif ~isfield(data,'ID') || ~isnumeric([data(:).ID])
                error('unitData:badIDFormat','units must have a field called ID that contains a numeric array of the ID numbers')
            elseif ~isfield(data,'array') ||  ~iscellstr({data.array})
                error('unitData:badarrayFormat','units must have a field called array that contains a cell array of strings, where each string specifies the array on which the unit was collected')
            elseif ~isfield(data,'monkey') || ~iscellstr({data.monkey})
                error('units:badMonkeyFormat','data must have a field called array that contains a cell array of strings, where each string specifies the monkey on which the unit was collected')
            elseif ~isfield(data,'spikes') 
                error('unitData:missingspikes','units must have a field called spikes containing tables of the spike times and waveforms')
            elseif ~isempty({data.spikes}) && (~isempty(find(cellfun(f,{data.spikes}),1)) ...
                    || ~isempty(find(cellfun(f2,{data.spikes}),1)) ...
                    || ~isempty(find(cellfun(f3,{data.spikes}),1)) ...
                    || ~isempty(find(cellfun(f4,{data.spikes}),1)) )
                error('unitData:badFormat','all elements in units.spikes must be tables with 2 columns: ts and wave. ts contains the timestamps of each wave, and wave contains the snippet of the threshold crossing')
            else
                units.data=data;
            end
        end
        function set.appendConfig(units,adata)
            if ~isfield(adata,'method') || ~ischar(adata.method) || ~( strcmp(adata.method,'shape') || strcmp(adata.method,'ISI') || strcmp(adata.method,'shapeISI') || strcmp(adata.method,'number'))
                error('unitData:badAppendMethod','unitData.appendConfig must have a method field, which must contain a string with one of the following values: shape, ISI, shapeISI, or number')
            elseif strcmp(adata.method,'ISI') || strcmp(adata.method,'shapeISI') 
                error('unitData:ISIStatsNotImplemented','computation of ISI statistics for merging units is not yet implemented. Please either use shape, or implement ISI stats.')
            elseif ~isfield(adata,'threshold') || ~isnumeric(adata.threshold)
                error('unitData:badAppendThreshold','unitData.appendConfig must have a threshold field containing the p-value deliniating different vs same')
            elseif ~isfield(adata,'SNRThreshold') || ~isnumeric(adata.SNRThreshold)
                error('unitData:badAppendSNRThreshold','unitData.appendConfig must have a field named SNRThreshold with a number indicating the minmum allowable value for SNR before unit sorting is rejected')
            elseif ~isfield(adata,'default') || ~ischar(adata.default) || ~(strcmp(adata.default,'unsorted') || strcmp(adata.default,'delete') || strcmp(adata.default,'invalid'))
                error('unitData:badAppendDefault','unitData.appendConfig must have a default field containing a string that defines what to do when units do not match. allowed values are unsorted, invalid or delete')
            else
                units.appendConfig=adata;
            end
        end
    end
    methods (Static = false)
        %general methods
        appendData(units,data,varargin)
        removeSorting(units,varargin)
        deleteUnsorted(units)
        unitName=getUnitName(units,unitNum)
        varargout=PESTH(units,eventTimes,preEventWindow,postEventWindow,unitNum,varargin)
    end
end