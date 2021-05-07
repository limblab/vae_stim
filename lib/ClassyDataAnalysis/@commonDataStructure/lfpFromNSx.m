function lfpFromNSx(cds,opts)
    %takes a handle to a cds object and an NEVNSx structure, and populates
    %the LFP field of the cds
    %
    % opts.array specifies the array name
    
    %get list of channels that have LFP data:
    lfpList=find(~cellfun('isempty',strfind(lower(cds.NSxInfo.NSx_labels),'elec')));
    if ~isempty(lfpList)
        %get list of frequencies at which 
        freq=unique(cds.NSxInfo.NSx_sampling(lfpList));
        if length(freq)>1
            error('lfpFromNSx:multipleFrequencies','LFP data was collected at multiple frequencies and lfpFromNSx is not designed to handle this. Please update lfpFromNSx, or alias the lfp frequencies and handle this manually')
        end
        
        subset=find(cds.NSxInfo.NSx_sampling(lfpList)==freq);
        achanIndex=lfpList(subset);
        lfp = [];%will preallocate below once we know how long the data is
        switch freq
            case 500
                nsLabel='NS1';
                timestamp=double(cds.NS1.MetaTags.Timestamp);
                timeRes=double(cds.NS1.MetaTags.TimeRes);
            case 1000
                nsLabel='NS2';
                timestamp=double(cds.NS2.MetaTags.Timestamp);
                timeRes=double(cds.NS2.MetaTags.TimeRes);
            case 2000
                nsLabel='NS3';
                timestamp=double(cds.NS3.MetaTags.Timestamp);
                timeRes=double(cds.NS3.MetaTags.TimeRes);
            case 10000
                nsLabel='NS4';
                timestamp=double(cds.NS4.MetaTags.Timestamp);
                timeRes=double(cds.NS4.MetaTags.TimeRes);
            case 30000
                nsLabel='NS5';
                timestamp=double(cds.NS5.MetaTags.Timestamp);
                timeRes=double(cds.NS5.MetaTags.TimeRes);
            otherwise
                error('analogFromNSx:unexpectedFrequency',['this function is not set up to handle data with collection frequency: ',num2str(freq)])
        end
        for c=1:length(achanIndex)
            lfp{c+1}= double(cds.(nsLabel).Data(cds.NSxInfo.NSx_idx(achanIndex(c)),:))';
        end    
        %now stick time on the front of lfp
        lfp{1}=roundTime((0:size(lfp{2},1)-1)' / cds.NSxInfo.NSx_sampling(achanIndex(c)) + timestamp/timeRes);
        labels=[{'t'},reshape(cds.NSxInfo.NSx_labels(lfpList(subset)),1,numel(cds.NSxInfo.NSx_labels(lfpList(subset))))];
        
        if ~isempty(lfp)
            % add array name to LFP label
            for idx_lfp = 2:length(labels)
                labels{idx_lfp} = [opts.array labels{idx_lfp}];
            end
            
            %convert lfp array to table:
            lfp=table(lfp{:},'VariableNames',labels);            
            lfp.Properties.VariableUnits=[{'s'},repmat({'mV'},1,numel(lfpList))];
            lfp.Properties.VariableDescriptions=[{'time'},repmat({'LFP in mV'},1,numel(lfpList))];
            lfp.Properties.Description='Filtered LFP in raw collection voltage. Voltage scale is presumed to be mV';
            if isempty(cds.lfp)
                set(cds,'lfp',lfp);
            elseif ~isempty(lfp)
                set(cds,'lfp',mergeTables(cds.lfp,lfp));
            end
            evntData=loggingListenerEventData('lfpFromNSx',[]);
            notify(cds,'ranOperation',evntData)
        end
    end    
end