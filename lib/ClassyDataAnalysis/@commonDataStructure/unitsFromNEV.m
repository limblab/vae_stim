function unitsFromNEV(cds,opts)
    %takes a cds handle an NEVNSx object and an options structure and
    %populates the units field of the cds
    unitList = unique([cds.NEV.Data.Spikes.Electrode;cds.NEV.Data.Spikes.Unit]','rows');
    %establish id table
    if isfield(opts,'array')
        array=opts.array;
    else
        warning('unitsFromNEV:noArrayName','The user did not specify an array name for this data. Please re-load the data and specify an array name')
        cds.addProblem('arrayNameUnknown: the user did not specify a name for the array this data comes from')
        array='?';
    end
    if isfield(opts,'monkey')
        monkey=opts.monkey;
    else
        warning('unitsFromNEV:noArrayName','The user did not specify an monkey name for this data. Please re-load the data and specify an array name')
        cds.addProblem('monkeyNameUnknown: the user did not specify a name for the monkey this data comes from')
        monkey='?';
    end
    %if we already have unit data, check that our new units come from a
        %different source so that we don't get duplicate entries
    if ~isempty(cds.units) && ~isempty(unitList) && ~isempty(find(strcmp({cds.units.array},opts.array),1,'first'))
        error('unitsFromNEV:sameArrayName','the cds and the current data have the same array name, which will result in duplicate entries in the units field. Re-load one of the data files using a different array name to avoid this problem')
    end
    %try loading a mapfile:
    noMap=true;
    if isfield(opts,'mapFile') && ~isempty(opts.mapFile)
        try
%             arrayMap={};
%             fid=fopen(opts.mapFile,'r');
%             inheader=true;
%             while ~feof(fid)
%                 tline=fgets(fid);
%                 if isempty(deblank(tline))
%                     continue
%                 end
%                 %skip header lines:
%                 if inheader && ~isempty(strfind(tline,'Cerebus mapping'))
%                     %this is our last header line, set the flag false and
%                     %continue to the next line:
%                     inheader=false;
%                     continue
%                 elseif inheader
%                     continue
%                 end
%                 if strcmp(tline(1:2),'//')
%                     %this should be our column labels
%                     colLabels=textscan(tline(3:end),'%s')';%strsplit kind of works but leaves an empty string at the end that screws up the rest of processing
%                     colLabels=colLabels{1};
%                     rowNumCol=strcmp(colLabels,'row');
%                     colNumCol=strcmp(colLabels,'col');
%                     bankCol=strcmp(colLabels,'bank');
%                     pinCol=strcmp(colLabels,'elec');
%                     labelCol=strcmp(colLabels,'label');
%                     continue
%                 end
%                 %if we got to this point we are on an actual data line:
%                 tmp=textscan(tline,'%s');
%                 tmp=tmp{1};
%                 if numel(tmp)~=5
%                     error('unitsFromNEV:unexpectedMapFormat',['Was expecting to find exactly 5 columns in the mapfile, instead found: ',num2str(numel(tmp))])
%                 end
%                 if exist('colLabels','var')
%                     %we have a newer map file with actual column labels
%                     %use the labels to assign values
%                     rowNum=str2num(tmp{rowNumCol})+1;
%                     colNum=str2num(tmp{colNumCol})+1;
%                     pin=str2num(tmp{pinCol});
%                     bank=char(tmp{bankCol});
%                     label=char(tmp{labelCol});
%                 else
%                     %we have an older file with no labels
%                     warning('unitsFromNEV:oldMapfile','This mapfile does not have column labels. This is typical of older mapfiles. The mapfile data has been processed assuming the older Blackrock format, but this may cause issues for custom mapfiles or other oddities')
%                     rowNum=str2num(tmp{1})+1;
%                     colNum=str2num(tmp{2})+1;
%                     pin=str2num(tmp{4});
%                     bank=char(tmp{3});
%                     label=char(tmp{5});
%                 end
%                 switch bank
%                     case 'A'
%                         chan=pin;
%                     case 'B'
%                         chan=pin+32;
%                     case 'C'
%                         chan=pin+64;
%                     case 'D'
%                         chan=pin+96;
%                     otherwise
%                         error('unitsFromNEV:badBankLabel',['unitsFromNEV is not configured to handle arrays with bank label: ',bank])
%                 end
%                 arrayMap=[arrayMap;{chan,pin,rowNum,colNum,bank,label}];
%             end
%             arrayMap=cell2table(arrayMap,'VariableNames',{'chan','pin','row','col','bank','label'});
            arrayMap=loadMapFile(opts.mapFile);
            noMap=false;
        catch ME
            noMap=true;
        end
    end
    if noMap
        if exist('ME','var')
            problemData.description='tried to load mapfile and failed';
            problemData.error=ME;
        else
            problemData.description='no map file was passed';
        end
        cds.addProblem('No Map file. Electrode locations and bank ID are not available in the units structure',problemData);
    end
    NEVChanList=[cds.NEV.ElectrodesInfo.ElectrodeID]';
    %initialize struct array:
    units=struct(   'chan',cell(numel(unitList),0),...
                    'ID',cell(numel(unitList),0),...
                    'wellSorted',cell(numel(unitList),0),...this is a stub as testSorting can't be run till the whole units field is populated
                    'spikes',repmat( cell2table(cell({0,2}),'VariableNames',{'ts','wave'}),numel(unitList),1),...
                    'monkey',cell(numel(unitList),0),...
                    'array',cell(numel(unitList),0),...
                    'bank',cell(numel(unitList),0),...
                    'pin',cell(numel(unitList),0),...
                    'label',cell(numel(unitList),0),...
                    'lowThreshold',cell(numel(unitList),0),...
                    'highThreshold',cell(numel(unitList),0),...
                    'lowPassCorner',cell(numel(unitList),0),...
                    'lowPassOrder',cell(numel(unitList),0),...
                    'lowPassType',cell(numel(unitList),0),...
                    'highPassCorner',cell(numel(unitList),0),...
                    'highPassOrder',cell(numel(unitList),0),...
                    'highPassType',cell(numel(unitList),0)...
                    );
    %loop through and unit entries for each unit
    for i = 1:size(unitList,1)        
        units(i).chan=unitList(i,1);
        units(i).ID=unitList(i,2);
        units(i).wellSorted=false;
        units(i).spikes=table(...timestamps for current unit from the NEV:
                                     [double(cds.NEV.Data.Spikes.TimeStamp(cds.NEV.Data.Spikes.Electrode==unitList(i,1) & ...
                                        cds.NEV.Data.Spikes.Unit==unitList(i,2)))/30000]',... 
                                    ...waves for the current unit from the NEV:    
                                    double(cds.NEV.Data.Spikes.Waveform(:,cds.NEV.Data.Spikes.Electrode==unitList(i,1) ...
                                    &  cds.NEV.Data.Spikes.Unit==unitList(i,2))'),...
                                    'VariableNames',{'ts','wave'});
        %check for resets in time vector
        idx=cds.skipResets(units(i).spikes.ts);
        if ~isempty(idx) && idx>1
            %if there were resets, remove everything before the resets
            units(i).spikes=units(i).spikes(idx+1:end,:);
        end
        %now fill out the info structure for the unit
        %find the appropriate row in NEV.ElectrodesInfo to get unit info:
        NEVidx=find(NEVChanList==units(i).chan);
        units(i).monkey=monkey;
        units(i).array=array;
        units(i).bank=cds.NEV.ElectrodesInfo(NEVidx).ConnectorBank;
        units(i).pin=cds.NEV.ElectrodesInfo(NEVidx).ConnectorPin;
        tmpLabel=cds.NEV.ElectrodesInfo(NEVidx).ElectrodeLabel;
        if isempty(tmpLabel)
            tmpLabel = num2str(i);
        end
        tmpLabel=strtrim(tmpLabel(int8(tmpLabel)>0));%get rid of null characters that pad the end of labels
        units(i).label=reshape(tmpLabel,[1,numel(tmpLabel)]);%deal with the fact that blackrock imports the labels as COLUMN arrays of characters for some obtuse reason...
        units(i).lowThreshold=cds.NEV.ElectrodesInfo(NEVidx).LowThreshold;
        units(i).highThreshold=cds.NEV.ElectrodesInfo(NEVidx).HighThreshold;
        units(i).lowPassCorner=cds.NEV.ElectrodesInfo(NEVidx).LowFreqCorner;
        units(i).lowPassOrder=cds.NEV.ElectrodesInfo(NEVidx).LowFreqOrder;
        units(i).lowPassType=cds.NEV.ElectrodesInfo(NEVidx).LowFilterType;
        units(i).highPassCorner=cds.NEV.ElectrodesInfo(NEVidx).HighFreqCorner;
        units(i).highPassOrder=cds.NEV.ElectrodesInfo(NEVidx).HighFreqOrder;
        units(i).highPassType=cds.NEV.ElectrodesInfo(NEVidx).HighFilterType;
        
        if noMap
            units(i).rowNum=nan;
            units(i).colNum=nan;
        else
            %find the correct row of our arrayMap:
            mapidx=find(arrayMap.chan==units(i).chan,1);
            %copy data from the map into the current unit entry:
            units(i).rowNum=arrayMap.row(mapidx);
            units(i).colNum=arrayMap.col(mapidx);
        end
        
    end
    if isempty(cds.units)
        cds.units=units;
    else
        cds.units=[cds.units,units];
    end
%    unitscds.testSorting; %tests each sorted unit to see if it is well-separated from background and other units on the same channel
    opData.array=array;
    opData.numUnitsAdded=size(unitList,1);
    evntData=loggingListenerEventData('unitsFromNEV',opData);
    notify(cds,'ranOperation',evntData)
end