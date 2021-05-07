function writeSessionSummary(cds) 
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %write_session_summary(cds)
    %this method parses the meta field of the cds and writes a flat text
    %file for parsing by humans and scripts. It will also write a line to
    %the limblabDataHistory.xlsx file containing all the same information
    %so that people can sort/organize the data in excel
    
    %% get path to fsmres folder based on system architecture:
    if ispc
        folderpath='fsmresfiles.fsm.northwestern.edu\fsmresfiles\Basic_sciences\Phys\L_MillerLab\data\cds';
    elseif ismac || isunix
        [~,folderpath]=unix('find /*/fsmresfiles/data* -maxdepth 1 -type d -name cds');
        folderpath=strtrim(folderpath);
    end
    %% build paths to various files:
    %excel database:
    fnameDH=[folderpath,filesep,'limblabDataHistory.xlsx'];
    %matlab backup database
    fnameBackup=[folderpath,filesep,'historyBackupMatlab.mat'];
    %summary file:
    summaryFolder=[folderpath,filesep,'summaryFiles',filesep,cds.meta.monkey];
    if exist(summaryFolder,'file')~=7
        mkdir(summaryFolder)
    end
    fnameSummary=[summaryFolder,filesep,cds.meta.cdsName,'.txt'];
    %cds file:
    cdsFolder=[folderpath,filesep,'dataFiles',filesep,cds.meta.monkey];
    if exist(cdsFolder,'file')~=7
        mkdir(cdsFolder)
    end
    fnameCds=[cdsFolder,filesep,cds.meta.cdsName,'.m'];
    %% build list of labels that we will want to write to the excel database:
    dataLabels={'Source File',...
                    'dateTime',...
                    'cdsName',...
                    'processedTime',...
                    'cdsVersion',...
                    'monkey',...
                    'array',...
                    'task',...
                    'ranBy',...
                    'hasUnits',...
                    'hasKinematics',...
                    'hasForce',...
                    'hasEmg',...
                    'hasLfp',...
                    'hasAnalog',...
                    'hasTriggers',...
                    'hasBumps',...
                    'hasChaoticLoad',...
                    'hasSorting',...
                    'numSorted',...
                    'numWellSorted',...
                    'numDualUnits',...
                    'numTrials',...
                    'numReward',...
                    'numAbort',...
                    'numFail',...
                    'numIncomplete',...
                    'percentStill',...
                    };
    try
        %% open the limblabDataHistory.xlsx file and backup mat and compare:
        [~,~,dataHistory]=xlsread(fnameDH,'dataHistory');
        if isempty(dir(fnameBackup))
            backupDH=[];
        else
            load(fnameBackup);
        end
        %% 
        if ~isequal(backupDH,dataHistory)
            reWriteFlag=true;
        else
            reWriteFlag=false;
        end
        colNames=dataHistory(1,:);
        excelLineNum=find(strcmp(cds.meat.cdsName,dataHistory(:,strcmp('cdsName',colNames))));
        excelData=cell(1,length(colNames));
        if ~isempty(excelLineNum)
            excelLineNum=size(dataHistory,1)+1;
        else
            warning('writeSessionSummary:cdsHistoryExists','summary data for a cds with this name already exists. That summary will be overwritten')
        end
        %open the text file for writing:
        fhandle=fopen(fnameSummary,'w');
        %loop through the dataLabels and write data for each to file. Also put
        %into cell array for writing to excel file:
        for i=1:numel(dataLabels)
            itemName=dataLabels{i};
            itemData=cds.meta.(itemName);
            if ischar(itemData)
                fprintf(fhandle,'%s:\t%s\n\r',itemName,itemData);
            elseif iscellstr(itemData)
                fprintf(fhandle,'%s:\t%s\n\r',itemName,strjoin(itemData,'::'));
            else
                fprintf(fhandle,'%s:\t%s\n\r',itemName,itemData);
            end
            excelData{strcmp(itemName,colNames)}=itemData;
        end

        %write line to limblabDataHistory.xlsx:
        if reWriteFlag
            status=xlswrite(fnameDH,[dataHistory;excelData],'dataHistory');
        else
            status=xlswrite(fnameDH,excelData,'dataHistory',excelLineNum);
        end
        if ~status
            warning('surrary data not written to limblabDataHistory.xlsx')
        end    

        evntData=loggingListenerEventData('writeSessionSummary',[]);
        notify(cds,'ranOperation',evntData)
    catch ME
        warning('writeSessionSummary:autoSaveFailed','failed to write session summary to fsmresfiles. If your script does not otherwise save the cds, this data may be lost! Details on error below')
        for i=1:numel(ME.stack)
            disp([message;{ME.stack(i).file;['line: ' num2str(ME.stack(i).line)]}]); 
        end
    end
        
end