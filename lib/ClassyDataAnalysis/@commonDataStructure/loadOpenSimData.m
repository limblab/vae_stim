function loadOpenSimData(cds,folderPath,dataType, deepLabCutFlag)
    %this is a method of the cds class and should be stored in the
    %@commonDataStructure folder with the other class methods.
    %
    %attempts to load Open Sim data from the given path in the cds.
    %
    % dataType can currently be one of:
    %   'joint_ang'
    %   'joint_vel'
    %   'joint_acc'
    %   'joint_dyn'
    %   'muscle_len'
    %   'muscle_vel'
    %   'hand_pos'
    %   'hand_vel'
    %   'hand_acc'
    %   'elbow_pos'
    %   'elbow_vel'
    %   'elbow_acc'
    %
    %   Given new deeplabcut pipeline, it makes more sense to shift the
    %   times here rather than in the opensim file. Include a motion
    %   tracking flag to know to search in the analog file.
    
    if ~strcmp(folderPath(end),filesep)
        folderPath=[folderPath,filesep];
    end
    
    prefix=cds.meta.rawFileName;
    if ~iscell(prefix)
        prefix={prefix};
    end
    foundFiles={};
    for i=1:numel(prefix)
        %find and strip extensions if present
        extLoc=max(strfind(prefix{i},'.'));
        if ~isempty(extLoc)
            prefix{i}=prefix{i}(1:extLoc-1);
        end
        
        % set the right file and header postfixes
        switch(dataType)
            case 'joint_ang'
                postfix = '_Kinematics_q.sto';
                header_post = '_ang';
            case 'joint_vel'
                postfix = '_Kinematics_u.sto';
                header_post = '_vel';
            case 'joint_acc'
                % postfix = '_Kinematics_dudt.sto';
                % header_post = '_acc';
                error('loadOpenSimData:unsupportedDataType','Joint accelerations are currently unsupported until dynamics are added to modeling')
            case 'joint_dyn'
                postfix = '_Dynamics_q.sto';
                header_post = ''; % already postfixed by 'moment'
            case 'muscle_len'
                postfix = '_MuscleAnalysis_Length.sto';
                header_post = '_len';
            case 'muscle_vel'
                % temporary until Fiber_velocity file is fixed: take
                % gradient of muscle lengths
                % postfix = '_MuscleAnalysis_FiberVelocity.sto';
                postfix = '_MuscleAnalysis_Length.sto';
                header_post = '_muscVel';
            case 'hand_pos'
                postfix = '_PointKinematics_hand_pos.sto';
                header_post = '_handPos';
            case 'hand_vel'
                postfix = '_PointKinematics_hand_vel.sto';
                header_post = '_handVel';
            case 'hand_acc'
                postfix = '_PointKinematics_hand_acc.sto';
                header_post = '_handAcc';
            case 'elbow_pos'
                postfix = '_PointKinematics_elbow_pos.sto';
                header_post = '_elbowPos';
            case 'elbow_vel'
                postfix = '_PointKinematics_elbow_vel.sto';
                header_post = '_elbowVel';
            case 'elbow_acc'
                postfix = '_PointKinematics_elbow_acc.sto';
                header_post = '_elbowAcc';
            otherwise
                error('loadOpenSimData:invalidDataType', 'Data type must be one of {''joint_ang'', ''joint_vel'', ''joint_dyn'', ''muscle_len'',''muscle_vel'',''hand_pos'',''hand_vel'',''hand_acc'',''elbow_pos'',''elbow_vel'',''elbow_acc''}')
        end
        fileNameList = {[folderPath,prefix{i},postfix]};
%         fileNameList={[folderPath,prefix{i},'_Kinematics_q.sto'];...
%             [folderPath,prefix{i},'_MuscleAnalysis_Length.sto']};
%             [folderPath,prefix{i},'_MuscleAnalysis_Length.sto'];...
%             [folderPath,prefix{i},'_Dynamics_q.sto']};
        for j=1:numel(fileNameList)
            foundList=dir(fileNameList{j});
            if ~isempty(foundList)
                %load data from file into table 'kin':
                fid=fopen(fileNameList{j});
                try
                %loop through the header till we find the first row of data:
                tmpLine=fgetl(fid);
                %check for correct file given dataType
                switch(dataType)
                    case 'joint_ang'
                        if ~strcmp(tmpLine,'Coordinates')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'joint_vel'
                        if ~strcmp(tmpLine,'Speeds')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'joint_acc'
                        if ~strcmp(tmpLine,'Accelerations')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'joint_dyn'
                        if ~strcmp(tmpLine,'Inverse Dynamics Generalized Forces')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'muscle_len'
                        if ~strcmp(tmpLine,'Length')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'muscle_vel'
                        % temporary until Fiber_velocity file is fixed: take
                        % gradient of muscle lengths
                        if ~strcmp(tmpLine,'Length')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'hand_pos'
                        if ~strcmp(tmpLine,'PointPosition')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'hand_vel'
                        if ~strcmp(tmpLine,'PointVelocity')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'hand_acc'
                        if ~strcmp(tmpLine,'PointAcceleration')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'elbow_pos'
                        if ~strcmp(tmpLine,'PointPosition')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'elbow_vel'
                        if ~strcmp(tmpLine,'PointVelocity')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    case 'elbow_acc'
                        if ~strcmp(tmpLine,'PointAcceleration')
                            error('loadOpenSimData:wrongFile',['Header in analysis file ' fileNameList{j} ' is incorrect'])
                        end
                    otherwise
                        error('loadOpenSimData:invalidDataType', 'Data type must be one of {''joint_ang'', ''joint_vel'', ''joint_dyn'', ''muscle_len'',''muscle_vel'',''hand_pos'',''hand_vel'',''hand_acc'',''elbow_pos'',''elbow_vel'',''elbow_acc''}')
                end
                while ~strcmp(tmpLine,'endheader')
                    if ~isempty(strfind(tmpLine,'nRows'))
                        nRow=str2double(tmpLine(strfind(tmpLine,'=')+1:end));
                    elseif ~isempty(strfind(tmpLine,'nColumns'))
                        nCol=str2double(tmpLine(strfind(tmpLine,'=')+1:end));
                    elseif ~isempty(strfind(tmpLine,'inDegrees'))
                        if ~isempty(strfind(tmpLine,'yes'))
                            unitLabel='deg';
                        else
                            unitLabel='rad';
                        end
                    end
                    tmpLine=fgetl(fid);
                end
                header=strsplit(fgetl(fid));
                %convert 'time' to 't' to match cds format:
                idx=find(strcmp(header,'time'),1);
                if isempty(idx)
                    %look for a 't' column
                    idx=find(strcmp(header,'t'),1);
                    if isempty(idx)
                        error('loadOpenSimData:noTime',['could not find a time column in the file: ', kinFileName])
                    end
                else
                    %convert 'time into 't'
                    header{idx}='t';
                end
                
                % look for badly named headers (mostly for point
                % kinematics)
                header_aliases = {'state_0','Y';'state_1','Z';'state_2','X'};
                for header_ctr = 1:size(header_aliases,1)
                    state_idx=find(strcmp(header,header_aliases{header_ctr,1}),1);
                    if ~isempty(state_idx)
                        header{state_idx} = header_aliases{header_ctr,2};
                    end
                end
                
                % convert header to specify type of data
                other_idx = find(~strcmp(header,'t'));
                for header_ctr = 1:length(other_idx)
                    header{other_idx(header_ctr)} = [header{other_idx(header_ctr)} header_post];
                end
                
                scanned_input = fscanf(fid,repmat('%f',[1,nCol]));
                a=reshape(scanned_input,[nCol,nRow])';
                %sanity check time:
                SR=mode(diff(a(:,1)));
                if size(a,1)~=round((1+ (max(a(:,1))-min(a(:,1)))/SR))
                    warning('loadOpenSimData:badTimeSeries',['the timeseries in the detected opensim data is missing time points. expected ',num2str((1+ (max(a(:,1))-min(a(:,1)))/SR)),' points, found ',num2str(size(a,1)),' points'])
                    disp('data will be interpolated to reconstruct missing points')
                    cds.addProblem('kinect data has missing timepoints, data in the cds has been interpolated to reconstruct them')
                end
                
                % resample data at uniform sampling rate
                [resampData,timevec] = resample(a(:,2:end),a(:,1));
                
                % Temporary until fiber velocity file is fixed: take
                % gradient for muscle velocity
                if strcmp(dataType,'muscle_vel')
                    for muscle_ctr = 1:size(resampData,2)
                        grad_interpData(:,muscle_ctr) = gradient(resampData(:,muscle_ctr),timevec);
                    end
                    resampData = grad_interpData;
                end

                kin=array2table([timevec,resampData],'VariableNames',header);
                unitsLabels=[{'s'},repmat({unitLabel},[1,nCol-1])];
                kin.Properties.VariableUnits=unitsLabels;
                %find sampling rate and look for matching rate in analog data:
                cds.analog{end+1}=kin;
                foundFiles=[foundFiles;fileNameList(j)];
                catch ME
                    fclose(fid);
                    rethrow(ME)
                end
                
                fclose(fid);
            else
                warning('loadOpenSimData:fileNotFound','The specified file: %s was not found. Check file name and try again.',fileNameList{j});
            end
        end
           
    end
    
    % set new data window
    cds.setDataWindow()
    
    logStruct=struct('folder',folderPath,'fileNames',foundFiles);
    evntData=loggingListenerEventData('loadOpenSimData',logStruct);
    notify(cds,'ranOperation',evntData)
end