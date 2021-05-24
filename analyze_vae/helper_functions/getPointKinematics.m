function [point_kin] = getPointKinematics(input_data)

    % extract inputs
    data_dir = input_data.data_dir;
    settings_path = input_data.settings_path;
    settings_fname = input_data.settings_fname;

    % generate mot file
    writeMotFile(input_data.mot_path, input_data.mot_name, input_data.t, input_data.joint_ang, input_data.joint_names, input_data.in_deg); % in_deg = 1;

    
    % Pull in the modeling classes straight from the OpenSim distribution
    import org.opensim.modeling.*

    % Go to the folder in the subject's folder where IK Results are
    ik_results_folder = fullfile(data_dir, 'IKResults');

    % specify where setup files will be printed.
    setupfiles_folder = fullfile(data_dir, 'AnalyzeSetup');

    % specify where results will be printed.
    results_folder = fullfile(data_dir, 'AnalyzeResults');

    % Get and operate on the files
    genericSetupForAn = [settings_path filesep settings_fname];
    
    % make analysis tool
    analyzeTool = AnalyzeTool(genericSetupForAn);

    % get the file names that match the ik_reults convention
    % this is where consistent naming conventions pay off
    trialsForAn = dir([input_data.mot_path filesep, input_data.mot_name]);
    nTrials =length(trialsForAn);

    for trial= 1:nTrials
        % get the name of the file for this trial
        motIKCoordsFile = trialsForAn(trial).name;

        % create name of trial from .trc file name
        name = regexprep(motIKCoordsFile,'.mot','');

        % get .mot data to determine time range
        motCoordsData = Storage(fullfile(ik_results_folder, motIKCoordsFile));

        % for this example, column is time
        initial_time = motCoordsData.getFirstTime();
        final_time = motCoordsData.getLastTime();

        analyzeTool.setName(name);
        analyzeTool.setResultsDir(results_folder);
        analyzeTool.setCoordinatesFileName(fullfile(ik_results_folder, motIKCoordsFile));
        analyzeTool.setInitialTime(initial_time);
        analyzeTool.setFinalTime(final_time);   

        outfile = ['Setup_Analyze_.xml'];
        analyzeTool.print(fullfile(setupfiles_folder, outfile));

        analyzeTool.run();
        fprintf(['Performing IK on cycle # ' num2str(trial) '\n']);

    end
    
    
    % load in point_kin and output
    % for hand and elbow, get acc, vel and pos.
    point_kin = []; 

    locs = {'elbow','hand'};
    vars = {'acc','vel','pos'};
    
    for loc = locs
        for var = vars
            var_name = [loc{1},'_',var{1}];
            temp_kin = [];

            % get file
            fname = [name,'_PointKinematics_',loc{1},'_',var{1},'.sto'];
            file_full_path = [results_folder filesep fname];
            
            % read in data
            fid=fopen(file_full_path);
            
            % loop through till end of header
            tmpLine=fgetl(fid);
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
            
            % get variable names and clean up badly named headers
            header=strsplit(fgetl(fid));
            header_aliases = {'state_0','Y';'state_1','Z';'state_2','X'};
            for header_ctr = 1:size(header_aliases,1)
                state_idx=find(strcmp(header,header_aliases{header_ctr,1}),1);
                if ~isempty(state_idx)
                    header{state_idx} = header_aliases{header_ctr,2};
                end
            end
            
            % get rest of file
            temp_kin = fscanf(fid,repmat('%f',[1,nCol]));
            temp_kin=reshape(temp_kin,[nCol,nRow])';
            fclose(fid);
            
            % make table with correct column names
            temp_kin = array2table(temp_kin,'VariableNames',header);
            
            % store in point_kin with appropriate name
            point_kin.(var_name) = temp_kin;
        end
    end
    
    
end