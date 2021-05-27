function [point_kin] = getDataFromOpensimFile(input_data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    point_kin = []; 

    trialsForAn = dir([input_data.mot_path filesep, input_data.mot_name]);
    name = regexprep(trialsForAn(1).name,'.mot','');
    
    locs = {'elbow','hand'};
    vars = {'acc','vel','pos'};
    
    for loc = locs
        for var = vars
            var_name = [loc{1},'_',var{1}];
            temp_kin = [];

            % get file
            fname = [name,'_PointKinematics_',loc{1},'_',var{1},'.sto'];
            file_full_path = [input_data.results_folder filesep fname];
            
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

