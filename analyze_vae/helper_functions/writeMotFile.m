function [  ] = writeMotFile(fpath, fname, t, joint_ang, joint_ang_names, in_deg)

    % make file
    if(isempty(strfind(fname,'.mot')))
        fname = [fname, '.mot'];
    end
    
    file_id = fopen([fpath filesep, fname],'wt');

    % write header
    fprintf(file_id,'Coordinates\n');
    fprintf(file_id,'version=1\n');
    fprintf(file_id,'nRows=%i\n',size(joint_ang,1));
    fprintf(file_id,'nColumns=%i\n',size(joint_ang,2)+1); % add one for time column
    
    deg_str = 'no';
    if(in_deg)
        deg_str='yes';
    end
    fprintf(file_id,['inDegrees=' deg_str '\n']);
    fprintf(file_id,'\n');
    fprintf(file_id,'Units are S.I. units (second, meters, Newtons, ...)\n');
    fprintf(file_id,'Angles are in degrees.\n');
    fprintf(file_id,'\n');
    fprintf(file_id,'endheader\n');
    
    % write time, joint names
    fprintf(file_id, 'time\t');
    for i_name = 1:numel(joint_ang_names)
        fprintf(file_id,[joint_ang_names{i_name},'\t']);
    end
    fprintf(file_id,'\n');
    
    % write data with tabs delimiting columns, rows are time points. Need
    % to generate time stamps
    data_str = '';
    for i_col = 1:(size(joint_ang,2)+1)
        data_str = [data_str ,'%.7f'];
        if(i_col == size(joint_ang,2)+1)
            data_str = [data_str,'\n'];
        else
            data_str = [data_str,'\t'];
        end
    end
    
    for i_data = 1:size(joint_ang,1)
        fprintf(file_id, data_str, [t(i_data), joint_ang(i_data,:)]);
    end
    % close file
    fclose(file_id);

end

