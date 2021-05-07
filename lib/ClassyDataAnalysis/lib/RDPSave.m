function RDPSave(obj,objName,targetDirectory)
    %saves an object assuming the folder tree from RDP exists
    if ~strcmp(targetDirectory(end),filesep)
        targetDirectory(end+1)=filesep;
    end
    if strcmp(objName,'session_summary')
        write_session_summary(data_struct.session_summary,strcat(targetDirectory,['Output_Data' filesep ,'session_summary.txt']))
    end
    if ischar(obj)%if the field is just a string like a list of file names
        fid=fopen(strcat(target_directory,['Output_Data' filesep],objName,'.txt'),'w+');
        fprintf(fid,'%s',obj);
        fclose(fid);
    else           
        %generate a local variable with the correct name:
        eval([objName, '=obj;']);
        %save(strcat(target_directory,['Output_Data' filesep],data_list{i},'.mat'),data_list{i},'-mat')
        dummy_var = [];  % Matlab will compress the first variable saved, making it slower to load, so we compress an empty array.
        save(strcat(targetDirectory,['Output_Data' filesep],objName,'.mat'),'dummy_var',objName,'-mat','-v7.3')
    end
    
end