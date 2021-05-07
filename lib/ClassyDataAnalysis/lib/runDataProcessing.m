function dataStruct = runDataProcessing(mainFunctionName,targetDirectory,varargin)
    %data processing wrapper
    %1)the user must pass the name of their main function: main_function_name
    %2)the name of the directory to put all the results in: targetDirectory
    %3)optionally a struct of configuration parameters for their processing
    %functions
    %
    %the script then perorms the following operations:
    %a)generates the lab standard folder tree
    %b)calls the function specified in main_function_name, passing in the
    %targetDirectory variable in case the function needs to explicitly save
    %figures or data
    %c)saves returned figures in .fig and .pdf formats. The name of the files
    %will be the name of the figure (NOT the title). If the name is empty the
    %figure wil be saved as Figure_1, Figure_2 etc. 
    %use set(H,'Name','fig_name') to set the name of figures
    %d)Saves the three input variables as separate files so that the data 
    %processing is reproducable.
    %e)Saves each field in the output data struct as a separate field. saves 
    %pstrings in flat text files and all other data in individual m-files. 
    %The names of data files will be the names of the fields of the
    %output data struct
    %
    %the user's analysis function should accept a single argument which is the
    %target folder to save the processed data into
    %the analysis function should return one cell array, and one struct:
    %the cell array should contain the handles to all the figures
    %that need to be saved. Those figures must not be closed by the
    %analysis code
    %the struct should contain the various data objects that should be
    %saved for later use. each field should be the name of the object,
    %which will be the file tname the object is saved under. If the object
    %is a simple string, the data will be written to a flat text file.
    %Lists of data files operated on and other general information should
    %be passed in this way. All other data types will be saved as .mat
    %files. For instance, a bdf can be saved by passing the bdf as one of 
    %the elements in the outpus struct.
    
    %% sanitize input:
    if ~strcmp(targetDirectory(end),filesep)
        disp(['appending trailing ' filesep ' character to folder name'])
        targetDirectory=[targetDirectory filesep];
    end

    %% make directory structure if it does not already exist
    
    if exist(strcat(targetDirectory,'Code'),'file')~=7
        mkdir(strcat(targetDirectory,'Code'))
    else
        warning('RUN_DATA_PROCESSING:FOLDER_EXISTS','A folder with processed data already exists, you may lose data if you continue')
        yesno=questdlg('The target folder already exists. If you continue data may be lost. Do you want to continue?','Folder already exists','Yes','No','No');
        if strcmp(yesno,'No')
            return
        end
    end
    if exist(strcat(targetDirectory,'Raw_Figures'),'file')~=7
        mkdir(strcat(targetDirectory,'Raw_Figures'))
    end
    if exist(strcat(targetDirectory,['Raw_Figures' filesep 'PDF']),'file')~=7
        mkdir(strcat(targetDirectory,['Raw_Figures' filesep 'PDF']))
    end
    if exist(strcat(targetDirectory,['Raw_Figures' filesep 'FIG']),'file')~=7
        mkdir(strcat(targetDirectory,['Raw_Figures' filesep 'FIG']))
    end
    if exist(strcat(targetDirectory,['Raw_Figures' filesep 'EPS']),'file')~=7
        mkdir(strcat(targetDirectory,['Raw_Figures' filesep 'EPS']))
    end
    if exist(strcat(targetDirectory,['Raw_Figures' filesep 'PNG']),'file')~=7
        mkdir(strcat(targetDirectory,['Raw_Figures' filesep 'PNG']))
    end
    if exist(strcat(targetDirectory,'Edited_Figures'),'file')~=7
        mkdir(strcat(targetDirectory,'Edited_Figures'))
    end
    if exist(strcat(targetDirectory,'Output_Data'),'file')~=7
        mkdir(strcat(targetDirectory,'Output_Data'))
    end
    if exist(strcat(targetDirectory,'Input_Data'),'file')~=7
        mkdir(strcat(targetDirectory,'Input_Data'))
    end

    %% save all the custom functions in the analysis to the code folder.
    %Specifically ignore all functions that are part of the Matlab built-in
    %functions or toolboxes
    
    %command_list=[getUserDependencies(main_function_name);{strcat(mfilename('fullpath'),'.m')};{which(main_function_name)}];
    commandList=[matlab.codetools.requiredFilesAndProducts(mainFunctionName),...
                    {strcat(mfilename('fullpath'),'.m')},...
                    {which(mainFunctionName)}];
    for i=1:length(commandList)
        [SUCCESS,MESSAGE,MESSAGEID] = copyfile(commandList{i},strcat(targetDirectory,'Code'));
        if SUCCESS
            disp(strcat('successfully copied ',commandList{i},' to the code folder'))
        else
            disp('script copying failed with the following message')
            disp(MESSAGE)
            disp(MESSAGEID)
        end
    end

    %% evaluate the main processing function
    if ~isempty(varargin)
        [figureList,dataStruct]=eval(strcat(mainFunctionName,'(targetDirectory,varargin{1})'));
    else
        [figureList,dataStruct]=eval(strcat(mainFunctionName,'(targetDirectory)'));
    end
    %% save all the figures
    for i=1:length(figureList)
        RDPSaveFig(figureList(i),targetDirectory)
    end

    %% save the input and output data structures
    if ~isempty(varargin)
        temp=varargin{1};
        save(strcat(targetDirectory,['Input_Data' filesep 'Input_structure.mat']),'temp','-mat')
    end
    fid=fopen(strcat(targetDirectory,['Input_Data' filesep 'targetDirectory.txt']),'w+');
            fprintf(fid,'%s',targetDirectory);
            fclose(fid);
    fid=fopen(strcat(targetDirectory,['Input_Data' filesep 'main_function_name.txt']),'w+');
            fprintf(fid,'%s',mainFunctionName);
            fclose(fid);    
    if ~isempty(dataStruct)
        data_list=fieldnames(dataStruct);
        for i=1:length(data_list)
            RDPSave(dataStruct.(data_list{i}),data_list{i},targetDirectory)
        end
    end
end
