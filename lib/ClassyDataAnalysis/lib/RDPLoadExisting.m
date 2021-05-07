function obj=RDPLoadExisting(objName,targetDirectory)
    %loads an object from a previous iteration of runDataProcessing.
    %Assumes the folder structure from RDP is intact, and that files within
    %the output_data folder have not been tampered with, so that every file
    %will have at MOST 2 variables: the variable of interest, and a second
    %variable called 'dummy_var' saved to prevent matlab from compressing
    %the data of interest
    if ~strcmp(targetDirectory(end),filesep)
        targetDirectory(end+1)=filesep;
    end
    tmp=load([targetDirectory,'Output_Data', filesep, objName,'.mat']);
    fnames=fieldnames(tmp);
    for i=1:numel(fnames)
        %confirm we aren't looking at the dummy that is saved first to
        %speed file loading. If not, put into output variable and return
        if ~strcmp(fnames{i},'dummy_var')
            obj=tmp.(fnames{i});
            return
        end
    end
    
end