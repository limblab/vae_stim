function isDone=RDPIsAlreadyDone(objName,targetDirectory)
    %checks the runDataProcessing directory structure to see if a specified
    %variable exists:
    if ~strcmp(targetDirectory(end),filesep)
        targetDirectory(end+1)=filesep;
    end
    isDone=exist([targetDirectory,'Output_Data' ,filesep,objName,'.mat'],'file');
end