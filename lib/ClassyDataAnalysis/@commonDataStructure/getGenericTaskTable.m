function getGenericTaskTable(cds) 
%this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %cds.getRWTaskTable(times)
    %getWFTaskTable returns no value, instead it populates the trials field
    %of the cds assuming the task is a wrist flexion task. Takes a single
    %input:times, which is a table with 4 columns: number, startTime,
    %endTime, and result. These times define the start and stop of trials
    %as indicated by the state words for trial start and trial end. the
    %result code will be a character 'R':reward 'A':abort 'F':fail
    %'I':incomplete.
    

    trials=table(0,cds.NEV.MetaTags.DataDurationSec, string('r'),...
                'VariableNames',{'startTime','endTime', 'result'});
    trials.Properties.VariableUnits={'s','s','string'};
    trials.Properties.VariableDescriptions={'start of file','end of file', 'result'};

    trials=[trials];
    trials.Properties.Description='Trial table for No task for use in trial data format';
    %cds.setField('trials',trials)
    set(cds,'trials',trials)
    evntData=loggingListenerEventData('getGenericTaskTable',[]);
    notify(cds,'ranOperation',evntData)
end