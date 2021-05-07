function setDataWindow(cds)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %setDataWindow sets the dataWindow and duration in the cds.meta field, 
    %based on the windows of data in the recorded signals
    
    meta = cds.meta;
    meta.dataWindow=[0 inf];
    %find the real data Window:
    if meta.hasEmg
        meta.dataWindow=[max(meta.dataWindow(1),cds.emg.t(1)),min(meta.dataWindow(2),cds.emg.t(end))];
    end
    if meta.hasLfp
        meta.dataWindow=[max(meta.dataWindow(1),cds.lfp.t(1)),min(meta.dataWindow(2),cds.lfp.t(end))];
    end
    if meta.hasKinematics
        meta.dataWindow=[max(meta.dataWindow(1),cds.kin.t(1)),min(meta.dataWindow(2),cds.kin.t(end))];
    end
    if meta.hasForce
        meta.dataWindow=[max(meta.dataWindow(1),cds.force.t(1)),min(meta.dataWindow(2),cds.force.t(end))];
    end
    if meta.hasAnalog
        for j=1:length(cds.analog)
            meta.dataWindow=[max(meta.dataWindow(1),cds.analog{j}.t(1)),min(meta.dataWindow(2),cds.analog{j}.t(end))];
        end
    end
    meta.duration=meta.dataWindow(end)-meta.dataWindow(1);
    
    set(cds,'meta',meta)
    
    %log the update to cds.meta
    evntData=loggingListenerEventData('setDataWindow',[]);
    notify(cds,'ranOperation',evntData)