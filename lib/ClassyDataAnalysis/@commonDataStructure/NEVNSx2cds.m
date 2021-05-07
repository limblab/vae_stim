function NEVNSx2cds(cds,opts)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files. This method is not user
    %accessible, but rather is called by the file2cds method.
    %
    %this method is essentially a wrapper around the various processing
    %routines like kinematicsFromNEV. NEVNSx2cds assumes that the NEV and
    %NSx data is already loaded into the cds, so the nev2NEVNSx method must
    %be called before NEVNSx2cds
    
    %% Initial setup
        % make sure LaTeX is turned off and save the old state so we can turn
        % it back on at the end
        defaultTextInterpreter = get(0, 'defaulttextinterpreter'); 
        set(0, 'defaulttextinterpreter', 'none');

        %initial setup
        
        %get the date of the file so processing that depends on when the
        %file was collected has something to work with
        opts.dateTime= [int2str(cds.NEV.MetaTags.DateTimeRaw(1)) '/' int2str(cds.NEV.MetaTags.DateTimeRaw(2)) '/' int2str(cds.NEV.MetaTags.DateTimeRaw(4)) ...
            ' ' int2str(cds.NEV.MetaTags.DateTimeRaw(5)) ':' int2str(cds.NEV.MetaTags.DateTimeRaw(6)) ':' int2str(cds.NEV.MetaTags.DateTimeRaw(7)) '.' int2str(cds.NEV.MetaTags.DateTimeRaw(8))];
        opts.duration= cds.NEV.MetaTags.DataDurationSec;
        if strcmp(opts.task, 'FR')
           opts.robot = false; 
        end
        %% Events:
        %if events are already in the cds, then we keep them and ignore any
        %new words in the NEVNSx. Otherwise we load the events from the
        %NEVNSx, followed by the task
        if isempty(cds.words)
            %do this first since the task check requires the words to already be processed, and task is required to work on kinematics and force
            cds.eventsFromNEV(opts)
            % if a task was not passed in, set task varable
            if strcmp(opts.task,'Unknown') %if no task label was passed into the function call try to get one automatically
                opts=cds.getTask(opts.task,opts);
            end
            
        end
        
    %% the kinematics
    
        %convert event info into encoder steps:
        if ~isempty(cds.words) && ~any(strcmp(opts.task,{'RT3D','none','cage','multi_gadget','ball_drop'}))
            cds.kinematicsFromNEV(opts)
        end
       

    %% the kinetics
        if opts.robot || any(strcmp(opts.task,{'WF','multi_gadget'}))
            cds.forceFromNSx(opts)
        end

    %% The Units
    % Build catalogue of entities
        unit_list = unique([cds.NEV.Data.Spikes.Electrode;cds.NEV.Data.Spikes.Unit]','rows');
        if ~isempty(unit_list)   
            cds.unitsFromNEV(opts)
        end
        
    %% EMG
        cds.emgFromNSx()
            
    %% LFP. any collection channel that comes in with the name chan* will be treated as LFP
        cds.lfpFromNSx(opts)

    %% Triggers
        %get list of triggers
        cds.triggersFromNSx()

    %% Analog
        cds.analogFromNSx()

    %% trial data
        %if we have databursts and we don't have a trial table yet, compute
        %the trial data, otherwise skip it
        if (~isempty(cds.databursts) && isempty(cds.trials))
            if strcmp(opts.task,'Unknown') 
                warning('NEVNSx2cds:UnknownTask','The task for this file is not known, the trial data table may be inaccurate')
            end
            cds.getTrialTable(opts)
        elseif (isempty(cds.databursts) && isempty(cds.trials) && strcmp(opts.task, 'Unknown'))
            opts.no_task = true;
            cds.getTrialTable(opts)
        end
    %% sanitize times so that all our data is in the same window.
        if ~opts.unsanitizedTimes
            cds.sanitizeTimeWindows
        end
    %% Set metadata. Some metadata will already be set, but this should finish the job
        cds.metaFromNEVNSx(opts)
        cds.setDataWindow()
    %% reset the text interpreter:
        set(0, 'defaulttextinterpreter', defaultTextInterpreter);
        
end
