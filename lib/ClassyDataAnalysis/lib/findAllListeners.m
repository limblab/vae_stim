function eventsWithListeners=findAllListeners(obj)
    %this function attempts to find all the events with listeners in obj or
    %its properties. 
    %eventsWithListeners=findAllListeners(obj)
    %returns a cell string array with the names of events in obj that still
    %have listeners in the matlab workspace. for events that are properties 
    %of obj, the entry will be of the form 'obj.propName'
    %
    %apparently the event.hasListeners function is
     %not part of the matlab environement under linux. We are going
     %to try to check the listeners, and throw a warning if we fail
     %to actually perform the check
     
     eventsWithListeners=[];
     %check events of the main object:
     eventList=events(obj);
     %remove built-in matlab events:
     eventList=eventList(~strcmp(eventList,'ObjectBeingDestroyed'));
     for i=1:numel(eventList)
         try
             if event.hasListener(obj,eventList{i})
                 eventsWithListeners=[eventList(numel(eventList)+1);eventList(i)];
             end
         catch ME
             warning('findAllListeners:FailedToCheckListeners','failed to check whether the listeners to cds events were properly cleared')
             disp('the event.hasListeners method is not available in some distirbutions (linux?). All listeners may be cleared but the cds destructor is not able to confirm')
             disp('check failed with the following error:')
             disp(error2CellStr(ME))
             return
         end
     end
     %check whether there are any listeners on the properties of
     %the object
    propList=properties(object);
    for i=1:numel(propList)
        eventList=events(obj.(propList{i}));
         %remove built-in matlab events:
         eventList=eventList(~strcmp(eventList,'ObjectBeingDestroyed'));
        if ~isempty(eventList)
            for j=1:numel(eventList)
                try
                     if event.hasListener(obj.(propList{i}),eventList{i})
                         eventsWithListeners=[eventList(numel(eventList)+1);{[propList{i},'.',eventList{i}]}];
                     end
                 catch ME
                     warning('findAllListeners:FailedToCheckListeners','failed to check whether the listeners to cds events were properly cleared')
                     disp('the event.hasListeners method is not available in some distirbutions (linux?). All listeners may be cleared but the cds destructor is not able to confirm')
                     disp('check failed with the following error:')
                     disp(error2CellStr(ME))
                     return
                 end
            end
        end
    end
end