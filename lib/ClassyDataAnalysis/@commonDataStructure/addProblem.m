function addProblem(cds,problem,varargin)
    %this is a method function for the common_data_structure (cds) class, and
    %should be located in a folder '@common_data_structure' with the class
    %definition file and other method files
    %
    %this function accepts a string and adds that string to the
    %cds.meta.knownProbelems property
    
    if numel(varargin)>1
        error('addProblem:tooManyInputs','addProblem only accepts a problem string and a single problem data structure')
    end
    if isempty(varargin)
        problemData=[];
    else
        problemData=varargin{1};
    end
    meta=cds.meta;
    meta.knownProblems=[meta.knownProblems;{problem,problemData}];
    set(cds,'meta',meta)
    
    
    evntData=loggingListenerEventData('addProblem',[]);
    notify(cds,'ranOperation',evntData)
end