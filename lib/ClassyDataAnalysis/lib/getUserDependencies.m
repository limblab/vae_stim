function functionList=getUserDependencies(fname)
    %returns a cell array with strings containing the functions that the
    %function fname depends on
    
    if ~isempty(strfind(fname,'@'))
        %we have a class method here. since class methods return all
        %class methods as dependencies we need to handle that specially or
        %we wind up in an infinite regress and hit the recursion limit
        %right away
        functionList=depfunLimblab(fname,'-toponly','-quiet');
        return
    end
    functionList={};
    
    commandList=depfunLimblab(fname,'-toponly','-quiet');
    for i=1:numel(commandList)
        if ~strfind(commandList{i},matlabroot)
            functionList=commandList(1);
            break
        end
    end
    for i=2:length(commandList)%skip the first element since that is the calling function
        if strfind(commandList{i},matlabroot)
            continue
        else
            temp=getUserDependencies(commandList{i});
            if isempty(temp)
                functionList(length(functionList)+1)=commandList(i);
            else
                functionList=[functionList;temp];
            end
        end
    end
end