function mask=list2tableMask(dataTable,list)
    %mask=list2mask(dataTable,list)
    %takes in a table and a list of column names, and returns a logical
    %vector that is 1 where a column in dataTable has a name that appears
    %in list, and is 0 otherwise. Intended for use in finding columns in
    %table.
    %list must be a cell array of strings
    
    if ~iscellstr(list)
        error('list2tableMask:badListFormat','the list input must be a cell array of strings')
    end
    
    mask=false(1,numel(dataTable.Properties.VariableNames));
    for j=1:numel(list)
        mask=mask | strcmp(list{j},dataTable.Properties.VariableNames);
    end
    
end