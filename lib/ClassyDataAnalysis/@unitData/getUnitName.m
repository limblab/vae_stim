function unitName=getUnitName(units,unitNum)
    %method of the unitData class. Should be saved in the @unitData folder
    %
    %unitName=unitData.getUnitName(unitNum) returns a formatted string with the unit Name
    %corresponding to the unit in the unitNum position of the unitData.data
    %struct array.
    %if unitNum is an array of indices, then getUnitName returns a cell
    %array of name strings
    
    numUnitsRequested=numel(unitNum);
    if numUnitsRequested==1
        unitName=[units.data(unitNum).array,'CH',num2str(units.data(unitNum).chan),'ID',num2str(units.data(unitNum).ID)];
    else
        unitName=cell(numUnitsRequested,0);
        for i=1:numUnitsRequested
            unitName{i}=[units.data(unitNum(i)).array,'CH',num2str(units.data(unitNum(i)).chan),'ID',num2str(units.data(unitNum(i)).ID)];
        end
    end
end