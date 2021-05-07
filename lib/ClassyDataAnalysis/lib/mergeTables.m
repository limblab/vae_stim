function merged=mergeTables(table1,table2)
    %takes two tables of time-series data, and merges them. Performs basic 
    %sanity checks to ensure same sampling rate, and no duplicate column
    %names. Truncates the output so that the resulting table exists only on
    %the common range of the two source tables. Both input tables MUST
    %have a column 't' indicating the time of each row. Both tables MUST
    %use the same units for time.
    tstart=table1.t(1);
    tend=table1.t(end);
    dt=mode(diff(table1.t));
    
    tstart2=table2.t(1);
    tend2=table2.t(end);
    dt2=mode(diff(table2.t));
    %check our frequencies
    if dt~=dt2
        error('mergeTable:differentFrequency',['Field: ',fieldName,' was collected at different frequencies in the cds and the new data and cannot be merged. Either re-load both data sets using the same filterspec, or refilter the data in one of the cds structures using decimation to get to the frequencies to match'])
    end
    %check if we have duplicate columns:
    for j=1:length(table1.Properties.VariableNames)
        if ~strcmp(table1.Properties.VariableNames{j},'t') && ~isempty(find(cell2mat({strcmp(table2.Properties.VariableNames,table1.Properties.VariableNames{j})}),1,'first'))
            error('mergeTable:duplicateColumns',['the column label: ',table1.Properties.VariableNames{j},' exists in the ',fieldName,' field of both cds and new data. All columns in the cds and new data except time must have different labels in order to merge'])
        end
    end
    mask=cell2mat({~strcmp(table2.Properties.VariableNames,'t')});
    merged=[table1(find(table1.t>=max(tstart,tstart2),1,'first'):find(table1.t>=min(tend,tend2),1,'first'),:),...
            table2(find(table2.t>=max(tstart,tstart2),1,'first'):find(table2.t>=min(tend,tend2),1,'first'),(mask))];
end