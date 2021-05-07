function [errorCellStr]=error2CellStr(ME)
    %this is a simple helper that converts a Matlab message error object
    %into a cell-string array for easy display/printing. This function simply
    %abstracts parsing of the ME structure to a one-line function call, eg:
    %disp(error2CellStr(ME))
    %would print the error message to the screen without halting execution
    %the way rethrow(ME) does.
    
    errorCellStr = {ME.identifier;ME.message};
    for i=1:numel(ME.stack)
        errorCellStr=[errorCellStr;{ME.stack(i).file;['line: ' num2str(ME.stack(i).line)]}]; 
    end
    
end