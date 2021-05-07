function appendLoadCellOffsets(offsets,labNum,dateTime)
% Use recorded file with still robot data to calculate load cell offsets
% and append offsets to lab-specific load cell offsets file in
% ClassyDataAnalysis/lib

    currentFilePath = mfilename('fullpath');
    [folder,~,~] = fileparts(currentFilePath);
    
    filePath = fullfile(folder,['LoadCell_lab' num2str(labNum) '_offsets.txt']);

    fid = fopen(filePath,'a');
    if fid==-1
        error('appendLoadCellOffsets:FileNotFound','File not found.')
    end
    
    % compose line to append
    str_offsets = num2str(offsets,'%f\t');
    
    % append with a line containing the date and load cell offsets
    fprintf(fid,[num2str(datenum(dateTime)) '\t' str_offsets '\n']);
    
    fclose(fid);
end