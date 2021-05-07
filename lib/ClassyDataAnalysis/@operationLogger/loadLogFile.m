function logEntry=loadLogFile(obj,fname)
    %this is a method of the operationLogger superclass and should be
    %stored in the @operationLogger folder with the other methods
    %
    %logEntry=loadLogFile(fname) looks for a file with the name fname in 
    %the working directory, opens it, and parses it to find the most recent
    %git log entry in the file, returning that entry to the calling function
    %this is intended to work on the output of the writeLocalGitLog.sh
    %linux shell script, and will probably not execute propterly on files
    %generated by other means
    
    if ~isunix
        warning('loadLogFile:notUnux','loadLogFile is intended to run on text files generated on Unix systems, and may not operate correctly on your computer')
    end
    fid=fopen(fname);
    
    c=textscan(fid,'%s','Delimiter','\n');
    if numel(c{1})<4
        %if we don't have a complete log entry, we weren't in a repo when
        %the local log file was generated so just return an empty folder
        logEntry=[];
        return
    end
    %if we have an empty cell where there were two carriage returns in the
    %file, remove it:
    if isempty(strtrim(c{1}{end-1}))
        c{1}(end-1)=[];
    end
    %put the cells back into a single string in the format expected by
    %operationLogger.getGitLog
    logEntry=strjoin(c{1},'\n');
    fclose(fid);
end