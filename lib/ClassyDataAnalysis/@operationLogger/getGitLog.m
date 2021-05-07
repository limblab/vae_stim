function varargout=getGitLog(obj,path,varargin)
    %getGitLog is a method of the operationLogger superclass and should be
    %saved in the @operationLogger folder with the other class methods
    %
    %takes the full path to an m-file and searches all the parent folders
    %to see if the file is in a git repository. At the first git repository
    % getGitLog will pull the git log into a string and then  break it into
    % a cell array, where each cell is one line of the log
    
    if ~isempty(varargin)
        fullLog=varargin{1};
    else
        fullLog=0;
    end
    %% search for a git repo in the directory tree of the given path
    temp=path;
    while length(temp)>=4 % if we haven't gotten down to the core drive, e.g. C:/
        temp=fileparts(temp);%cut the last folder off temp
        if (exist([temp,filesep,'.git'],'file'))==7
            %if we are in the main git folder, break, leaving 'temp' as the
            %path to the core repo            
            break
        end
    end
    if temp<4
        %if we didn't find a git repo in the parent tree of the file,
        %return empty structures
        varargout{1}=[];
        if nargout==2
            varargout{2}=[];
        end
        return
    end
    %% get strings with the file log and repo log:
   %This method needs to change directories to function. store the current
   %directory so we can return at the end of execution
    workingDir=pwd;
    if ispc
        
        cd(fileparts(path));
        %get the git log for our repo:
        if fullLog
            gitLogString=evalc(['!git --git-dir=',temp,filesep,'.git log']);
        else
            gitLogString=evalc(['!git --git-dir=',temp,filesep,'.git log -1']);
        end
        %get the git log for our file:
        %dont forget to restrict the log call to the first record, as returning
        %all records basically causes matlab to hang as it parses the input
        if strcmp(path(end-1:end),'.m')
            fileLogString=evalc(['!git log -n 1 -- ', path]);
        else
            fileLogString=evalc(['!git log -n 1 -- ', path, '.m']);
        end
        
    elseif isunix
        %get path to the shell script that pulls git logs and saves them in
        %a matlab friendly format. The script should be saved in the
        %@operationLogger folder
        [classFolder,~,~]=fileparts(obj.locateMethod('operationLogger','operationLogger'));
        cd(classFolder);
        if isempty(dir('*writeLocalLog.sh'))
            error('getGitLog:missingLinuxBashScript','operationLogger.getGitLog requries the bash script writeLocalLog.sh to be stored in the @operationLogger class definition folder')
        end
        %generate the repoGitLog.tmp and fileGitLog.tmp
        system(['export TERM=ansi; ./writeLocalLog.sh ',temp,' ',path]);
        %load data from the files into the string variables:
        fileLogString=obj.loadLogFile('fileGitLog.tmp');
        gitLogString=obj.loadLogFile('repoGitLog.tmp');
        %clear the temporary files we generated:
        system('export TERM=ansi; ./clearLocalLog.sh');
    else
        error('getGitLog:untestedEnvironment','getGitLog is tested on windows and linux PCs, and my not run on macs')        
    end
    %% now that we have the log strings, parse the main repo log into a struct:
    %if our file is in a git repo, find the home directory for the git repo

    gitLog=strsplit(gitLogString,'\n');
    if ~isempty(gitLogString)
        for i=1:length(gitLog)
            %get the commit hash
            if strfind(gitLog{i},'commit ')
                gitLogStruct.hash=gitLog{i}(8:end);
            end
            %get the file author
            if strfind(gitLog{i},'Author ')
                gitLogStruct.author=gitLog{i}(8:end);
            end
            %get the commit user
            if strfind(gitLog{i},'Date:   ')
                gitLogStruct.date=gitLog{i}(8:end);
            end
        end
    end
    varargout{1}=gitLogStruct;

    %% if necessary parse the filelog string into a struct:
    if nargout==2
        fileLog=strsplit(fileLogString,'\n');
        if ~isempty(fileLogString)
            for i=1:length(fileLog)
                %get the commit hash
                if strfind(fileLog{i},'commit ')
                    fileLogStruct.hash=fileLog{i}(8:end);
                end
                %get the file author
                if strfind(fileLog{i},'Author: ')
                    fileLogStruct.author=fileLog{i}(8:end);
                end
                %get the commit user
                if strfind(fileLog{i},'Date:   ')
                    fileLogStruct.date=fileLog{i}(8:end);
                end
            end
            varargout{2}=fileLogStruct;
        else
            varargout{2}='not in git';
        end
    end
    %% set the directory back to the working directory
    cd(workingDir)
    return
end