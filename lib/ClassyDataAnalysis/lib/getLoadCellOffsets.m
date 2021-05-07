function [dates, offsets] = getLoadCellOffsets(labNum,dateTime)
% GETLOADCELLOFFSETS gets the list of load cell offsets for labNum and
% returns cell arrays for dates and offsets corresponding to the offsets
% taken just before and just after the date given by dateTime (one of each)

currentFilePath = mfilename('fullpath');
[folder,~,~] = fileparts(currentFilePath);

filePath = fullfile(folder,['LoadCell_lab' num2str(labNum) '_offsets.txt']);

% first line should be a comment, start reading from second line
fileContents = dlmread(filePath,'\t',1,0);

fullDates = fileContents(:,1);
fullOffsets = fileContents(:,2:7);

% find index of offsets from just before
tempDates = fullDates;
tempDates(tempDates > datenum(dateTime)) = NaN;
[beforeVal,beforeIdx] = min(abs(tempDates-datenum(dateTime)));

tempDates = fullDates;
tempDates(tempDates < datenum(dateTime)) = NaN;
[afterVal,afterIdx] = min(abs(tempDates-datenum(dateTime)));

if ~isnan(beforeVal) && ~isnan(afterVal)
    dates = cell(2,1);
    offsets = cell(2,1);
    dates{1} = datestr(fullDates(beforeIdx));
    dates{2} = datestr(fullDates(afterIdx));
    offsets{1} = fullOffsets(beforeIdx,:);
    offsets{2} = fullOffsets(afterIdx,:);
elseif ~isnan(beforeVal)
    dates = cell(1,1);
    offsets = cell(1,1);
    dates{1} = datestr(fullDates(beforeIdx));
    offsets{1} = fullOffsets(beforeIdx,:);
elseif ~isnan(afterVal)
    dates = cell(1,1);
    offsets = cell(1,1);
    dates{1} = datestr(fullDates(afterIdx));
    offsets{1} = fullOffsets(afterIdx,:);
else
    dates = {};
    offsets = {};
end

