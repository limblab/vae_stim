
#writes most recent git log entry for input path to file

gitPath=$1
filePath=$2
currPath=$PWD
fileLogFile="fileGitLog.tmp"
repoLogFile="repoGitLog.tmp"
cd $gitPath
git log -1 > $currPath/$repoLogFile
git log -1 $filePath > $currPath/$fileLogFile

