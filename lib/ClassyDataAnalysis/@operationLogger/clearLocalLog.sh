#this script is paired with writeLocalLog.sh, 
#which generates  copies of the file and repo level
#git logs in filGitLog.tmp and repoGitLog.tmp respectively
#this script deletes those two files, however that action 
#could easily be taken by another program, eg Matlab, directly
fileLogFile="fileGitLog.tmp"
repoLogFile="repoGitLog.tmp"
rm $fileLogFile
rm $repoLogFile

