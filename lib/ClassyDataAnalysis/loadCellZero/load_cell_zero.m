%%

lab=6;
task='taskRW';
folder='C:\Users\rhc307\Projects\limblab\data-preproc\Misc\LoadCell\20180801\';
fname='Loadcell_20180801_still';
% Make CDS files

cds = commonDataStructure();
cds.file2cds([folder fname],lab,'ignoreJumps',task,'getLoadCellOffsets','useAbsoluteStillThresh');