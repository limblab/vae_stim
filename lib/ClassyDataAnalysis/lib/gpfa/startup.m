p = mfilename('fullpath'); % location of this file
p = strsplit(p,filesep); % pull apart the file path
p = strjoin(p(1:end-1),filesep); % rejoin without the filename

% adding relative path names in case the gpfa folder isn't currently on the
% path
addpath(p);
addpath([p,filesep,'core_gpfa']);
addpath([p,filesep,'core_twostage']);
addpath([p,filesep,'plotting']);
addpath([p,filesep,'util']);
addpath([p,filesep,'util',filesep,'precomp']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code checks for the relevant MEX files (such as .mexa64
% or .mexglx, depending on the machine architecture), and it creates the
% mex file if it can not find the right one.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Toeplitz Inversion
path(path,[p,filesep,'util',filesep,'invToeplitz']);
% Create the mex file if necessary.
if ~exist(sprintf('util/invToeplitz/invToeplitzFastZohar.%s',mexext),'file')
  try
    eval(sprintf('mex -outdir util/invToeplitz util/invToeplitz/invToeplitzFastZohar.c'));
    fprintf('NOTE: the relevant invToeplitz mex files were not found.  They have been created.\n');
  catch
    fprintf('NOTE: the relevant invToeplitz mex files were not found, and your machine failed to create them.\n');
    fprintf('      This usually means that you do not have the proper C/MEX compiler setup.\n');
    fprintf('      The code will still run identically, albeit slower (perhaps considerably).\n');
    fprintf('      Please read the README file, section Notes on the Use of C/MEX.\n');
  end
end
  
% Posterior Covariance Precomputation  
path(path,[p,filesep,'util',filesep,'precomp']);
% Create the mex file if necessary.
if ~exist(sprintf('util/precomp/makePautoSumFast.%s',mexext),'file')
  try
    eval(sprintf('mex -outdir util/precomp util/precomp/makePautoSumFast.c'));
    fprintf('NOTE: the relevant precomp mex files were not found.  They have been created.\n');
  catch
    fprintf('NOTE: the relevant precomp mex files were not found, and your machine failed to create them.\n');
    fprintf('      This usually means that you do not have the proper C/MEX compiler setup.\n');
    fprintf('      The code will still run identically, albeit slower (perhaps considerably).\n');
    fprintf('      Please read the README file, section Notes on the Use of C/MEX.\n');
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
clear p;