% %get PDs from CObump data
%     if ~exist('NEVNSx','var')
%         NEVNSx=cerebus2NEVNSx('E:\local processing\Han\20160411_COBump_ctrHold_Delay','Han_20160411_COBump_area2_001');
%     end
%% load test data into cds

    if ~exist('cds','var')
        cds=commonDataStructure();
        cds.file2cds('C:\Users\csv057\Documents\Data\Han_20160415_COBump_area2_tucker_001-01.nev','arrayS1Area2','monkeyHan',6,'ignoreJumps','taskCObump', 'ranByChris', 'mapFileY:\limblab\lab_folder\Animal-Miscellany\Han_13B1\map files\Left S1\SN 6251-001459.txt');
        save('C:\Users\csv057\Documents\Data\output-01.mat','cds','-v7.3')
    end
    %
    
%% create new experiment object
    ex=experiment();

% set which variables to load from cds
    ex.meta.hasLfp=false;
    ex.meta.hasKinematics=true;
    ex.meta.hasForce=true;
    ex.meta.hasUnits=true;
    ex.meta.hasTrials=true;

% set configuration parameters that are not default 
%pdConfig setup:
    ex.bin.pdConfig.useParallel=true;
    ex.bin.pdConfig.pos=true;
    ex.bin.pdConfig.vel=true;
    ex.bin.pdConfig.force=true;
    ex.bin.pdConfig.speed=true;
    ex.bin.pdConfig.units={};%just use all of them
    
% set binConfig parameters:
%     ex.binConfig.include(1).field='lfp';
%         ex.binConfig.include(1).which={};
    ex.binConfig.include(1).field='units';
        ex.binConfig.include(1).which=find([ex.units.data.ID]>0 & [ex.units.data.ID]<255);
    ex.binConfig.include(2).field='kin';
        ex.binConfig.include(2).which={};
    ex.binConfig.include(3).field='force';
        ex.binConfig.include(3).which={};
        ex.binConfig.filterConfig.sampleRate = 1000;
        ex.binConfig.filterConfig.cutoff = 100;
        
% set firingRateConfig parameters
    ex.firingRateConfig.cropType='tightCrop';
    ex.firingRateConfig.offset=-.015;
    ex.firingRateConfig.sampleRate = 1000;
    %ex.firingRateConfig.lags=[-2 3];
    
% load experiment from cds:
ex.addSession(cds)

% calculate the firing rate
    %ex.calcFiringRate()
% bin the data
    ex.binData()
    save('C:\Users\csv057\Documents\MATLAB\GPFA\gpfa\ex.mat','ex','-v7.3')
  %% Moving from the experiment to get it to work with GPFA 
  rewardRows = strmatch('R',ex.trials.data.result);
  rewardRowMask = zeros(length(ex.trials.data.number(:,1)),1);
  for i = 1:length(ex.trials.data.number(:,1))
      if ismember(i, rewardRows)
          rewardRowMask(i, 1) = 1;
      end
  end
  nonbumpTrials = ex.trials.data(ex.trials.data.ctrHoldBump ==0 & ex.trials.data.delayBump ==0 & ex.trials.data.moveBump ==0 & ~isnan(ex.trials.data.movePeriod) & ~isnan(ex.trials.data.goCueTime) & rewardRowMask,:);
  trialStartEnd = [nonbumpTrials.tgtOnTime, nonbumpTrials.goCueTime];
  dat = struct('trialId',[], 'spikes', []);
  for i = 1:length(trialStartEnd) %grab the firing of neurons during this period.
      dat(i).trialId =  nonbumpTrials.number(i);
      dat(i).spikes(:,:) = table2array(ex.bin.data(find(ex.bin.data.t>trialStartEnd(i,1), 1):find(ex.bin.data.t>trialStartEnd(i,2),1), 12:end))'./1000;
  end

  rightTrials = nonbumpTrials(nonbumpTrials.tgtDir == 0, :);
  upTrials = nonbumpTrials(nonbumpTrials.tgtDir == 90, :);
  leftTrials = nonbumpTrials(nonbumpTrials.tgtDir == 180, :);
  downTrials = nonbumpTrials(nonbumpTrials.tgtDir == 270, :);
  rightTrialNums = rightTrials.number;
  upTrialNums = upTrials.number;
  leftTrialNums = leftTrials.number;
  downTrialNums = downTrials.number;
  
for i = 1:length(dat)
    dat(i).spikes = dat(i).spikes(:, end-399:end);
end