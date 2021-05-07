gpfaConfig.trials = nonbumpTrials.number;
gpfaConfig.dimension = 8;
gpfaConfig.segLength = 40;
gpfaConfig.windows=trialStartEnd;
set(ex.bin, 'gpfaConfig', gpfaConfig);
ex.bin.fitGpfa;