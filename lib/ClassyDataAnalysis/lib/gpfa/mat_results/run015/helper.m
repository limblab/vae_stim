count = 1;
for i = 1:length(seqTrain)
    if ismember(seqTrain(i).trialId, leftTrialNums)
        tmep2(count) = seqTrain(i);
        count = count+1;
    end
end