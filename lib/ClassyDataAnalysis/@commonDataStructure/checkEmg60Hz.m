function checkEmg60Hz(cds)
    %this is a method of the cds class and is not intended for general use
    %this function uses historical data to build an LDA classifier and
    %return whether the EMG strems of the cds contain concerning levels of
    %power in the 60hz harmonics. The data for the classifier will be the
    %ratio of 60hz harmonic power to other power measures, as put out by
    %the power60HzRatio() function.
    
    if isempty(cds.emg)
        return
    end
    
    %historic data:
        [pRatios,classification]=kevinData;
        [rTemp,cTemp]=jangoData;
        pRatios=[pRatios;rTemp];
        classification=[classification;cTemp];
        
    %build classifier:
        discr=fitcdiscr(pRatios,classification);
    %get current ratio data:
        currPRatios=power60HzRatio(cds.emg);
    % loop through emg and test each one, adding a problem if we classify
        numEmg=size(currPRatios,1);
        for i=1:numEmg
            C=predict(discr,currPRatios(:,i));
            if ~C
                %this is classed as a contaminated signal
                problemString=['EMG contaminated with 60Hz noise'];
                problemData.emgName=cds.emg.VariableNames{i+1};
                problemData.powerRatios=currPRatios;
                problemData.historicalData=pRatios;
                cds.addProblem(problemString,problemData)
            end
        end
    eventData=loggingListenerEventData('checkEmg60Hz',[]);
    notify(cds,'ranOperation',eventData)
end

function [pRatios,classification]=kevinData()
    %Hybrid_Kevin_05152015:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (bad)
        classification=[0 1 1 1 1 0]';
        pRatios=[];
        
    %Hybrid_Kevin_05192015:
        %FCU (bad),FCR [classification;(good), ECU (good), ECR (good), FDS (good), FDP (bad)
        classification=[classification;[0 1 1 1 1 0]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_05202015:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (bad), FDP (good)
        classification=[classification;[0 1 1 1 0 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_05212015:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[0 1 1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_05252015 file1:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[0 1 1 1 1 1]'];
        pRatios=[pRatios;];
    %Hybrid_Kevin_05252015 file2:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (bad), FDP (good)
        classification=[classification;[0 1 1 1 0 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_05262015:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (bad)
        classification=[classification;[0 1 1 1 1 0]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_06032015 file1:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[0 1 1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_06042015 file1:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[0 1 1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_06062015 file1:
        %FCU (bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[1 1 1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Kevin_06082015 file1:
        %FCU (??, marked bad),FCR (good), ECU (good), ECR (good), FDS (good), FDP (good)
        classification=[classification;[0 1 1 1 1 1]'];
        pRatios=[pRatios;];
end
function [pRatios,classification]=jangoData()
    %Hybrid_Jango_07232014
        %FCU (good), FCR (??, marked bad), ECU (good), ECR (good)
        classification=[1 0 1 1]';
        pRatios=[];
        
    %Hybrid_Jango_07242014
        %FCU (good), FCR (??, marked bad), ECU (good), ECR (good)
        classification=[classification;[1 0 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_07252014
        %FCU (good), FCR (good), ECU (good), ECR (good)
        classification=[classification;[1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_08192014
        %FCU (good), FCR (good), ECU (good), ECR (good)
        classification=[classification;[1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_08202014
        %FCU (good), FCR (good), ECU (good), ECR (good)
        classification=[classification;[1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_08212014
        %FCU (good), FCR (good), ECU (good), ECR (good)
        classification=[classification;[1 1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_09232014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_09252014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_09262014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_10042014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_10102014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_10112014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_10122014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_11062014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
        
    %Hybrid_Jango_11072014
        %FCU (good), FCR (good), ECR (good)
        classification=[classification;[1 1 1]'];
        pRatios=[pRatios;];
end