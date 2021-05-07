function RList=power60HzRatio(x)
    %takes a table, where the first column is time (t), and computes the
    %ratio of power in the 60hz harmonics to the total power, the ratio 
    %power of power in the 60hz harmonics to power around 200hz and the
    %ratio of power in the 60hz harmonics to power in the bands immediately
    %adjacent to the harmonics. Results will be presented in tabular
    %format, with each column represinting a ratio, and each row
    %corresponding to one of the columns in x. Only the first 3 harmonics
    %will be considered (60,120 and 180hz). Power at the harmonics is
    %computed in a 10Hz window (+/-5hz on either side of the harmonic).
    %Power adjacent to the harmonic is computed as the mean of the power in
    %the 5Hz bin just below and the 5Hz bin just above the harmonic. E.g.
    %at the 60Hz harmonic the harmonic power is computed from 55-65Hz, and
    %the adjacent power is computed from 50-55Hz, and 65-70Hz. Power around
    %200Hz is computed in the 185-225Hz window
    
    F=1/mode(diff(x.t));
    RList=zeros(size(x,2)-1,9);
    for i=2:size(x,2)
        %for each EMG we have, get the power of the whole signal, and the
        %60hz power
        pAll=bandpower(x{:,i},F,[0 F/2]);
        p60=bandpower(x{:,i},F,[55 65]);
        p60Fringe=(bandpower(x{:,i},F,[50 55])+bandpower(x{:,i},F,[65 70]))/2;
        p120=bandpower(x{:,i},F,[115 125]);
        p120Fringe=(bandpower(x{:,i},F,[110 115])+bandpower(x{:,i},F,[125 130]))/2;
        p180=bandpower(x{:,i},F,[175 185]);
        p180Fringe=(bandpower(x{:,i},F,[160 175])+bandpower(x{:,i},F,[185 190]))/2;
        p200=bandpower(x{:,i},F,[185 225]);
        RList(i,:)=[p60/pAll,p60/p200,p60/p60Fringe,...
                    p120/pAll,p120/p200,p120/p120Fringe,...
                    p180/pAll,p180/p200,p180/p180Fringe];
    end
    labels={'60_All','60_200','60_60Fringe',...
            '120_All','120_200','120_60Fringe',...
            '180_All','180_200','180_60Fringe'};
    RList=array2table(RList,'VariableNames',labels);
        
end