%% takes in PD differences with surrounding neighbors and finds percentage of all the differences
% in 20? bins to 180




for n = 1: numel(PDDiffNeighbors)
    if PDDiffNeighbors(n) > 180
        PDDiffNeighbors(n) = abs(PDDiffNeighbors(n) - 360);
    else
        PDDiffNeighbors(n) = PDDiffNeighbors(n);
    end
end

PDDiffNeighbors = PDDiffNeighbors';

for n = 1:numel(PDDiffNeighbors)
    if PDDiffNeighbors(n) < 20
        PDDiff(n,1) = PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 40
        PDDiff(n,2)= PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 60
        PDDiff(n,3)= PDDiffNeighbors(n);     
    elseif PDDiffNeighbors(n) < 80
        PDDiff(n,4)= PDDiffNeighbors(n);      
    elseif PDDiffNeighbors(n) < 100
        PDDiff(n,5)= PDDiffNeighbors(n);      
    elseif PDDiffNeighbors(n) < 120
        PDDiff(n,6)= PDDiffNeighbors(n);       
    elseif PDDiffNeighbors(n) < 140
        PDDiff(n,7)= PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 140
        PDDiff(n,8)= PDDiffNeighbors(n);
    else
        PDDiff(n,9)= PDDiffNeighbors(n);
    end
end

for n = 1: numel(PDDiff(1,:))
    x = find(PDDiff(:,n));
    PDs(n) = 100*(numel(x)/ numel(PDDiffNeighbors));
end

eval(string('PDs' + in))