function [tunedNeurons] = irisPlot( PM_pdData, DL_pdData)
%IRISPLOT Create iris plot for PM and DL
%   Inputs - PM and DL pdData tables with PDs calculated (extract from
%   binnedData object), logical array of neurons to plot (if tuned)
%   Outputs - logical array of which neurons were plotted, based on
%   original data

%extract relevant information
angsPM = PM_pdData.velPD;
dirCIPM = PM_pdData.velPDCI;
angsDL = DL_pdData.velPD;
dirCIDL = DL_pdData.velPDCI;

% check tuned neurons
% isTuned_params = struct('move_corr','vel','CIthresh',pi/3);
% tunedNeurons = checkIsTuned(PM_pdData,isTuned_params)...
%             & checkIsTuned(DL_pdData,isTuned_params);
% 
% if(~isempty(which_neurons))
%     tunedNeurons = tunedNeurons & which_neurons;
% end
% angsPM = angsPM(which_neurons);
% angsDL = angsDL(which_neurons);

%plot circles
h=polar(linspace(-pi,pi,1000),ones(1,1000));
set(h,'linewidth',2,'color',[1 0 0])
hold all
h=polar(linspace(-pi,pi,1000),0.5*ones(1,1000));
set(h,'linewidth',2,'color',[0.6 0.5 0.7])

% plot changes with alpha dependent on CI width
for unit_ctr = 1:length(angsPM)
    h=polar(linspace(angsPM(unit_ctr),angsDL(unit_ctr),2),linspace(0.5,1,2));
    set(h,'linewidth',2,'color',[0.1 0.6 1])
end

%plot circles
h=polar(linspace(-pi,pi,1000),ones(1,1000));
set(h,'linewidth',2,'color',[1 0 0])
hold all
h=polar(linspace(-pi,pi,1000),0.5*ones(1,1000));
set(h,'linewidth',2,'color',[0.6 0.5 0.7])

set(findall(gcf, 'String','  0.2','-or','String','  0.4','-or','String','  0.6','-or','String','  0.8',...
        '-or','String','  1') ,'String', ' '); % remove a bunch of labels from the polar plot; radial and tangential'

end

