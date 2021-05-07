function plotMovie(seq, xspec, varargin)
%
% plot3D(seq, xspec, ...)
%
% Plot neural trajectories in a three-dimensional space.
%
% INPUTS:
%
% seq        - data structure containing extracted trajectories
% xspec      - field name of trajectories in 'seq' to be plotted 
%              (e.g., 'xorth' or 'xsm')
%
% OPTIONAL ARGUMENTS:
%
% dimsToPlot - selects three dimensions in seq.(xspec) to plot 
%              (default: 1:3)
% nPlotMax   - maximum number of trials to plot (default: 20)
% redTrials  - vector of trialIds whose trajectories are plotted in red
%              (default: [])
%
% @ 2009 Byron Yu -- byronyu@stanford.edu

  dimsToPlot = 1:3;
  nPlotMax   = 50;
  redTrials  = [];
  blueTrials = [];
  greenTrials = [];
  fileName = 'NewFile.avi';
  assignopts(who, varargin);
    v = VideoWriter(fileName);
    v.FrameRate = 8;
    open(v);

  if size(seq(1).(xspec), 1) < 3
    fprintf('ERROR: Trajectories have less than 3 dimensions.\n');
    return
  end

  f = figure;
  pos = get(gcf, 'position');
  set(f, 'position', [pos(1) pos(2) 1.3*pos(3) 1.3*pos(4)]);
  
  for  n = 1: (length(seq))
    dat1{n} = seq(n).(xspec)(dimsToPlot,:);
  end
  seqBlack = [];
  maxTrialLength = max(cellfun(@length, dat1));
  redPoints = cell(maxTrialLength,1); 
  greenPoints =cell(maxTrialLength,1); 
  bluePoints =cell(maxTrialLength,1); 
  blackPoints = cell(maxTrialLength,1);

  for i = 1:max(cellfun(@length, dat1)); % iterate through time steps
      for j = 1: length(seq) % iterate through trials
          if ismember(seq(j).trialId, redTrials)
              col = [1 0 0]; % red
              lw  = .5;
              if i <= length(dat1{j})
                redPoints{i}(:,end+1) = dat1{j}(:,i);
              end
        elseif  ismember(seq(j).trialId, blueTrials)
            col = [0 0 1];
            lw = .5;
            if i <= length(dat1{j})
                bluePoints{i}(:,end+1) = dat1{j}(:,i);
            end
        elseif ismember(seq(j).trialId, greenTrials)
            col = [0 1 0];
            lw = .5;
          if i <= length(dat1{j})
            greenPoints{i}(:,end+1) = dat1{j}(:,i);
          end
        else
          col = 0.2 * [1 1 1]; % gray
          lw = 0.5;
        if i <= length(dat1{j})
          blackPoints{i}(:,end+1) = dat1{j}(:,i);
        end
          end
      %{
      if i <= length(dat1{j})
        scatter(dat1{j}(1,i), dat1{j}(2,i), 40, col);
      else 
        scatter(dat1{j}(1, end), dat1{j}(2,end), 40, col);
      end
      hold on
      %}
      end
      %{
      axis equal;
      mov(i) = getframe;
      writeVideo(v, mov(i));
      if i ~= max(cellfun(@length, dat1))
            cla
      end
      %}
      %}
      if ~isempty(redPoints{i})  
        meanRed(i,:) = mean(redPoints{i},2);
      end
      if ~isempty(bluePoints{i})
        meanBlue(i,:) = mean(bluePoints{i},2);
      end
      if ~isempty(greenPoints{i})
        meanGreen(i,:) = mean(greenPoints{i},2);
      end
      if ~isempty(blackPoints{i})
        meanBlack(i,:) = mean(blackPoints{i},2);
      end
  end
  figure
  xmax = max(vertcat((meanBlack(:,1)), (meanRed(:,1)), (meanBlue(:,1)), (meanGreen(:,1))));
  xmin = min(vertcat((meanBlack(:,1)), (meanRed(:,1)), (meanBlue(:,1)), (meanGreen(:,1))));
  ymax = max(vertcat((meanBlack(:,2)), (meanRed(:,2)), (meanBlue(:,2)), (meanGreen(:,2))));
  ymin = min(vertcat((meanBlack(:,2)), (meanRed(:,2)), (meanBlue(:,2)), (meanGreen(:,2))));
  for i = 1:min([length(meanBlack(:,1)), length(meanRed(:,1)), length(meanBlue(:,1)), length(meanGreen(:,1))])
      scatter(meanBlack(i, 1), meanBlack(i,2), 'k')
      xlim([xmin, xmax])
      ylim([ymin, ymax])
      hold on
      scatter(meanRed(i,1), meanRed(i,2), 'r')
      scatter(meanBlue(i,1), meanBlue(i,2), 'b')
      scatter(meanGreen(i,1), meanGreen(i,2), 'g')
      mov(i) = getframe;
      writeVideo(v, mov(i));
      if i ~= min([length(meanBlack(:,1)), length(meanRed(:,1)), length(meanBlue(:,1)), length(meanGreen(:,1))])
          cla
      end

      
  end
  if isequal(xspec, 'xorth')
    str1 = sprintf('$$\\tilde{\\mathbf x}_{%d,:}$$', dimsToPlot(1));
    str2 = sprintf('$$\\tilde{\\mathbf x}_{%d,:}$$', dimsToPlot(2));
    str3 = sprintf('$$\\tilde{\\mathbf x}_{%d,:}$$', dimsToPlot(3));
  else
    str1 = sprintf('$${\\mathbf x}_{%d,:}$$', dimsToPlot(1));
    str2 = sprintf('$${\\mathbf x}_{%d,:}$$', dimsToPlot(2));
    str3 = sprintf('$${\\mathbf x}_{%d,:}$$', dimsToPlot(3));
  end
  xlabel(str1, 'interpreter', 'latex', 'fontsize', 24);
  ylabel(str2, 'interpreter', 'latex', 'fontsize', 24);
  zlabel(str3, 'interpreter', 'latex', 'fontsize', 24);
  close(v);

  
  

