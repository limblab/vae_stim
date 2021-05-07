function varargout=PESTH(units,eventTimes,preEventWindow,postEventWindow,unitNum,varargin)
    %This is a method of the unitData class and should be saved in the
    %@unitData folder with the other class methods.
    %
    %operates on an unitData structure to produce a PESTH for the specified
    %unit. 
    %
    %unitData.PESTH(events,preEventWindow,postEventWindow,unitNum)
    %   generates a PESTH plot in a new figure.
    %   -events:a vector containing the event timestamps
    %   -preEventWindow: a scalar indicating the pre-event time to include
    %    in the histogram
    %   -postEventWindow: a scalar indicating the post evenet time to
    %    include in the histogram
    %   -unitNum: index of the unit to use in the unitData.data struct
    %    array
    %[histData]=PESTH(...)
    %   returns the histogram object containing data on the plot
    %[histData,histErr]=PESTH(...)
    %   returns the standard Deviation of the count values in the histogram
    %   bins
    %[histData,histErr,H]=PESTH(...)
    %   returns the handle of the figure
    %[histData,histErr,H,axisID]=PESTH(...)
    %   returns the handle of the axis on which the histogram was plotted
    %PESTH(...,'key',val)
    %   accepts key value pairs to specify details of operation:
    %   -'useAxis': axis handle to place histogram on. allows placement of
    %       histogram into subplot. If empty the 
    %   -'numBins':number of bins to use, default is 100. Can also feed an
    %    array of bin edges
    %   -'markZero':true/false, flags PESTH to draw a line at zero. Default
    %    is true
    %   -'zeroMarkColor': string specifying the color of the line at 0,
    %    default is red
    %   -'useRate': true/false, flags PESTH to use the rate (spikes/time) 
    %    rather than the count
    %   -'plotErr': true/false, flags PESTH to include error bars on the
    %    histogram
    
    %handle variable input:
    if ~isempty(varargin)
        if mod(numel(varargin),2)
            error('PESTH:oddNumberExtraArguments','variable inputs must be in key-value pairs')
        end
        for i=1:2:numel(varargin)
            switch varargin{i}
                case 'useAxis'
                    axisID=varargin{i+1};
                case 'numBins'
                    numBins=varargin{i+1};
                case 'markZero'
                    markZero=varargin{i+1};
                case 'zeroMarkColor'
                    zeroMarkColor=varargin{i+1};
                case 'useRate'
                    useRate=varargin{i+1};
                case 'plotErr'
                    plotErr=varargin{i+1};
                otherwise
                    error('PESTH:badKey',['did not recognize the key: ',varargin{i}])
            end
        end

    end
    if ~exist('numBins','var')
        numBins=100;
    end
    if ~exist('axisID','var')
        axisID=axes;
        H=get(axisID,'Parent');
    end
    if ~exist('markZero','var')
        markZero=true;
    end
    if ~exist('zeroMarkColor','var')
        zeroMarkColor='r';
    end
    if ~exist('useRate','var')
        useRate=false;
    end
    if ~exist('plotErr','var')
        plotErr=false;
    end
    %collect the spikes:
    spikes=[];
    for i=1:numel(eventTimes)
        spikeMask=units.data(unitNum).spikes.ts>eventTimes(i)-preEventWindow & units.data(unitNum).spikes.ts<eventTimes(i)+postEventWindow;
        spikes=[spikes; units.data(unitNum).spikes.ts(spikeMask)-eventTimes(i)];
    end
    %make the histogram plot:
    [histData.counts,histData.edges]=histcounts(spikes,numBins);
    if useRate
        histData.counts=histData.counts/numel(spikes)/mode(diff(histData.edges));
    end
    axes(axisID)%force the appropriate axis:
    bar(histData.edges(1:end-1)+mode(diff(histData.edges)/2),histData.counts)
    
    %use the bins in histData, and find the actual variability in the spike
    %data: 
    spikeCounts=nan(numel(eventTimes),numel(histData.counts));
    for i=1:numel(eventTimes)
        for j=1:numel(histData.counts)
            spikeCounts(i,j)=numel(find(units.data(unitNum).spikes.ts>eventTimes(i)+histData.edges(j) & units.data(unitNum).spikes.ts<eventTimes(i)+histData.edges(j+1)));
        end
    end
    
    histErr=std(spikeCounts)/size(spikeCounts,1);%std error
    if useRate
        histErr=histErr/mode(diff(histData.edges));
    end
    if plotErr
        hold on
        binCtrs=histData.edges(1:end-1)+mode(diff(histData.edges))/2;
        errorbar(axisID,binCtrs,histData.counts,histErr,'+k')
    end
    
    %if needed, plot our vertical line
    if markZero
        hold on
        plot([0 0],[0,max(get(axisID,'YTick'))],zeroMarkColor)
    end
    %assign outputs:
    if nargout>=1
        varargout{1}=histData;
    end
    if nargout>=2
        varargout{2}=histErr;
    end
    if nargout>=3
        varargout{3}=H;
    end
    if nargout>=4
        varargout{4}=axisID;
    end
end