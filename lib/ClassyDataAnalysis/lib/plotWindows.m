function h=plotWindows(data,windows,colLabels,varargin)

    %plots columns of data where the first column of data exists in the
    %windows specified in windows
    mask=false(size(data,2),1);
    for i=1:size(windows,1)
        mask(data.t>windows(i,1) & data.t<windows(i,2))=true;
    end
    
    h=figure;
    hold on
    if ~isempty(varargin) && strcmp(varargin{1},'vs')
        %plot columns against eachother. Only works with 2 colLabels
        if numel(colLabels)~=2
            error()
        end
        if numel(varargin)>1
            plot(data.(colLabels{1})(mask),data.(colLabels{2})(mask),varargin{2:end})
        else
            plot(data.(colLabels{1})(mask),data.(colLabels{2})(mask))
        end
        xlabel(colLabels{1})
        ylabel(colLabels{2})
        title([colLabels{2} ' vs ' colLabels{1}])
    else
        for i=1:numel(colLabels)
            plot(data.t(mask),data.(colLabels{i})(mask),varargin{:})
        end
        xlabel('t')
    end