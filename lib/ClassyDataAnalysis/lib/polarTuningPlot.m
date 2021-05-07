function [H,axisID]=polarTuningPlot(dirs,activity,varargin)
    %function to taketuning data and plot a tuning circle
    %polarTuningPlot(dirs,activity)
    %  generates a circular plot of activity against dirs. polarTuningPlot
    %  closes the tuning circle so you don't need to manually append the
    %  first point to the end of the data
    %[H]=polarTuningPlot(...)
    %  returns a handle to the figure
    %[H,axisID]=polarTuningPlot(...)
    %  returns a handle to the axes the plot is on
    %[...]=polarTuningPlot(...,'key',value)
    %  accepts key/value pairs to provide additional functionality:
    %   'PD': 3 elemement vector [PD, CI_low, CI_high]
    %   'useAxis': axis handle to plot on
    %   'useColor': string with a color-spec character, forces the plotting
    %     color. Default color is 'b'
    %   'useScale': scalar, forces the radial scale by plotting a point at
    %     the value passed
    
    if ~(numel(dirs)==numel(activity))
        error('polarTuningPlot:mismatchedData','data must be in vectors of direction/amplitude pairs')
    end
    if min(size(dirs))>1 || min(size(activity))>1
        error('polarTuningPlot:nonVectorData','data must be in vectors')
    end
    
    
    if ~isempty(varargin)
        if mod(numel(varargin),2)
            error('polarTuning:oddNumberOptionalArguments','optional arguments must be in key-value pairs')
        end
        for i=1:2:numel(varargin)
            switch(varargin{i})
                case 'PD'
                    PD=varargin{i+1};
                case 'CI'
                    CI=varargin{i+1};
                case 'useAxis'
                    axisID=varargin{i+1};
                    H=axisID.Parent;
                case 'useColor'
                    colorID=varargin{i+1};
                case 'useScale'
                    rScale=varargin{i+1};
                otherwise
                    error('polarTuning:unrecognizedKey',['did not recognize the key: ', varargin{i}])
            end
        end
    end
    if ~exist('colorID','var')
        colorID='b';
    end
    
    if size(activity,1)>1
        activity=reshape(activity,[1,numel(activity)]);
    end
    if size(dirs,1)>1
        dirs=reshape(dirs,[1,numel(dirs)]);
    end
    if ~exist('axisID','var')
        H=figure;
        axisID=axes;
    end
    if exist('useScale','var')
        %draw a point to set the polar scale
        polar(0,rScale,'k')
    end
    
    if exist('CI','var')
        %add CI to the plot (this goes before the actual polar plot so we 
        %can set the range if needed):
        if ~exist('useScale','var')
            %draw a point using the max of the CI to force the scale:
            polar(0,max(max((CI))),'k')
        end
        if size(CI,2)>2
            CI=CI';
        end
        [xCart,yCart]=pol2cart((pi/180)*[dirs,dirs(1),dirs(end:-1:1),dirs(end)]',[CI(:,1);CI(1,1);CI(end:-1:1,2);CI(end,2)]);
        axes(axisID);%force axes to be current for patch command, should be unneccessary
        patch(xCart,yCart,colorID,'FaceAlpha',.3,'EdgeColor','none')
        hold on
    end
    polar(axisID,(pi/180)*[dirs,dirs(1)],[activity,activity(1)],colorID)
    if exist('PD','var')
        %add a PD line to the plot:
        rMax=max(activity);
        hold on
        polar(axisID,[PD(1),PD(1)],[0,rMax],colorID)
        [xCart,yCart]=pol2cart([PD(2), PD(2), PD(3), PD(3) ],[0,rMax,rMax,0]);
        axes(axisID);%force axes to be current for patch command, should be unneccessary
        patch(xCart,yCart,colorID,'FaceAlpha',.3,'EdgeColor','none')
    end

end


