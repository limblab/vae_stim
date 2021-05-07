function affine_xform = loadRawMarkerData(cds,marker_data_path,affine_xform)
% Given a cds, marker data from color tracking, and a save location, will
% spatiotemporally align the markers to the data in the CDS, smooth the
% markers, transform the coordinates for use in OpenSim and load into cds
%
% This function needs the KinectTracking library to operate
%% load marker data file
md = load(marker_data_path);

%% 4. PUT KINECT MARKER LOCATIONS IN HANDLE COORDINATES
% rotation_known=0; %Whether the rotation matrix is already known (from another file from that day)
% figure out if rotation known
if nargin>3
    error('Too many arguments')
elseif nargin==3
    [md,affine_xform] = realignMarkerSpacetime(cds,md,affine_xform);
else
    % first file of the day, affine xform unknown
    [md,affine_xform] = realignMarkerSpacetime(cds,md);
end


%% 5. SMOOTH OUT MARKERS

md = smoothMarkerData(md);

%% 6. PUT KINECT DATA INTO OPENSIM COORDINATES

[md,~] = transformForOpenSim(md,cds);

%% make marker table
% find meta data
num_markers = 10; % ONLY USED 10 MARKERS FOR ROBOT DATA
start_idx = find(md.t>=0,1,'first');
num_frames = length(md.t)-start_idx+1;
marker_names = {'Marker_1','Marker_2','Marker_3','Marker_4','Marker_5','Marker_6','Marker_7','Marker_8','Shoulder_JC','Pronation_Pt1'};

% convert to table
marker_time = md.t(start_idx:end);
marker_pos = md.pos(:,:,start_idx:end);
md_table = table;
md_table.Frame = (1:num_frames)';
md_table.t= marker_time';
for fn = 1:num_markers
    md_table.(marker_names{fn}) = squeeze(marker_pos(fn,:,:))';
end

%% add to CDS (passed by reference)
%append new data into the analog cell array:
%stick the data in a new cell at the end of the cds.analog
%cell array:
cds.analog{end+1}=md_table;

% set new data window
cds.setDataWindow()

logStruct=struct('fileName',marker_data_path);
evntData=loggingListenerEventData('loadRawMarkerData',logStruct);
notify(cds,'ranOperation',evntData)

