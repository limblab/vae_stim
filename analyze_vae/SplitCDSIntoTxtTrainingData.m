%% Load cds File into td

filename = 'Han_20160315_RW_CDS_001.mat';


params.cont_signal_names = {'joint_ang','joint_vel','opensim_hand_pos','opensim_hand_vel','opensim_hand_acc','opensim_elbow_pos','opensim_elbow_vel','opensim_elbow_acc'};
params.array_name = 'S1';
td = loadTDfromCDS(['D:\Lab\Data\StimModel\cds\',filename] , params);

smoothParams.signals = {'joint_vel'};
smoothParams.width = 0.10;
smoothParams.calc_rate = false;
td = smoothSignals(td,smoothParams);
%% process file
% rebin 
td = binTD(td, 5);

%% load from .mat file

filename = 'Han_20160315_RW_CDS_001.mat';
pathname = 'D:\Lab\Data\StimModel';

load([pathname filesep filename]);

% Smooth kinematic variables
%%


%% extract opensim signals
opensim_sigs = {'joint_ang','joint_vel','opensim_hand_pos','opensim_hand_vel',...
    'opensim_elbow_pos','opensim_elbow_vel'};

joint_labels = {...
        'shoulder_adduction',...
        'shoulder_rotation',...
        'shoulder_flexion',...
        'elbow_flexion',...
        'radial_pronation',...
        'wrist_flexion',...
        'wrist_abduction',...
        };
for i_sig = 1:length(opensim_sigs)
    switch lower(opensim_sigs{i_sig})
        case 'pos'
            labels = {'x','y'};
        case 'vel'
            labels = {'vx','vy'};
        case 'acc'
            labels = {'ax','ay'};
        case 'force'
            labels = {'fx','fy','fz','mx','my','mz'};
        case 'motor_control'
            labels = {'MotorControlSho','MotorControlElb'};
        case 'joint_ang'
            labels = strcat(joint_labels,'_ang');
        case 'joint_vel'
            labels = strcat(joint_labels(),'_vel');
        case 'opensim_hand_pos'
            labels = strcat({'X','Y','Z'},{'_handPos'});
        case 'opensim_hand_vel'
            labels = strcat({'X','Y','Z'},{'_handVel'});
        case 'opensim_hand_acc'
            labels = strcat({'X','Y','Z'},{'_handAcc'});
        case 'opensim_elbow_pos'
            labels = strcat({'X','Y','Z'},{'_elbowPos'});
        case 'opensim_elbow_vel'
            labels = strcat({'X','Y','Z'},{'_elbowVel'});
        case 'opensim_elbow_acc'
            labels = strcat({'X','Y','Z'},{'_elbowAcc'});
    end
    
    % get label idx in opensim_names
    opensim_idx = [];
    for i_label = 1:numel(labels)
        opensim_idx(end+1) = find(strcmpi(td.joint,labels{i_label})==1);
    end
    
    % extract data
    td.(opensim_sigs{i_sig}) = td.opensim(:,opensim_idx);
    
end


%% downsample vel to have a uniform movement distribution
n_samps = ceil(length(td.opensim_hand_vel)*0.4);

ang = atan2(td.opensim_hand_vel(:,2),td.opensim_hand_vel(:,1));
edges = linspace(min(ang)-0.0001,max(ang)+0.0001,200);
[counts,edge,idx] = histcounts(ang,edges);

idx(idx==0) = nan;
weight_count = nan(size(idx));
weight_count(~isnan(idx)) = counts(idx(~isnan(idx)));

weight_mat = 1./weight_count;
weight_mat(isnan(weight_mat)) = 0;

train_idx = datasample((1:1:length(td.opensim_hand_vel)),n_samps,'Replace',false,'weights',weight_mat);

vel_samped = td.opensim_hand_vel(train_idx,:);
ang_samp = atan2(vel_samped(:,2),vel_samped(:,1));
histogram(ang_samp)

%% set nan's as -10000 so that python code doesn't fail

td.joint_vel(any(isnan(td.joint_vel),2),:) = [];

%% Extract joint_vel data in txt file
joint_vel = td.joint_vel;
% joint_vel_no_nan = joint_vel(all(joint_vel > -1000,2),:);

[~,mu,sigma] = zscore(joint_vel);
joint_vel = (joint_vel-mu)./sigma;
% joint_vel = fix(joint_vel * 10^6)/10^6;
% train_joint_vel = joint_vel(train_idx,:);

% writematrix(joint_vel, 'Han_20160325_RW_SmoothNormalizedJointVel_50ms.txt')
%%
dlmwrite([pathname,filesep,'Han_20160315_RW_SmoothRawJointVel_50ms.txt'],joint_vel,'delimiter',',','newline','pc')





