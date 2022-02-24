%% Load cds File into td

filename = 'Han_20160315_RW_CDS_001.mat';


params.cont_signal_names = {'joint_ang','joint_vel',...
    'opensim_hand_pos','opensim_hand_vel','opensim_hand_acc',...
    'opensim_elbow_pos','opensim_elbow_vel','opensim_elbow_acc',...
    'muscle_len','muscle_vel'};
params.array_name = 'S1';
td = loadTDfromCDS(['D:\Lab\Data\StimModel\cds\',filename] , params);

smoothParams.signals = {'joint_vel','opensim_hand_vel','muscle_vel'};
smoothParams.width = 0.10;
smoothParams.calc_rate = false;
td = smoothSignals(td,smoothParams);
%% process file
% rebin 
td = binTD(td, 5);


%% get during rewarded trial mask

during_rewarded_trial = zeros(size(td.joint_vel,1),1);

for i_start = 1:numel(td.idx_startTime)
    if(td.result(i_start)=='R')
        during_rewarded_trial(td.idx_startTime(i_start):td.idx_endTime(i_start)) = 1;
    end 
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

sub_joint_vel_norm = joint_vel_norm(train_idx,:);

%% Extract joint_vel data in txt file
joint_vel = td.joint_vel;
% joint_vel_no_nan = joint_vel(all(joint_vel > -1000,2),:);

[~,mu,sigma] = zscore(joint_vel);
joint_vel_norm = (joint_vel-mu)./sigma;
% joint_vel = fix(joint_vel * 10^6)/10^6;
% train_joint_vel = joint_vel(train_idx,:);

% writematrix(joint_vel, 'Han_20160325_RW_SmoothNormalizedJointVel_50ms.txt')
%%
muscle_vel = td.muscle_vel;
muscle_len = td.muscle_len;

muscle_vel = fillmissing(muscle_vel,'constant',0);
muscle_len = fillmissing(muscle_len,'constant',0);
[~,mu,sigma] = zscore(muscle_vel);
norm_muscle_vel = (muscle_vel(train_idx,:)-mu)./sigma;

hand_vel = td.opensim_hand_vel;
hand_vel = fillmissing(hand_vel,'constant',0);
[~,mu,sigma] = zscore(hand_vel);
norm_hand_vel = (hand_vel(train_idx,:)-mu)./sigma;
hand_vel = hand_vel(train_idx,:);
during_rewarded_trial_train = during_rewarded_trial(train_idx);

joint_vel = td.joint_vel;
joint_vel = fillmissing(joint_vel,'constant',0);
[~,mu,sigma] = zscore(joint_vel);
norm_joint_vel = (joint_vel(train_idx,:)-mu)./sigma;
joint_vel = joint_vel(train_idx,:);
joint_ang = td.joint_ang(train_idx,:);
%%
dlmwrite([pathname,filesep,'Han_20160315_RW_SmoothNormalizedJointVel_uniformAngDist_50ms.txt'],sub_joint_vel_norm,'delimiter',',','newline','pc')
dlmwrite([pathname,filesep,'Han_20160315_RW_SmoothNormalizedJointVel_50ms.txt'],joint_vel_norm,'delimiter',',','newline','pc')





