%% Load cds File into td

filename = 'Han_20170203_COactpas_SmoothKin_50m.mat';


params.cont_signal_names = {'joint_vel'};
params.array_name = 'S1';
trial_data = loadTDfromCDS(append('~/Documents/Documents/Thesis_Seminar/Model/data/',filename) , params);
%% process file
% rebin 
trial_data = binTD(trial_data, 5);

%% load from .mat file

filename = 'Han_201603015_RW_SmoothKin_50ms.mat';
pathname = 'D:\Lab\Data\StimModel';

load([pathname filesep filename]);

% Smooth kinematic variables
% smoothParams.signals = {'joint_vel'};
% smoothParams.width = 0.10;
% smoothParams.calc_rate = false;
% td = smoothSignals(trial_data,smoothParams);


%% downsample vel to have a uniform movement distribution
n_samps = ceil(length(td.vel)*3/7);

ang = atan2(td.vel(:,2),td.vel(:,1));
edges = linspace(min(ang)-0.0001,max(ang)+0.0001,200);
[counts,edge,idx] = histcounts(ang,edges);

idx(idx==0) = nan;
weight_count = nan(size(idx));
weight_count(~isnan(idx)) = counts(idx(~isnan(idx)));

weight_mat = 1./weight_count;
weight_mat(isnan(weight_mat)) = 0;

train_idx = datasample((1:1:length(td.vel)),n_samps,'Replace',false,'weights',weight_mat);

vel_samped = td.vel(train_idx,:);
ang_samp = atan2(vel_samped(:,2),vel_samped(:,1));
histogram(ang_samp)

%% Extract joint_vel data in txt file
joint_vel = td.joint_vel;
joint_vel_no_nan = rmmissing(joint_vel);
[min_val] = min(joint_vel);
max_val = max(joint_vel);
joint_vel = (joint_vel-min_val)./(max_val-min_val);
% [~,mu,sigma] = zscore(joint_vel_no_nan);
% joint_vel = (joint_vel-mu)./sigma;
joint_vel = fix(joint_vel * 10^6)/10^6;
% train_joint_vel = joint_vel(train_idx,:);
% writematrix(joint_vel, 'Han_20160325_RW_SmoothNormalizedJointVel_50ms.txt')
dlmwrite([pathname,filesep,'Han_20160315_RW_SmoothNormalizedRangeJointVel_50ms.txt'],joint_vel,'delimiter',',','newline','pc')





