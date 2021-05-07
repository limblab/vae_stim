%% IF STARTING WITH CDS -Load and process cds File

%filename = 'Han_20160325_RW_CDS_001.mat';
%filename = 'Han_20170203_COactpas_CDS_001.mat';
%filename = 'Han_20160322_RW_CDS_001.mat';
filename = 'Han_20160315_RW_CDS_001.mat';

params.cont_signal_names = {'joint_vel','pos','vel','acc'};
params.array_name = 'S1';
trial_data = loadTDfromCDS(append('~/Documents/Documents/Thesis_Seminar/Model/data/',filename) , params);
%% Use trial data to process signals
%rebin to 50 ms
trial_data = binTD(trial_data, 5);
%% If starting with .mat file
%filename = 'Han_201702035_COactpas_SmoothKin.mat';
% pathname = '~/Documents/Documents/Thesis_Seminar/Model/data/';
% load([pathname filesep filename]);

%Smooth spikes
smoothParams.signals = {'S1_spikes'};
smoothParams.width = 0.03;
smoothParams.calc_rate = true;
td = smoothSignals(trial_data,smoothParams);

%Get speed
td.speed = sqrt(td.vel(:,1).^2 + td.vel(:,2).^2);

%Get rectified velocity
td.vel_rect = [td.vel(:,1) td.vel(:,2) -td.vel(:,1) -td.vel(:,2)];
td.vel_rect(td.vel_rect < 0) = 0;

%Get accel
td.acc = diff(td.vel)./td.bin_size;
td.acc(end+1,:) = td.acc(end,:);

%Remove offset
td.pos(:,1) = td.pos(:,1)+0;
td.pos(:,2) = td.pos(:,2)+32;

% Smooth kinematic variables
smoothParams.signals = {'pos','vel','acc','joint_vel'};
smoothParams.width = 0.10;
smoothParams.calc_rate = false;
td = smoothSignals(td,smoothParams);

%Get rid of unsorted units
sorted_idx = find(td.S1_unit_guide(:,2)~=0);
td.S1_spikes = td.S1_spikes(:,sorted_idx);
td.S1_unit_guide = td.S1_unit_guide(sorted_idx,:);


save('Han_201603015_RW_SmoothKin_50ms.mat')

%% Extract joint_vel data in txt file
nan = find(isnan(td.joint_vel));
td.joint_vel(nan) = 0;
td.joint_vel = normalize(td.joint_vel, 'range', [-1 1]);
joint_vel = td.joint_vel;
joint_vel = fix(joint_vel * 10^6)/10^6;
writematrix(joint_vel, 'Han_20160315_RW_SmoothNormalizedJointVel_50ms.txt')

