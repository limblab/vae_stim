pathname = 'D:\Lab\Data\StimModel';

%% make a joint vel entry into trial data 

td = td_all{1}{2};
td.joint_vel = td.opensim(:,8:14);

%% get deep lab cut idx for shoulder, elbow and hand
markernames = {'shoulder','elbow1','hand2'};
data = [];
for i_m = 1:numel(markernames)
    dlc_idx = find(~cellfun(@isempty,strfind(td.dlc_pos_names,markernames{i_m})));
    data = [data, td.dlc_pos(:,dlc_idx)];
end

td.joint_cart = data;
    
%%

% Smooth kinematic variables
smoothParams.signals = {'joint_vel','joint_cart'};
smoothParams.width = 0.10;
smoothParams.calc_rate = false;
td = smoothSignals(td,smoothParams);


%% Extract joint_vel data in txt file
joint_vel = td.joint_vel;
joint_vel = rmmissing(joint_vel);
joint_vel = zscore(joint_vel);
joint_vel = fix(joint_vel * 10^6)/10^6;

joint_cart = td.joint_cart;
joint_cart = rmmissing(joint_cart);
joint_cart = zscore(joint_cart);

% joint_cart = 2*(joint_cart-min(joint_cart))./(max(joint_cart)-min(joint_cart)) - 1;

joint_cart = fix(joint_cart*10^6)/10^6;
% writematrix(joint_vel, 'Han_20160325_RW_SmoothNormalizedJointVel_50ms.txt')
% dlmwrite([pathname,filesep,'Han_20201204_RT3D_SmoothNormalizedJointCart_50ms.txt'],joint_cart,'delimiter',',','newline','pc')

%%

firing_rates = readtable([pathname,filesep,'vae_rates_Chips_20151211_RW_dropout60_lambda0.1_learning5e-05_n-epochs2000_n-neurons900_2021-03-01-071739.csv']);
firing_rates = firing_rates{:,:};
%%

field_len = length(td.joint_cart);
td_fieldnames = fieldnames(td);
[~,mask] = rmmissing(td.joint_cart);

for i_field = 1:numel(td_fieldnames)
    if(length(td.(td_fieldnames{i_field})) == field_len)
        td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
    end
end

% add new firing rates to td
td.firing_rates = firing_rates;

% calculate PDs for signals
%call model output signals for this

markername = 'hand2';
dlc_idx = [find(strcmpi(td.dlc_pos_names,[markername,'_x'])),...
    find(strcmpi(td.dlc_pos_names,[markername,'_y'])),...
    find(strcmpi(td.dlc_pos_names,[markername,'_z']))];
                
td.dlc_vel_handxy = td.dlc_vel(:,dlc_idx);

pdParams = struct(...
    'out_signals','firing_rates',...
    'in_signals','dlc_vel_handxy',...
    'bootForTuning',false,...
    'num_boots',1,...
    'verbose',false,...
    'doing_3D_pd',true);
temp_pdTable = getTDPDs3D(td,pdParams);

% velPD =rad2deg(temp_pdTable.dlc_vel_handxyPD);
% velPD(velPD<0) = velPD(velPD<0)+360;
velPD = temp_pdTable.dlc_vel_handxyPD;
figure('Position',[680 558 1077 420]);
subplot(1,2,1)
rose(velPD);
velPD =reshape(velPD, [sqrt(length(velPD)),sqrt(length(velPD))]);

% Put PDs in 30x30 heatmap
subplot(1,2,2)
velPD_adj = velPD;
velPD_adj(velPD_adj<0) = velPD_adj(velPD_adj<0)+2*pi;
fig = imagesc(rad2deg(velPD_adj));
colormap(hsv);
% jet_wrap = vertcat(jet,flipud(jet));
% fig.Colormap = hsv;
% fig.Title='Heatmap of PDs from 30x30 area 2 neurons';
% fig.GridVisible = 'off';
colorbar