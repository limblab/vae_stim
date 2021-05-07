%% load file
filename = 'Han_201603015_RW_SmoothKin_50ms.mat';

%Nathan's mac path
%   pathname = '~/Documents/Documents/Thesis_Seminar/Model/data/';

%Joe's windows path
pathname = 'D:\Lab\Data\StimModel';

load([pathname filesep filename]);

rel= version('-release');
is_old_matlab = str2num(rel(1:4)) < 2018;

%% load in firing rates
%fr_file = 'vae_rates_Han_20160325_RW_dropout90_lambda1_learning5e-05_n-epochs1500_n-neurons1600_2021-03-02-032351.csv';
%fr_file = 'vae_rates_Han_20160325_RW_dropout95_lambda1_learning5e-05_n-epochs1500_n-neurons1600_2021-03-02-032351.csv';
%fr_file = 'firing_rates_20210223.csv';
fr_file = 'vae_rates_Han_20160325_RW_dropout91_lambda20_learning1e-05_n-epochs600_n-neurons1600_rate6.0_2021-03-13-184825.csv';

%make firing rate array
firing_rates = readtable([pathname,filesep, fr_file]);
firing_rates = firing_rates{:,:};

%add to td
td.VAE_firing_rates = firing_rates(:,:);

% map dimensions
map = sqrt(numel(firing_rates(1,:)));
map = [map map];

%% IF USING PARTIAL DATA SET: cut off first elements of td to match td time bins with firing_rates
% td.acc(1:30000,:) = [];
% td.pos(1:30000,:) = [];
% td.vel(1:30000,:) = [];
% td.joint_vel(1:30000,:) = [];
% td.S1_spikes(1:30000,:) = [];
% td.speed(1:30000,:) = [];
% td.vel_rect(1:30000,:) = [];

%% find decoder predictors of hand velocities from firing rates
hand_vel = td.train_vel;
fr = td.VAE_firing_rates(td.train_idx,:);

%% match up data lengths

field_len = length(td.vel);
td_fieldnames = fieldnames(td);
[~,mask] = rmmissing(td.vel);

for i_field = 1:numel(td_fieldnames)
    if(length(td.(td_fieldnames{i_field})) == field_len)
        td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
    end
end




%% perform dropout reguralization
lr = 0.001;
dec = rand(size(fr,2),2)-0.5;
bias = [0,0];
num_iters = 5000;
dropout_rate = 0.93;

vaf_list_drop = zeros(num_iters,2);
vaf_list_mdl = zeros(num_iters,2);
n_neurons = size(fr,2);
for i_iter = 1:num_iters
    % dropout inputs
    keep_mask = zeros(n_neurons,1);
    keep_mask(datasample(1:1:n_neurons,ceil(n_neurons*(1-dropout_rate)))) = 1;
    x = (fr.*keep_mask');
    
    vel_pred = x*dec + bias;
    
    d_dec = -2*x'*(hand_vel-vel_pred)/length(dec);
    d_bias = mean(-2*(hand_vel-vel_pred));
    
    bias = bias - lr*d_bias;
    dec = dec - lr*d_dec;
    
    vaf_list_drop(i_iter,:) = compute_vaf(hand_vel,vel_pred);
    vaf_list_mdl(i_iter,:) = compute_vaf(hand_vel,(fr*dec)*(1-dropout_rate) + bias);
    
    if(mod(i_iter,100)==0)
        disp(vaf_list_mdl(i_iter,:))
    end
end
    

%% adjust dec to deal with dropout rate
dec = dec*(1-dropout_rate);
%% load in decoder from file


filename = 'Han_20160315_RW_dec_bias_dropout91_fr6.mat';

load([pathname filesep filename]);

%% predict hand velocity and get vaf
hand_vel = td.vel;
hand_vel_hat = fr_lagged*dec + bias;
vaf_pred = compute_vaf(hand_vel,hand_vel_hat);
b = hand_vel_hat\hand_vel;

%% find predicted hand velocities using firing rates
% A*dec = b, A\b = dec

set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18); 
hold on
subplot(1,2,1)

plot(hand_vel(:,1),hand_vel_hat(:,1),'.');set(gca,'FontName','Helvetica','FontSize',18);
hold on;
plot([-30,30],[-30,30],'k--');
ylabel('Decoded hand x-velocity (cm/s)')
xlabel('Hand x-velocity (cm/s)')
%title('Decoder Performance');
subplot(1,2,2)

plot(hand_vel(:,2),hand_vel_hat(:,2),'.');set(gca,'FontName','Helvetica','FontSize',18);
hold on;
plot([-30,30],[-30,30],'k--');
ylabel('Decoded hand y-velocity (cm/s)')
xlabel('Hand y-velocity (cm/s)')


%% split tds
%Split TD
splitParams.split_idx_name = 'idx_startTime';
splitParams.linked_fields = {'trialID','result'};
td_trim = splitTD(td,splitParams);

% trim tds
% td_trim = trimTD(td_trim,{'idx_startTime',15},{'idx_startTime',30});

%% get rid of non-reward trials
x=[td_trim.result]=='R';
td_trim = td_trim(x);

%% plot reaches
% for trial= 1:length(td_trim)
%     reach = plot(td_trim(trial).pos(:,1),td_trim(trial).pos(:,2));
%     hold on
% end
% hold off

%% calculate PDs for signals
%call model output signals for this
params.out_signals = 'VAE_firing_rates';
params.in_signals = {'vel'};
params.num_boots = 0;
pdtable = getTDPDs(td_trim, params);

pdtable =rad2deg(pdtable.velPD);
pdtable(pdtable<0) = pdtable(pdtable<0)+360;
figure();
hist(pdtable);
pdtable =reshape(pdtable, map);

for n = 1:numel(dec(:,1))
    pdtable_decoded(n) = atan2(dec(n,2), dec(n,1));
end

pdtable_decoded =rad2deg(pdtable_decoded);
pdtable_decoded(pdtable_decoded<0) = pdtable_decoded(pdtable_decoded<0)+360;
figure();
hist(pdtable_decoded);
pdtable_decoded =reshape(pdtable_decoded, map);

 %% histogram of angular differences
% for x = 1:numel(pdtable)
%     PDdiff(x) = angleDiff(pdtable(x),pdtable_decoded(x),0);
% end
% figure();set(gcf,'Color','White')
% hist(PDdiff,20)
% title('Difference between actual PDs and decoder PDs')
% xticks([-180 0 180])
% xlabel('Angular difference (actual - decoded = difference)')

%% Circular Histogram of PDs
theta = deg2rad(pdtable);
theta = reshape((theta).', 1, []);
his = polarhistogram(theta, 36);set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
title('Distribution of PDs (degrees)')

%% Put PDs in 30x30 heatmap
if(~is_old_matlab)
    fig = heatmap(pdtable);set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
    fig.CellLabelColor = 'none';
    % jet_wrap = vertcat(jet,flipud(jet));
    fig.Colormap = hsv;
    fig.Title='Heatmap of PDs from 40x40 area 2 neurons (degrees)';

    fig.GridVisible = 'off';
    fig.XDisplayLabels = nan(size(pdtable(:,1)));
    fig.YDisplayLabels = nan(size(pdtable(:,1)));
    xlabel('Neurons (n = 40)')
    ylabel('Neurons (n = 40)')
else
    fig = imagesc(pdtable);
    colormap(colorcet('C4'));
    
    colorbar;
end
%% Get grid of X and Y coords for 30
clear neuron_location
num_rows = map(1);
num_cols = map(2);
[i,j] = ind2sub([num_rows,num_cols],1:num_rows*num_cols);
neuron_location(:,1) = i';
neuron_location(:,2) = j';

%% histogram of neighboring PDs vs non-neighboring PDs
neuron = randsample(numel(pdtable), 50);
rad = 2;

for ns = 1:numel(neuron)
    i(ns) = neuron_location(neuron(ns),1);
    j(ns) = neuron_location(neuron(ns),2);
    for n = 1:numel(neuron_location(:,1))
            if sqrt((neuron_location(n,1) - i(ns))^2 + (neuron_location(n,2)-j(ns))^2) <= rad
                PDDiffNeighbors(ns,n) = abs(pdtable(neuron(ns)) - pdtable(n));
            else
                PDDiffNonNeighbors(ns,n) = abs(pdtable(neuron(ns)) - pdtable(n));
            end
    end
end

x = find(PDDiffNeighbors);
PDDiffNeighbors = PDDiffNeighbors(x);
x = find(PDDiffNonNeighbors);
PDDiffNonNeighbors = PDDiffNonNeighbors(x);


for n = 1: numel(PDDiffNeighbors)
    if PDDiffNeighbors(n) > 180
        PDDiffNeighbors(n) = abs(PDDiffNeighbors(n) - 360);
    else
        PDDiffNeighbors(n) = PDDiffNeighbors(n);
    end
end

PDDiffNeighbors = PDDiffNeighbors';

for n = 1:numel(PDDiffNeighbors)
    if PDDiffNeighbors(n) < 20
        PDDiff(n,1) = PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 40
        PDDiff(n,2)= PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 60
        PDDiff(n,3)= PDDiffNeighbors(n);     
    elseif PDDiffNeighbors(n) < 80
        PDDiff(n,4)= PDDiffNeighbors(n);      
    elseif PDDiffNeighbors(n) < 100
        PDDiff(n,5)= PDDiffNeighbors(n);      
    elseif PDDiffNeighbors(n) < 120
        PDDiff(n,6)= PDDiffNeighbors(n);       
    elseif PDDiffNeighbors(n) < 140
        PDDiff(n,7)= PDDiffNeighbors(n);
    elseif PDDiffNeighbors(n) < 160
        PDDiff(n,8)= PDDiffNeighbors(n);
    else
        PDDiff(n,9)= PDDiffNeighbors(n);
    end
end

for n = 1: numel(PDDiff(1,:))
    x = find(PDDiff(:,n));
    PDs(n) = 100*(numel(x)/ numel(PDDiffNeighbors));
end
PDsNeighbor = PDs;


for n = 1: numel(PDDiffNonNeighbors)
    if PDDiffNonNeighbors(n) > 180
        PDDiffNonNeighbors(n) = abs(PDDiffNonNeighbors(n) - 360);
    else
        PDDiffNonNeighbors(n) = PDDiffNonNeighbors(n);
    end
end

PDDiffNonNeighbors = PDDiffNonNeighbors';
for n = 1:numel(PDDiffNonNeighbors)
    if PDDiffNonNeighbors(n) < 20
        PDDiff(n,1) = PDDiffNonNeighbors(n);
    elseif PDDiffNonNeighbors(n) < 40
        PDDiff(n,2)= PDDiffNonNeighbors(n);
    elseif PDDiffNonNeighbors(n) < 60
        PDDiff(n,3)= PDDiffNonNeighbors(n);     
    elseif PDDiffNonNeighbors(n) < 80
        PDDiff(n,4)= PDDiffNonNeighbors(n);      
    elseif PDDiffNonNeighbors(n) < 100
        PDDiff(n,5)= PDDiffNonNeighbors(n);      
    elseif PDDiffNonNeighbors(n) < 120
        PDDiff(n,6)= PDDiffNonNeighbors(n);       
    elseif PDDiffNonNeighbors(n) < 140
        PDDiff(n,7)= PDDiffNonNeighbors(n);
    elseif PDDiffNonNeighbors(n) < 160
        PDDiff(n,8)= PDDiffNonNeighbors(n);
    else
        PDDiff(n,9)= PDDiffNonNeighbors(n);
    end
end

for n = 1: numel(PDDiff(1,:))
    x = find(PDDiff(:,n));
    PDs(n) = 100*(numel(x)/ (numel(PDDiffNonNeighbors)));
end
PDscomb = [PDsNeighbor(:), PDs(:)];
x = 20:20: 180;        
bar(x, PDscomb, 'grouped'); set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
legend( 'neighbors','non-neighbors')
title('Difference in PD for neurons based on distance')
xlabel('Difference in direction (degrees)')
ylabel('Percentage')


clear PDDiff PDDiffNonNeighbors PDDiffNeighbors PDs PDsNeighbor


%% divide neurons into sections based on closest PDs
clear PD
pdDirs = [ 0 45 90 135 180 225 270 315 360];
for neur = 1:numel(pdtable)
    for dir = 1:numel(pdDirs)
        if pdtable(neur) > (pdDirs(dir) - 22.5) && pdtable(neur) < (pdDirs(dir) + 22.5)
            PD(dir, neur) = pdtable(neur);
        else
            PD(dir, neur) = 0;
        end
    end
end
PD= PD';

pd0 = find(PD(:,1) | PD(:,9));
pd45 = find(PD(:,2));
pd90 = find(PD(:,3));
pd135 = find(PD(:,4));
pd180 = find(PD(:,5));
pd225 = find(PD(:,6));
pd270 = find(PD(:,7));
pd315 = find(PD(:,8));

%% Select neuron to stim and calculate radius of activation
% select random neurons based on PD

neuron = [randsample(pd90, 30)]; %randsample(pd0,1), randsample(pd45,1),randsample(pd90,1),randsample(pd135,1),randsample(pd180,1),randsample(pd225,1),randsample(pd270,1),randsample(pd315,1)];
% 944 350 57 344 1332 1508 293 1390  -- good neurons for 240 dir
%select stim params
current = [ 15 30 50 100]; %current in µA
k = []; %0.100, 0.250, 0.325, 0.500]; %space constant in mm
clear rad;

%% activate using stoney circle activation
k = 1296; %space constant in mm
for i = 1:numel(current)
    r(i) = sqrt(current(i)/k); %radius in mm
    rad(i) = round(r(i)*10); %arbitrary radius of neural squares activated - given each neuron is 0.1 mm from each other 
end

for ns = 1:numel(neuron)
    i(ns) = neuron_location(neuron(ns),1);
    j(ns) = neuron_location(neuron(ns),2);
end

clear activate
for x =1:numel(rad)
    for n = 1:numel(neuron_location(:,1))
        for ns = 1:numel(neuron)
            if sqrt((neuron_location(n,1) - i(ns))^2 + (neuron_location(n,2)-j(ns))^2) <= rad(x)
                activate(x,ns).neuron(n) = 1;
            else
                activate(x,ns).neuron(n) = 0;
            end
        end
    end
end
clear activated
for x=1:numel(rad)
    for ns = 1:numel(neuron)
        activated(x,ns).rad= find(activate(x,ns).neuron);
    end
end

%% activate using probabalistic exponential decay
k = [0.100 0.250, 0.325, 0.500]; %0.100 ,  0.250, 0.325,  0.500]; %space constants in mm
clear P_act
clear activate
for x =1:numel(current)
    for n = 1:numel(neuron_location(:,1))
        for ns = 1:numel(neuron)
            i(ns) = neuron_location(neuron(ns),1);
            j(ns) = neuron_location(neuron(ns),2);
            dis(n) = (sqrt((neuron_location(n,1) - i(ns))^2 + (neuron_location(n,2)-j(ns))^2)/10);
            P_act(ns,x).spaceP(1,n) = exp(-dis(n)/k(x));
        end
    end
end


for x =1:numel(current) 
    for n = 1:numel(neuron_location(:,1)) %total neuron number
        for ns = 1:numel(neuron) %stim neuron number
            activator = rand(1);
            if activator < P_act(ns,x).spaceP(n)
                activate(x,ns).neuron(n) = 1;
            else
                activate(x,ns).neuron(n) = 0;
            end
        end
    end
end

clear Pactivated
for x=1:numel(current)
    for ns = 1:numel(neuron)
        Pactivated(x,ns).rad= find(activate(x,ns).neuron);
    end
end
              

%% activate neurons for a specific movement to a certain Hz
trial =[ 6 ]; %chose trial 
hz = 5; %firing rates from file (x spikes per bin)
binSize = 0.050; %bin size in seconds
rates_inHz = hz/binSize; %converting to firing rate to Hz  ~15 Hz for now
startMove_idx = 8; %time you want to start movement 3
stimStart_idx = startMove_idx + 1; %time you want to start stim at 4
stimLength_idx =stimStart_idx + 4; %how many bins you want to stim to last (4 bins @ 50 ms = 200 ms)
stimEnd_idx = stimLength_idx + 1; %time of start post-stim movement 
endMove_idx = stimEnd_idx + 1 ; %time you want to end movement 10
clear td_move
% plot movement reach
td_move = td_trim;
for x = 1:numel(trial)
    xoffset = td_move(trial(x)).pos(startMove_idx,1);
    yoffset = td_move(trial(x)).pos(startMove_idx,2);
    td_move(trial(x)).pos(startMove_idx:endMove_idx,1) = td_move(trial(x)).pos(startMove_idx:endMove_idx,1) - xoffset;
    td_move(trial(x)).pos(startMove_idx:endMove_idx,2) = td_move(trial(x)).pos(startMove_idx:endMove_idx,2) - yoffset;
    plot(td_move(trial(x)).pos(startMove_idx:endMove_idx,1),td_move(trial(x)).pos(startMove_idx:endMove_idx,2));set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
    hold on
    ax = gca;
    ax.XAxisLocation= 'origin';
    ax.YAxisLocation = 'origin';
    plot(td_move(trial(x)).pos(startMove_idx, 1),td_move(trial(x)).pos(startMove_idx,2), '*')
    plot(td_move(trial(x)).pos(stimStart_idx, 1),td_move(trial(x)).pos(stimStart_idx,2), '*')
    plot(td_move(trial(x)).pos(stimEnd_idx, 1),td_move(trial(x)).pos(stimEnd_idx,2), '*')
    plot(td_move(trial(x)).pos(endMove_idx, 1),td_move(trial(x)).pos(endMove_idx,2), '*')
    title('Hand Position During Movement Window')
    xlabel('x-hand position (cm)')
    ylabel('y-hand position (cm)')
    legend('movement', 'starting point', 'stim start', 'stim end' , 'end point', 'Location', 'northwest')
    xlim([-20 20])
    ylim([-20 20])

end

%% compare decoded handvel between stim and no stim between activation cases
clear compareDirs
for i = 1:numel(current)
    for t = 1:numel(trial)
        for ns = 1:numel(neuron)
            td_stim = td_trim; % resets previous stim, comment out if doing simultaneous stimulations
            for p = 1:numel(activated(i,ns).rad)
                td_stim(trial(t)).VAE_firing_rates(stimStart_idx:stimLength_idx,activated(i,ns).rad(p)) = hz;               
            end
            compareDirs(i,ns).CActivated = pdtable(activated(i,ns).rad); %the activated PDs of neurons from circular activation
            compareDirs(i,ns).decodedNoStim = atan2d(td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,2),td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,1));
            if compareDirs(i,ns).decodedNoStim < 0
                compareDirs(i,ns).decodedNoStim = compareDirs(i,ns).decodedNoStim + 360;
            end
            td_stim(trial(t)).hand_vel_hat(startMove_idx:endMove_idx,:) = td_stim(trial(t)).VAE_firing_rates(startMove_idx:endMove_idx,:)*dec + bias; %stim decoded hand vel
            compareDirs(i,ns).CdecodedStim = atan2d(td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,2),td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,1));
            if  compareDirs(i,ns).CdecodedStim < 0
                 compareDirs(i,ns).CdecodedStim =  compareDirs(i,ns).CdecodedStim + 360;
            end
            compareDirs(i,ns).Cdifference = compareDirs(i,ns).CdecodedStim - compareDirs(i,ns).decodedNoStim;
            td_stim = td_trim; % resets previous stim for Probabilistic activation, comment out if doing simultaneous stimulations
            td_stim(t).Phand_vel_hat = td_stim(t).hand_vel_hat;
            for p = 1:numel(Pactivated(i,ns).rad)
                td_stim(trial(t)).VAE_firing_rates(stimStart_idx:stimLength_idx,Pactivated(i,ns).rad(p)) = hz;
            end
            compareDirs(i,ns).PActivated = pdtable(Pactivated(i,ns).rad); %activated PDs of neurons from probabilistic
            td_stim(trial(t)).Phand_vel_hat(startMove_idx:endMove_idx,:) = td_stim(trial(t)).VAE_firing_rates(startMove_idx:endMove_idx,:)*dec + bias; %stim decoded hand vel
            compareDirs(i,ns).PdecodedStim = atan2d(td_stim(trial(t)).Phand_vel_hat(stimStart_idx + 2,2),td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,1));
            if compareDirs(i,ns).PdecodedStim < 0 
                compareDirs(i,ns).PdecodedStim = compareDirs(i,ns).PdecodedStim + 360;
            end
            compareDirs(i,ns).Pdifference = compareDirs(i,ns).PdecodedStim - compareDirs(i,ns).decodedNoStim;
        end        
        % plot differences
        figure();
        plot(1, mean([compareDirs(i,:).Cdifference]), 'k*', 'MarkerSize', 30)
        set(gcf,'Color','White');
        set(gca,'FontName','Helvetica','FontSize', 18);
        hold on
        errorbar(1, mean([compareDirs(i,:).Cdifference]), std([compareDirs(i,:).Cdifference])/sqrt(numel(neuron)), 'k', 'LineWidth', 3,'CapSize', 15)
        plot(2, mean([compareDirs(i,:).Pdifference]), 'r*', 'MarkerSize', 30)
        errorbar(2, mean([compareDirs(i,:).Pdifference]), std([compareDirs(i,:).Pdifference])/sqrt(numel(neuron)), 'r', 'LineWidth', 3, 'CapSize', 15)
        xlim([0 3])
        xticks([ 1 2])
        xticklabels( {'CA', 'PED'})
        title({'Angular Difference Due to Stimulation for'; 'Two Different Stimulation Methods'})
        ylabel('Angular Difference (degrees)')
        xlabel('Stimulation Method')
        figure();set(gcf,'Color','White');
        % plot activation spread
        subplot(1,2,1)
        polarhistogram([compareDirs(i,:).CActivated], 24, 'Normalization', 'probability', 'FaceColor', 'k', 'LineWidth', 2)
        sgtitle('Preferred Directions of Activated Neurons','FontName','Helvetica','FontSize',24)
        set(gcf,'Color','White');
        set(gca,'FontName','Helvetica','FontSize',18);
        thetaticks([0 90 180 270])
        rticks([])
        subplot(1,2,2)
        polarhistogram([compareDirs(i,:).PActivated], 24, 'Normalization', 'probability', 'FaceColor','r', 'LineWidth', 2)
        set(gcf,'Color','White');
        set(gca,'FontName','Helvetica','FontSize',18);
        thetaticks([0 90 180 270])
        rticks([])
    end
end
noStimDir = compareDirs(i,ns).decodedNoStim;

%% compare decoded handvel between current cases
str = '#0072BD';
r = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#D95319';
b = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#EDB120';
g = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#7E2F8E';
p = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
color = [r,b,g,p];


for i = 1:numel(current)
    for t = 1:numel(trial)
        for ns = 1:numel(neuron)
            td_stim = td_trim; % resets previous stim, comment out if doing simultaneous stimulations
            for p = 1:numel(Pactivated(i,ns).rad)
                td_stim(trial(t)).VAE_firing_rates(stimStart_idx:stimLength_idx,Pactivated(i,ns).rad(p)) = hz;
            end
            td_stim(trial(t)).hand_vel_hat(startMove_idx:endMove_idx,:) = td_stim(trial(t)).VAE_firing_rates(startMove_idx:endMove_idx,:)*dec + bias; %stim decoded hand vel
            compareDirs(i,ns).currentDirs = atan2d(td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,2),td_stim(trial(t)).hand_vel_hat(stimStart_idx + 2,1));
            if compareDirs(i,ns).currentDirs < 0
                compareDirs(i,ns).currentDirs = compareDirs(i,ns).currentDirs + 360;
            end
            compareDirs(i,ns).currentDiff = compareDirs(i,ns).currentDirs - noStimDir;
        end
    end
    %plot differences
    plot(i, mean([compareDirs(i,:).currentDiff]), '*', 'MarkerEdgeColor', color(((3*i)-2):((3*i)-2)+(numel(r)-1)), 'MarkerSize', 30)
    hold on
    errorbar(i, mean([compareDirs(i,:).currentDiff]), std([compareDirs(i,:).currentDiff])/sqrt(numel(neuron)), 'Color' ,color(((3*i)-2):((3*i)-2)+(numel(r)-1)), 'LineWidth', 3,'CapSize', 15)
    set(gcf,'Color','White');
    set(gca,'FontName','Helvetica','FontSize',18);
    xlim([0 5])
    xticks([1 2 3 4])
    xticklabels({'15 µA','30 µA','50 µA','100 µA'})
    xlabel('Current (µA)')
    ylabel('Angular Difference (degrees)')
    title({'Angular Difference Due to';'Different Current Amplitudes'})
end
figure();
for i = 1:numel(current)
%    plot activated PDs
    subplot(2,2,i)
    polarhistogram([compareDirs(i,:).PActivated], 24, 'Normalization', 'probability', 'FaceColor',color(((3*i)-2):((3*i)-2)+(numel(r)-1)), 'LineWidth', 2)
    set(gcf,'Color','White');
    set(gca,'FontName','Helvetica','FontSize',18);
    sgtitle('Preferred Directions of Activated Neurons','FontName','Helvetica','FontSize',24)
    thetaticks([0 90 180 270])
    rticks([])
end



%% Plot radius of activation of different stim currents during movements
pdtableDummy = pdtable;
for x = 1:numel(trial)
    for n = 1:numel(activated)
        td_stim = td_trim; % resets previous stim, comment out if doing simultaneous stimulations
        
        for p = 1:numel(activated(n).rad)
            td_stim(trial(x)).VAE_firing_rates(stimStart_idx:stimLength_idx,activated(n).rad(p)) = hz;
            pdtableDummy(activated(n).rad(p)) =  NaN;
        end
        during_stim = reshape(td_stim(trial(x)).VAE_firing_rates(stimStart_idx,:), map);
        if(~is_old_matlab)
            figure()
            heatmap(during_stim, 'GridVisible', 'off','Colormap', parula,'XDisplayLabels', nan(size(during_stim(:,1))),'YDisplayLabels', nan(size(during_stim(:,1)))); set(gcf,'Color','White');set(gcf,'Color','White'); 
            set(gca,'FontName','Helvetica','FontSize',18);
            figure()
            heatmap(pdtableDummy, 'GridVisible', 'off','Colormap', hsv,'XDisplayLabels', nan(size(during_stim(:,1))),'YDisplayLabels', nan(size(during_stim(:,1)))); set(gcf,'Color','White');set(gcf,'Color','White'); 
            set(gca,'FontName','Helvetica','FontSize',18);
        else
            imagesc(during_stim);
            colorbar;
        end
        title('PDs during stim') % for ' + string(current(n)) + ' µA');
        xlabel('40 Neurons')
        ylabel('40 Neurons')
    end
end
%% heatmap of pre-stim firing rates, stim firing rates, and post-stim firing rates using best current 
% Make sure to redo above section with best current if multiple (pick one in current array)
for t = 1:numel(trial)
    pre_stim = reshape(td_stim(trial(t)).VAE_firing_rates(startMove_idx,:), map);
    during_stim = reshape(td_stim(trial(t)).VAE_firing_rates(stimStart_idx,:), map);
    post_stim = reshape(td_stim(trial(t)).VAE_firing_rates(stimEnd_idx,:), map);
    
    if(~is_old_matlab)
        heatmap(pre_stim,'GridVisible', 'off','Colormap', parula,'XDisplayLabels', nan(size(during_stim(:,1))), 'YDisplayLabels', nan(size(during_stim(:,1)))); set(gcf,'Color','White'); 
        set(gca,'FontName','Helvetica','FontSize',18);
    else
        imagesc(pre_stim);
    end
    title('Firing rates (Spikes/50 ms) before stim')
    caxis([0 hz])
    xlabel('Neurons (n = 40)')
    ylabel('Neurons (n = 40)')
    figure()
    if(~is_old_matlab)
        heatmap(during_stim, 'GridVisible', 'off','Colormap', parula,'XDisplayLabels', nan(size(during_stim(:,1))), 'YDisplayLabels', nan(size(during_stim(:,1)))); set(gcf,'Color','White'); 
        set(gca,'FontName','Helvetica','FontSize',18);
    else
        imagesc(during_stim);
    end
    title('Firing rates (Spikes/50 ms) during stim ' + string(current(n)) + ' µA')
    caxis([0 hz])
    xlabel('Neurons (n = 40)')
    ylabel('Neurons (n = 40)')
    figure()
    if(~is_old_matlab)
        heatmap(post_stim, 'GridVisible', 'off','Colormap', parula, 'XDisplayLabels', nan(size(during_stim(:,1))), 'YDisplayLabels', nan(size(during_stim(:,1)))); set(gcf,'Color','White'); 
        set(gca,'FontName','Helvetica','FontSize',18);
    else
        imagesc(post_stim);
    end
    title('Firing rates (Spikes/50 ms) after stim')
    caxis([0 hz])
    xlabel('Neurons (n = 40)')
    ylabel('Neurons (n = 40)')
end


%% plot decoded movement
td_ex = td_trim; %example td

for x = 1:numel(trial)
%     td_ex(trial(x)).vel(startMove_idx:endMove_idx,:) = td_ex(trial(x)).VAE_firing_rates(startMove_idx:endMove_idx,:) * dec + bias; %regular decoded hand vel
    plot(td_trim(trial(x)).pos(startMove_idx:endMove_idx,1),td_trim(trial(x)).pos(startMove_idx:endMove_idx,2), 'k') %plot actual movement
    set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
    hold on;
    for time = startMove_idx+1:endMove_idx
        td_ex(trial(x)).pos(time,1) = td_ex(trial(x)).pos(time-1,1) + (td_ex(trial(x)).hand_vel_hat(time-1,1)*binSize);%stim effect on hand pos in x dir
        td_ex(trial(x)).pos(time,2) = td_ex(trial(x)).pos(time-1,2) + (td_ex(trial(x)).hand_vel_hat(time-1,2)*binSize);%stim effect on hand pos in y dir
    end
    plot(td_ex(trial(x)).pos(startMove_idx:endMove_idx,1),td_ex(trial(x)).pos(startMove_idx:endMove_idx,2), 'r') %plot decoded movement
    legend('actual movement', 'decoded movement', 'Location', 'northeast')
    xlabel('x-hand position (cm)')
    ylabel('y-hand position (cm)')
    title('Decoded Movement Vs Actual Movement')
end

%% Use decoder predictors to change velocity, compute movement output from velocities
str = '#0072BD';
r = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#D95319';
b = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#EDB120';
g = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#7E2F8E';
p = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
color = [r,b,g,p];


td_ex = td_trim; %example td
close all
for x = 1:numel(trial) %number of trials
    hold off
    for ns = 1:numel(activated(1,:)) %number of neuron inputs
        for n = 1:numel(activated(:,1)) %number of current inputs
            td_stim = td_trim; % resets previous stim
            for p = 1:numel(activated(n,ns).rad)
                td_stim(trial(x)).VAE_firing_rates(stimStart_idx:stimLength_idx,activated(n,ns).rad(p)) = hz;
            end
%             td_ex(trial(x)).vel(startMove_idx:endMove_idx,:) = td_ex(trial(x)).VAE_firing_rates(startMove_idx:endMove_idx,:) * dec +bias; %regular decoded hand vel
            td_stim(trial(x)).hand_vel_hat(startMove_idx:endMove_idx,:) = td_stim(trial(x)).VAE_firing_rates(startMove_idx:endMove_idx,:)* dec + bias; %stim decoded hand vel
            for time = startMove_idx+1:endMove_idx
                td_stim(trial(x)).pos(time,1) = td_stim(trial(x)).pos(time-1,1) + (td_stim(trial(x)).hand_vel_hat(time-1,1)*binSize);%stim effect on hand pos in x dir
                td_stim(trial(x)).pos(time,2) = td_stim(trial(x)).pos(time-1,2) + (td_stim(trial(x)).hand_vel_hat(time-1,2)*binSize);%stim effect on hand pos in y dir
                td_ex(trial(x)).pos(time,1) = td_ex(trial(x)).pos(time-1,1) + (td_ex(trial(x)).hand_vel_hat(time-1,1)*binSize);%stim effect on hand pos in x dir
                td_ex(trial(x)).pos(time,2) = td_ex(trial(x)).pos(time-1,2) + (td_ex(trial(x)).hand_vel_hat(time-1,2)*binSize);%stim effect on hand pos in y dir
            end
           xoffset = td_stim(trial(x)).pos(startMove_idx,1);
           moveDir(ns, n).stim(1:numel(startMove_idx:endMove_idx),1) = td_stim(trial(x)).pos(startMove_idx:endMove_idx,1) - xoffset; %movement x-axis stim case
           yoffset = td_stim(trial(x)).pos(startMove_idx,2);
           moveDir(ns,n).stim(1:numel(startMove_idx:endMove_idx),2) = td_stim(trial(x)).pos(startMove_idx:endMove_idx,2) - yoffset; %movement y-axis stim case
           xoffset = td_ex(trial(x)).pos(startMove_idx,1);
           moveDir(ns, n).decMove(1:numel(startMove_idx:endMove_idx),1) = td_ex(trial(x)).pos(startMove_idx:endMove_idx,1) - xoffset; %movement x-axis no stim case
           yoffset = td_ex(trial(x)).pos(startMove_idx,2);
           moveDir(ns,n).decMove(1:numel(startMove_idx:endMove_idx),2) = td_ex(trial(x)).pos(startMove_idx:endMove_idx,2) - yoffset; %movement x-axis no stim case
           plot(td_stim(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,1),td_stim(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,2), 'Color', color(((3*n)-2):((3*n)-2)+(numel(r)-1))) %plot stim of decoded movement
           set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
           hold on
           st = plot(td_stim(trial(x)).pos(stimStart_idx+1,1),td_stim(trial(x)).pos(stimStart_idx+1,2), '-k'); %plot stim of decoded movement
           set(get(get(st(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
           st = plot(td_stim(trial(x)).pos(stimStart_idx + 2,1),td_stim(trial(x)).pos(stimStart_idx + 2,2), '*m'); %plot stim of decoded movement
           set(get(get(st(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
           title('Hand Position and the Effect of ICMS neuron ' + string(neuron(ns)))
           
        end
        %plot(td_trim(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,1),td_trim(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,2), 'k') %plot actual movement
        plot(td_ex(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,1),td_ex(trial(x)).pos(stimStart_idx+1:stimStart_idx + 2,2),  'g') %plot decoded movement
        st = plot(td_ex(trial(x)).pos(stimStart_idx+1,1),td_ex(trial(x)).pos(stimStart_idx+1,2), '-k'); %plot stim of decoded movement
        st = plot(td_ex(trial(x)).pos(stimStart_idx + 2,1),td_ex(trial(x)).pos(stimStart_idx + 2,2), '*m'); %plot stim of decoded movement
        legend( string(current(n-3)) + ' µA', string(current(n-2)) + ' µA',string(current(n-1)) + ' µA',string(current(n)) + ' µA', '0 µA', 'PD', 'end', 'Location', 'northeastoutside')
        xlabel('X-hand position (cm)')
        ylabel('Y-hand position (cm)')
    end
end



%% histograms for 1 run of activation
clear PDcomp
str = '#0072BD';
r = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#D95319';
b = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#EDB120';
g = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#7E2F8E';
p = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
color = [r,b,g,p];


time = stimStart_idx + 2;

for ns = 1:numel(activated(1,:)) %number of neuron inputs
    for n = 1:numel(activated(:,1)) %number of current inputs 
        moveDir(ns,n).direction = atan2d((moveDir(ns, n).stim(time,2)) - (moveDir(ns, n).decMove(time,2)),(moveDir(ns, n).stim(time,1)) - (moveDir(ns, n).decMove(time,1)));
        moveDir(ns,n).directionMove = atan2d(((moveDir(ns, n).stim(time,2))) ,(moveDir(ns, n).stim(time,1))) - atan2d((moveDir(ns, n).decMove(time,2)), (moveDir(ns, n).decMove(time,1)));
        moveDir(ns,n).stimMove = atan2d(((moveDir(ns, n).stim(time,2))) ,(moveDir(ns, n).stim(time,1))) - pdtable(neuron(ns));
        moveDir(ns,n).noStim = atan2d(moveDir(ns, n).decMove(time,2),moveDir(ns, n).decMove(time,1)) - pdtable(neuron(ns));
        if moveDir(ns,n).stimMove < 0
            moveDir(ns,n).stimMove = moveDir(ns,n).stimMove + 360;
        end
        if moveDir(ns,n).stimMove > 180
            moveDir(ns,n).stimMove = moveDir(ns,n).stimMove - 360;
        end
        if moveDir(ns,n).direction < 0
            moveDir(ns,n).direction = moveDir(ns,n).direction + 360;
        end
%         if moveDir(ns,n).directionMove < 0
%             moveDir(ns,n).directionMove = moveDir(ns,n).directionMove + 360;
%         end
        for p = 1:numel(activated(n,ns).rad)
            PDcomp(ns,n).dirs(p) = pdtable(activated(n,ns).rad(p));  %find PDs of activated neurons
            PDcomp(ns,n).diff(p) = moveDir(ns,n).direction - PDcomp(ns,n).dirs(p); %subtract direction of stim - direction of PD
            PDcomp(ns,n).stimDiff(p) = pdtable(neuron(ns)) - pdtable(activated(n,ns).rad(p));
            if PDcomp(ns,n).diff(p) < 0
               PDcomp(ns,n).diff(p) = PDcomp(ns,n).diff(p) + 360;
            end
%             if PDcomp(ns,n).diff(p) > 180
%                PDcomp(ns,n).diff(p) = PDcomp(ns,n).diff(p) - 360;
%             end
            if PDcomp(ns,n).stimDiff(p) < 0
               PDcomp(ns,n).stimDiff(p) = PDcomp(ns,n).stimDiff(p) + 360;
            end
%             if PDcomp(ns,n).stimDiff(p) > 180
%                PDcomp(ns,n).stimDiff(p) = PDcomp(ns,n).stimDiff(p) - 360;
%             end
        end
%         figure();
%         histogram(PDcomp(ns,n).diff, 'Normalization', 'probability', 'BinWidth', 10, 'BinLimits', [-180 180], 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
%         set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',12);
%         %title( 'Difference between direction of stimulation effect on decoded movement and PDs of activated neurons')
%         xlabel('Angular Difference (stim - PD of neuron) in degrees')
%         ylabel('Fraction of activated neurons')
%         xticks([-180 0 180])
%         ylim([0 1])
    end
end
for ns = 1:numel(activated(1,:)) %number of neuron inputs
    figure();
    for n = 1:numel(activated(:,1)) %number of current inputs
        plot(current(n), abs(moveDir(ns,n).directionMove),'*', 'MarkerEdgeColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'MarkerSize', 20)
        set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',12);
        title( 'Difference between decoded movement with and without stimulation')
        xlabel('Current (µA)')
        ylabel('Angular Difference (stim movement - decoded movement) in degrees')
        xlim([0 120])
        hold on
    end
    legend( string(current(n)) + ' µA','Location', 'northwest','NumColumns', 3)
    figure();
    for n = 1:numel(activated(:,1)) %number of current inputs
        plot(current(n), moveDir(ns,n).stimMove, '*', 'MarkerEdgeColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'MarkerSize', 20); 
        set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',12);
        hold on
        title('Angular difference between effect on stimulated decoded movement and PD of stimulated neuron')
        xlabel('Current (µA)')
        ylabel('Angular Difference (PD - angle of movement) in degrees')        
    end
    plot(0, moveDir(ns,n).noStim, '*', 'MarkerEdgeColor', 'g', 'MarkerSize', 20); 
    legend( string(current(n)) + ' µA', 'No Stim','Location', 'northeast','NumColumns', 3)
    xlim([-5 120])
%     for n = 1:numel(activated(:,1)) %number of current inputs
%         for iter = 1:iterations
%             x = find(PDcomp(ns,n).stimDiff(:,iter));
%             if iter == 1
%                 PDcomp(ns,n).forHisto2 = PDcomp(ns,n).stimDiff(x);
%             else
%                 PDcomp(ns,n).forHisto2 = vertcat(PDcomp(ns,n).stimDiff(x,iter),PDcomp(ns,n).forHisto2);
%             end
%         end
%     end
    for n = 1:numel(activated(:,1)) %number of current inputs
        figure();
        polarhistogram(deg2rad(pdtable((activated(n,ns).rad))), 24, 'Normalization', 'probability', 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
        set(gcf,'Color','White'); set(gca,'FontName','Helvetica','FontSize',12);
        %title( 'Angular difference between PD of stimulated neuron and PD activated neurons')
        rlim([0 0.25])
    end
end



%% Multiple runs of different activations
iterations = 100;
clear PDcomp
clear moveDir
for iter = 1:iterations
    for x =1:numel(current)
        for n = 1:numel(neuron_location(:,1)) %total neuron number
            for ns = 1:numel(neuron) %stim neuron number
                activator = rand(1);
                if activator < P_act(ns,x).spaceP(n)
                    activate(x,ns).neuron(n) = 1;
                else
                    activate(x,ns).neuron(n) = 0;
                end
            end
        end
    end
    
    clear activated
    for x=1:numel(current)
        for ns = 1:numel(neuron)
            activated(x,ns).rad= find(activate(x,ns).neuron);
        end
    end
    
    
    for x = 1:numel(trial) %number of trials
        for ns = 1:numel(activated(1,:)) %number of neuron inputs
            for n = 1:numel(activated(:,1)) %number of current inputs
                td_stim = td_trim; % resets previous stim, comment out if doing simultaneous stimulations
                for p = 1:numel(activated(n,ns).rad)
                    td_stim(trial(x)).VAE_firing_rates(stimStart_idx:stimLength_idx,activated(n,ns).rad(p)) = hz;
                    moveDir(ns,n).activated(p,iter) = deg2rad(pdtable((activated(n,ns).rad(p))));

                end
                %             td_ex(trial(x)).vel(startMove_idx:endMove_idx,:) = td_ex(trial(x)).VAE_firing_rates(startMove_idx:endMove_idx,:) * dec +bias; %regular decoded hand vel
                td_stim(trial(x)).hand_vel_hat(startMove_idx:endMove_idx,:) = td_stim(trial(x)).VAE_firing_rates(startMove_idx:endMove_idx,:)* dec + bias; %stim decoded hand vel
                for time = startMove_idx+1:endMove_idx
                    td_stim(trial(x)).pos(time,1) = td_stim(trial(x)).pos(time-1,1) + (td_stim(trial(x)).hand_vel_hat(time-1,1)*binSize);%stim effect on hand pos in x dir
                    td_stim(trial(x)).pos(time,2) = td_stim(trial(x)).pos(time-1,2) + (td_stim(trial(x)).hand_vel_hat(time-1,2)*binSize);%stim effect on hand pos in y dir
                    td_ex(trial(x)).pos(time,1) = td_ex(trial(x)).pos(time-1,1) + (td_ex(trial(x)).hand_vel_hat(time-1,1)*binSize);%stim effect on hand pos in x dir
                    td_ex(trial(x)).pos(time,2) = td_ex(trial(x)).pos(time-1,2) + (td_ex(trial(x)).hand_vel_hat(time-1,2)*binSize);%stim effect on hand pos in y dir
                end
                xoffset = td_stim(trial(x)).pos(startMove_idx,1);
                moveDir(ns, n).stim(1:numel(startMove_idx:endMove_idx),1) = td_stim(trial(x)).pos(startMove_idx:endMove_idx,1) - xoffset; %movement x-axis stim case
                yoffset = td_stim(trial(x)).pos(startMove_idx,2);
                moveDir(ns,n).stim(1:numel(startMove_idx:endMove_idx),2) = td_stim(trial(x)).pos(startMove_idx:endMove_idx,2) - yoffset; %movement y-axis stim case
                xoffset = td_ex(trial(x)).pos(startMove_idx,1);
                moveDir(ns, n).decMove(1:numel(startMove_idx:endMove_idx),1) = td_ex(trial(x)).pos(startMove_idx:endMove_idx,1) - xoffset; %movement x-axis no stim case
                yoffset = td_ex(trial(x)).pos(startMove_idx,2);
                moveDir(ns,n).decMove(1:numel(startMove_idx:endMove_idx),2) = td_ex(trial(x)).pos(startMove_idx:endMove_idx,2) - yoffset; %movement x-axis no stim case
            end
        end
    end
    str = '#0072BD';
    r = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
    str = '#D95319';
    b = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
    str = '#EDB120';
    g = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
    str = '#7E2F8E';
    p = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
    color = [r,b,g,p];
    
    
    time = 3;
    
    for ns = 1:numel(activated(1,:)) %number of neuron inputs
        for n = 1:numel(activated(:,1)) %number of current inputs
            moveDir(ns,n).direction(iter) = atan2d((moveDir(ns, n).stim(time,2)) - (moveDir(ns, n).decMove(time,2)),(moveDir(ns, n).stim(time,1)) - (moveDir(ns, n).decMove(time,1))); %angle of stim
            moveDir(ns,n).directionMove(iter) = atan2d(((moveDir(ns, n).stim(time,2))) ,(moveDir(ns, n).stim(time,1))) - atan2d((moveDir(ns, n).decMove(time,2)), (moveDir(ns, n).decMove(time,1))); % angle of movement change
            moveDir(ns,n).stimMove(iter) = atan2d(((moveDir(ns, n).stim(time,2))) ,(moveDir(ns, n).stim(time,1))) - pdtable(neuron(ns));
            moveDir(ns,n).noStim = atan2d(moveDir(ns, n).decMove(time,2),moveDir(ns, n).decMove(time,1)) - pdtable(neuron(ns));
            if moveDir(ns,n).direction(iter) < 0
                moveDir(ns,n).direction(iter) = moveDir(ns,n).direction(iter) + 360;
            end
            if moveDir(ns,n).stimMove(iter) < 0
                moveDir(ns,n).stimMove(iter) = moveDir(ns,n).stimMove(iter) + 360;
            end
            if moveDir(ns,n).stimMove(iter) > 180
                moveDir(ns,n).stimMove(iter) = moveDir(ns,n).stimMove(iter) - 360;
            end
            for p = 1:numel(activated(n,ns).rad) %number of activated neurons
                PDcomp(ns,n).dirs(p,iter) = pdtable(activated(n,ns).rad(p));  %find PDs of activated neurons
                PDcomp(ns,n).diff(p,iter) = moveDir(ns,n).direction(iter) - PDcomp(ns,n).dirs(p,iter); %subtract direction of stim - direction of PD
                PDcomp(ns,n).stimDiff(p,iter) = pdtable(neuron(ns)) - pdtable(activated(n,ns).rad(p));
                if PDcomp(ns,n).diff(p,iter) < 0
                    PDcomp(ns,n).diff(p,iter) = PDcomp(ns,n).diff(p,iter) + 360;
                end
                if PDcomp(ns,n).diff(p,iter) > 180
                    PDcomp(ns,n).diff(p,iter) = PDcomp(ns,n).diff(p,iter) - 360;
                end
                if PDcomp(ns,n).stimDiff(p,iter) < 0
                    PDcomp(ns,n).stimDiff(p,iter) = PDcomp(ns,n).stimDiff(p,iter) + 360;
                end
                if PDcomp(ns,n).stimDiff(p,iter) > 180
                    PDcomp(ns,n).stimDiff(p,iter) = PDcomp(ns,n).stimDiff(p,iter) - 360;
                end
            end
        end
    end
end

%% Histograms from multiple runs

for ns = 1:numel(activated(1,:)) %number of neuron inputs
    figure();set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
    for n = 1:numel(activated(:,1)) %number of current inputs
        plot(current(n), mean(moveDir(ns,n).directionMove),'*', 'MarkerEdgeColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'MarkerSize', 20)
        hold on
        title( 'Difference between decoded movement with stimulation and without stimulation')
        xlabel('Currents (µA)' )
        ylabel('Angular Difference (stim movement - decoded movement) in degrees')
        xlim([0 120])
    end
    for n = 1:numel(activated(:,1))
        errorbar(current(n), mean(moveDir(ns,n).directionMove), (std(moveDir(ns,n).directionMove))/sqrt(iterations), 'Color', color(((3*n)-2):((3*n)-2)+(numel(r)-1)))
        hold on
    end
    %legend(string(current(n-3)) + ' µA', string(current(n-2)) + ' µA', string(current(n-1)) + ' µA', string(current(n)) + ' µA','Location', 'northeast','NumColumns', 3)
%     for n = 1:numel(activated(:,1)) %number of current inputs
%         for iter = 1:iterations
%             x = find(PDcomp(ns,n).diff(:,iter));
%             if iter == 1
%                 PDcomp(ns,n).forHisto = PDcomp(ns,n).diff(x,iter);
%             else
%                 PDcomp(ns,n).forHisto = vertcat(PDcomp(ns,n).diff(x,iter),PDcomp(ns,n).forHisto);
%             end
%         end
%         figure();set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
%         histogram(PDcomp(ns,n).forHisto, 'Normalization', 'probability', 'BinWidth', 10, 'BinLimits', [-180 180], 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
%         title( 'Difference between direction of stimulation effect on decoded movement and PDs of activated neurons')
%         xlabel('Angular Difference (stim - PD of neuron) in degrees')
%         ylabel('Fraction of activated neurons')
%         xticks([-180 0 180])
%         ylim([0 1])
%     end
    figure();
    for n = 1:numel(activated(:,1)) %number of current inputs
       plot(current(n), mean(moveDir(ns,n).stimMove),'*', 'MarkerEdgeColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'MarkerSize', 20)
        set(gcf,'Color','White');set(gca,'FontName','Helvetica','FontSize',18);
        hold on
        %histogram(moveDir(ns,n).stimMove, 'Normalization', 'probability', 'BinWidth', 10, 'BinLimits', [-180 180], 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
        title( 'Angular difference between effect on stimulated decoded movement and PD stimulated neuron')
        xlabel('Current (µA)')
        ylabel('Angular Difference (PD - angle of movement) in degrees')
        xlim([-5 120])      
    end
    for n = 1:numel(activated(:,1))
        errorbar(current(n), mean(moveDir(ns,n).stimMove), (std(moveDir(ns,n).stimMove))/sqrt(iterations), 'Color', color(((3*n)-2):((3*n)-2)+(numel(r)-1)))
        hold on
        plot(0, moveDir(ns,n).noStim, '*', 'MarkerEdgeColor', 'g', 'MarkerSize', 20); 
    end
    for n = 1:numel(activated(:,1)) %number of current inputs
        for iter = 1:iterations
            x = find(PDcomp(ns,n).stimDiff(:,iter));
            if iter == 1
                PDcomp(ns,n).forHisto2 = PDcomp(ns,n).stimDiff(x,iter);
            else
                PDcomp(ns,n).forHisto2 = vertcat(PDcomp(ns,n).stimDiff(x,iter),PDcomp(ns,n).forHisto2);
            end
        end
    end
%     for n = 1:numel(activated(:,1)) %number of current inputs
%         figure();set(gcf,'Color','White'); set(gca,'FontName','Helvetica','FontSize',18);
%         histogram(PDcomp(ns,n).forHisto2, 'Normalization', 'probability', 'BinWidth', 10, 'BinLimits', [-180 180], 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
%         title( 'Angular difference between PD of stimulated neuron and PD activated neurons')
%         xlabel('Angular Difference (PD stimulated - PD activated) in degrees')
%         ylabel('Fraction of activated neurons')
%         xticks([-180 0 180])
%         ylim([0 1])
%     end
    for n = 1:numel(activated(:,1)) %number of current inputs
        for iter = 1:iterations
            x = find(moveDir(ns,n).activated(:,iter));
            if iter == 1
                PDcomp(ns,n).newThing = moveDir(ns,n).activated(x,iter);
            else
                PDcomp(ns,n).newThing = vertcat(moveDir(ns,n).activated(x,iter),PDcomp(ns,n).newThing);
            end
        end
    end
    for n = 1:numel(activated(:,1)) %number of current inputs
        figure();
        polarhistogram(PDcomp(ns,n).newThing, 24, 'Normalization', 'probability', 'FaceColor', color(((3*n)-2):((3*n)-2)+(numel(r)-1)), 'LineWidth', 2);
        set(gcf,'Color','White'); set(gca,'FontName','Helvetica','FontSize',12);
        %title( 'Angular difference between PD of stimulated neuron and PD activated neurons')
        rlim([0 0.35])
    end
end












