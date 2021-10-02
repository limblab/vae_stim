input_data.folderpath = 'C:\Users\Joseph Sombeck\Box\Miller-Grill_S1-stim\ModelData\SpatialResponse';

input_data.amp_list = [5:5:50,100];
input_data.n_soma = 25;

output_data = getBioModelData(input_data); % this takes a few minutes

%% gaussian instead of uniform filter

elec_loc = [1500,1000,1500]; 
max_dist = 2275; % um
dist_bins = 75:10:max_dist;
bin_width = 100;
total_count = zeros(numel(input_data.amp_list),numel(dist_bins)-1);
act_count = zeros(size(total_count));

gauss_std = 75;

colors_use = inferno(11);

figure('Position',[680 558 400 420]); hold on;
color_idx=1;
for i_amp = [1,2,3,4,6,8,10,11] %1:2:numel(input_data.amp_list)
    temp_dist = getDistance(elec_loc, output_data{i_amp}.soma_coord);
    is_act = output_data{i_amp}.soma_act;
    for i_bin = 1:numel(dist_bins)
        is_in_bin = (temp_dist > (dist_bins(i_bin)-bin_width/2)) & (temp_dist <= (dist_bins(i_bin)+bin_width/2));
        gauss_vals = normpdf(temp_dist,dist_bins(i_bin),gauss_std);
        
        total_count(i_amp,i_bin) = sum(gauss_vals);
        act_count(i_amp,i_bin) = sum(gauss_vals(is_act==1));
    end
    act_prop(i_amp,:) = act_count(i_amp,:)./total_count(i_amp,:);

    plot(dist_bins/1000,act_prop(i_amp,:)','linewidth',2,'color',colors_use(color_idx+1,:))
    color_idx=color_idx+1;
end

act_prop = act_count./total_count;

formatForLee(gcf); set(gca,'fontsize',14);
xlabel('Distance (mm)');
ylabel('Probability of activation');

ylim([0,1]); 
xlim([0,1.5]);

l=legend('5 \muA','10 \muA', '15 \muA','20 \muA','30 \muA','40 \muA','50 \muA','100 \muA')
set(l,'box','off')

act_prop = [zeros(1,size(act_prop,2));act_prop];
amplitudes = [0,input_data.amp_list];
distances = dist_bins;



%%

elec_loc = [1500,1000,1500]; 
max_dist = 2275; % um
dist_bins = 75:25:max_dist;
bin_width = 100;
total_count = zeros(numel(input_data.amp_list),numel(dist_bins)-1);
act_count = zeros(size(total_count));

for i_amp = 1:numel(input_data.amp_list)
    temp_dist = getDistance(elec_loc, output_data{i_amp}.soma_coord);
    is_act = output_data{i_amp}.soma_act;
    for i_bin = 1:numel(dist_bins)
        is_in_bin = (temp_dist > (dist_bins(i_bin)-bin_width/2)) & (temp_dist <= (dist_bins(i_bin)+bin_width/2));
        total_count(i_amp,i_bin) = sum(is_in_bin);
        act_count(i_amp,i_bin) = sum(is_in_bin & is_act);
    end
    
    
end

act_prop = act_count./total_count;

figure();
plot(dist_bins/1000,act_prop(:,:)','linewidth',2)
formatForLee(gcf); set(gca,'fontsize',14);
xlabel('Distance (mm)');
ylabel('Probability of activation');

ylim([0,1]); 
xlim([0,2]);

act_prop = [zeros(1,size(act_prop,2));act_prop];
amplitudes = [0,input_data.amp_list];
distances = dist_bins;
%% old approach
%% get activation function as a distance from stim electrode for each amplitude

elec_loc = [1500,1000,1500]; 
max_dist = 2275; % um
dist_bins = 75:50:max_dist;
total_count = zeros(numel(input_data.amp_list),numel(dist_bins)-1);
act_count = zeros(size(total_count));

for i_amp = 1:numel(input_data.amp_list)
    temp_dist = getDistance(elec_loc, output_data{i_amp}.soma_coord);
    is_act = output_data{i_amp}.soma_act;
    
    total_count(i_amp,:) = histcounts(temp_dist,dist_bins);
    act_count(i_amp,:) = histcounts(temp_dist(is_act==1),dist_bins);
end

act_prop = act_count./total_count;

figure();
plot((dist_bins(1:end-1)+mean(diff(dist_bins))/2)/1000,act_prop(1:end-1,:)','linewidth',2)
formatForLee(gcf); set(gca,'fontsize',14);
xlabel('Distance (mm)');
ylabel('Probability of activation');

ylim([0,1]); 
xlim([0,2]);