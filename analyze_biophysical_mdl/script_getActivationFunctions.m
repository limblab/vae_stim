input_data.folderpath = 'C:\Users\Joseph Sombeck\Box\Miller-Grill_S1-stim\ModelData\SpatialResponse';

input_data.amp_list = [5:5:50,100];
input_data.n_soma = 25;

output_data = getBioModelData(input_data); % this takes a few minutes


%% get activation function as a distance from stim electrode for each amplitude

elec_loc = [1500,1000,1500]; 
max_dist = 2275; % um
dist_bins = 75:100:max_dist;
total_count = zeros(numel(input_data.amp_list),numel(dist_bins)-1);
act_count = zeros(size(total_count));

for i_amp = 1:numel(input_data.amp_list)
    temp_dist = getDistance(elec_loc, output_data{i_amp}.soma_coord);
    is_act = output_data{i_amp}.soma_act;
    total_count(i_amp,:) = histcounts(temp_dist,dist_bins);
    act_count(i_amp,:) = histcounts(temp_dist(is_act==1),dist_bins);
end

act_prop = act_count./total_count;

plot(act_prop')

