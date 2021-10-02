%% simulate stoney activation function using different ranges of k

max_r = 2; % mm
n_neurons = 10000;


k_dist = 'normal'; % 'uniform' uses k_range, 'normal' uses k_mean and k_std
k_range = [300, 3000];

k_meas = [848,507,1990,1033,1127,844,924,3460,2320,272,1032,1150];
k_mean = mean(k_meas);
k_std = std(k_meas)/sqrt(numel(k_meas));

%% assign random distances and k_vals (thresholds) to neurons
r_vals = rand(n_neurons,1)*max_r; % in mm

switch k_dist
    case 'uniform'
        k_vals = rand(n_neurons,1)*(diff(k_range)) + k_range(1);
    case 'normal'
        k_vals = normrnd(k_mean, k_std, n_neurons,1);
        k_vals(k_vals < min(k_range)) = min(k_range);
end
    
thresholds = k_vals.*(r_vals.^2); %in uA; I = k*r^2 from Stoney

%% "stimulate" : if current is above threshold, call activated. Get histogram vals
amp_vals = [5,10,15,20,25,30,35,40,45,50,100];
bin_edges = [0:0.05:max_r]; % mm
bin_centers = bin_edges(1:end-1)+mode(diff(bin_edges));

hist_vals = zeros(numel(amp_vals),numel(bin_edges)-1);
colors_use = inferno(11);

n_neurons_bin = histcounts(r_vals,bin_edges);
figure('Position',[680 558 400 420]); hold on;
for i_amp = [1,2,3,4,6,8,10,11]%1:numel(amp_vals)
    is_act = thresholds < amp_vals(i_amp);
    hist_vals(i_amp,:) = histcounts(r_vals(is_act==1),bin_edges)./n_neurons_bin;
    plot(bin_centers,hist_vals(i_amp,:),'color',colors_use(i_amp,:),'linewidth',2)

end


formatForLee(gcf); set(gca,'fontsize',14);
xlabel('Distance (mm)');
ylabel('Probability of activation');

ylim([0,1]); 
xlim([0,1.5]);

l=legend('5 \muA','10 \muA', '15 \muA','20 \muA','30 \muA','40 \muA','50 \muA','100 \muA')
set(l,'box','off')









