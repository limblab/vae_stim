%% get folder path for maps and trial data
    pathname = 'D:\Joseph\stimModel';
    td_filename = 'Han_20160315_RW_smoothKin_jointAngjointVel_50ms_td.mat';
    map_foldername = 'Han_20160315_big_maps';
    fr_files = dir([pathname filesep map_foldername filesep '*.csv']);

    load([pathname filesep 'td' filesep td_filename]);
    td_load = td;
    
    
%% for each file, load map. Do analyses
    output_data = cell(numel(fr_files),1);
    for i_file = 1:numel(fr_files)
        disp(i_file);
        td = td_load;
        
        % load firing rates
        firing_rates = readtable([fr_files(i_file).folder filesep fr_files(i_file).name]);
        firing_rates = firing_rates{:,:};
        
        %add to td
        bin_size = td(1).bin_size;
        td.VAE_firing_rates = firing_rates(:,:)/bin_size;
        map_dim = sqrt(numel(firing_rates(1,:)) + [0,0]);
        
        % get location of each neuron
        locs = zeros(map_dim(1)*map_dim(2),2);
        [locs(:,1), locs(:,2)] = ind2sub([map_dim(1), map_dim(2)],1:map_dim(1)*map_dim(2)); 

        % match up data lengths
        field_len = length(td.vel);
        td_fieldnames = fieldnames(td);
        [~,mask] = rmmissing(td.vel);

        for i_field = 1:numel(td_fieldnames)
            if(length(td.(td_fieldnames{i_field})) == field_len)
                td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
            end
        end    

        % analyze data....
        input_data = [];
        input_data.block_size = 0.06; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
        input_data.td = td;
        input_data.locs = locs;
        input_data.map_dim = map_dim;
        input_data.dec_path = 'D:\Joseph\stimModel\decoders';
        input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
        input_data.sigma = 2;
        output_data{i_file} = getStimModelSimulationData(input_data);
        output_data{i_file}.sigma = input_data.sigma;
    end
    


%% check decoder for each map
% use decoder to get predicted hand velocities
    figure();
    ax_list = [];
    for i = 1:numel(output_data)
        td = output_data{i}.td;
        ax_list(end+1)=subplot(2,numel(output_data),i); hold on;
        plot(td.vel(:,1), td.pred_vel(:,1),'.');
        plot([-30,30],[-30,30],'k--','linewidth',2);
        xlabel('Hand vel (cm/s)');
        ylabel('Pred hand vel (cm/s)');
        formatForLee(gcf); set(gca,'fontsize',8);

        ax_list(end+1)=subplot(2,numel(output_data),i+numel(output_data)); hold on;
        plot(td.vel(:,2), td.pred_vel(:,2),'.');
        plot([-30,30],[-30,30],'k--','linewidth',2);
        formatForLee(gcf); set(gca,'fontsize',8);
        xlabel('Hand vel (cm/s)');
        ylabel('Pred hand vel (cm/s)');
        
    end
    linkaxes(ax_list,'xy');

%% compare decoder PD to PDs found using movement and an encoder
    figure();
    for i = 1:numel(output_data)
        subplot(1,numel(output_data),i)
        pd_table = output_data{i}.pd_table;
        dec = output_data{i}.dec;
        
        PD_dec_diff = rad2deg(angleDiff(pd_table.velPD, atan2(dec(:,2),dec(:,1)),1));
        bin_edges = [0:10:180];

        histogram(abs(PD_dec_diff),bin_edges,'Normalization','probability')
        xlabel('|PD diff| (degrees)');
        ylabel('Proportion of neurons');
        formatForLee(gcf);
        set(gca,'fontsize',14);
        
    end



%% analyze data from experiment 1    
%% look at how activated population compares to stimulation location PD
    % compare across amplitudes and activation functions. 
    
    for i = 1:numel(output_data)
        f=plotActivatedPopulationVsStimPD(output_data{i}.amp_input_data,output_data{i}.amp_output_data,output_data{i}.pd_table)
        f.Name = ['Han_20160315_activatedPop_block060_model_based_map',num2str(i)];
    end
    
   
%% compare similarity between activated population against some neighborhood
    % metric. The goal here is to simply show that being in the middle of a
    % cluster is better than being near the edge. Also, some variance among
    % the similarity of the activated population would be nice.
    
    % for each sim, compute a similarity between activated population and
    % stim location. Also compute a similarity between neighbors and stim
    % location
    
    for i = 1:numel(output_data)
        f=plotNeighborSimilarityVsActivatedSimilarity(output_data{i}.amp_input_data, output_data{i}.amp_output_data,output_data{i}.pd_table.velPD);
        f.Name = ['Han_20160315_neighBorSim_block060_model_based_map',num2str(i)];
%         saveFiguresLIB(f,fpath,f.Name);
    end
    
%% look at how effect of stimulation compares to PDs of stimulation location
    

    for i = 1:numel(output_data)
        f=plotStimEffectVsStimPD(output_data{i}.amp_input_data, output_data{i}.amp_output_data, output_data{i}.pd_table);
        f.Name = ['Han_20160315_pdEffect_block060_model_based_map',num2str(i)];
%         saveFiguresLIB(f,fpath,f.Name);
    end
    

    
%% get psychometric curve shift due to stimulation and plot for each condition and map
    PSE = zeros(numel(output_data), numel(output_data{1}.class_output_data), numel(output_data{1}.class_output_data(1).stim_amp));
    color_list = inferno(size(PSE,3)+1);
    for i = 1:numel(output_data)

        PSE(i,:,:) = computePsychometricCurveShift(output_data{i}.class_output_data);
        
%         figure(); hold on;
%         for i_amp = 1:size(PSE,3)
%             subplot(1,size(PSE,3),i_amp)
%             histogram(PSE(i,:,i_amp),-90:10:90);
%         end
    end

    
% get distribution of predictable experiments for each map
    color_list = inferno(numel(output_data{1}.class_output_data(1).stim_amp)+1);
    PSE = zeros(numel(output_data), numel(output_data{1}.class_output_data), numel(output_data{1}.class_output_data(1).stim_amp));
    prob_predictable = zeros(size(PSE,1),size(PSE,3));
    for i_exp = 1:numel(output_data)
        PSE(i_exp,:,:) = computePsychometricCurveShift(output_data{i_exp}.class_output_data);
    end
    
    figure(); hold on;
    for i_amp = 1:size(PSE,3)
        prob_predictable(:,i_amp) = sum(PSE(:,:,i_amp) > 0,2)/(size(PSE,2)); % 20 degree threshold for now
        
        plot(output_data{1}.class_output_data(1).stim_amp(i_amp) + rand(size(prob_predictable,1),1)*2-1, prob_predictable(:,i_amp),...
            'o','markersize',6,'color',color_list(i_amp,:),'linewidth',1);
        plot(output_data{1}.class_output_data(1).stim_amp(i_amp),median(prob_predictable(:,i_amp)),...
            '_','markersize',20,'color',color_list(i_amp,:),'linewidth',3);
    end
   
    xlabel('Amplitude (\muA)');
    ylabel('Proportion biased toward PD');
    formatForLee(gcf);
    set(gca,'fontsize',14);
    
    plot([10,60],[0.5,0.5],'k--','linewidth',1.25)
    ylim([0.4,1])
    
    ax=gca;
    ax.XTick = [10:10:60];
    ax.XAxis.MinorTickValues = [10:5:60];
%% get distance of PD distribution from uniform dist
    bin_edges = linspace(-pi,pi,20);
    uniform_MSE = zeros(numel(output_data),1);
    
    
    ideal_prop = 1/numel(bin_edges-1);
    for i_exp = 1:numel(output_data)
        pd_list = output_data{i_exp}.pd_table.velPD;
        
        pd_prop = histcounts(pd_list,bin_edges,'Normalization','Probability');
%         figure();
%         rose(pd_list,bin_edges);
       	uniform_MSE(i_exp) = mean((pd_prop-ideal_prop).^2);
        
    end
    
    
%% get neighbor metric for each experiment in each file. Also get PSD for each experiment in each file
    all_neigh_sim = [];
    all_PSE = [];
    for i=1:numel(output_data)
        sim_input_data = [];
        sim_input_data.locs = output_data{i}.class_input_data.locs;
        sim_input_data.metric_name = 'PD';
        sim_input_data.PD = output_data{i}.pd_table.velPD; % PD must be in radians!
        sim_input_data.is_ang = 1;

        sim_input_data.nbor_max_r = 0.08; % in mm
        sim_input_data.nbor_min_r = 0;

        file_neigh = getClassNeighborsSimilarity(sim_input_data,output_data{i}.class_output_data);
        
        all_neigh_sim = [all_neigh_sim;file_neigh];
        
        temp_PSE = computePsychometricCurveShift(output_data{i}.class_output_data);
        all_PSE = [all_PSE; temp_PSE];
    end

    plot(all_neigh_sim,all_PSE(:,1),'.','markersize',10)
    
    
    
    