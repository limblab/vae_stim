%% get folder path for maps and trial data
    pathname = 'D:\Joseph\stimModel';
    td_filename = 'Han_20160325_RW_smoothKin_jointAngjointVel_50ms_td.mat';
    map_foldername = 'Han_20160325_maps';
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
    
    cd(pathname);
    save('Han_20160325_many_amps_sim_data_sigma2_block060_modelAct_20210504.mat','-v7.3');
    clear;
    
 %% get folder path for maps and trial data
    pathname = 'D:\Joseph\stimModel';
    td_filename = 'Han_20160325_RW_smoothKin_jointAngjointVel_50ms_td.mat';
    map_foldername = 'Han_20160325_maps';
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
        input_data.block_size = 0.075; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
        input_data.td = td;
        input_data.locs = locs;
        input_data.map_dim = map_dim;
        input_data.dec_path = 'D:\Joseph\stimModel\decoders';
        input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
        input_data.sigma = 2;
        output_data{i_file} = getStimModelSimulationData(input_data);
        output_data{i_file}.sigma = input_data.sigma;
    end
    cd(pathname);
    save('Han_20160325_many_amps_sim_data_sigma2_block075_modelAct_20210504.mat','-v7.3');
    clear;   
    
    
 %% get folder path for maps and trial data
    pathname = 'D:\Joseph\stimModel';
    td_filename = 'Chips_20151211_RW_smoothKin_jointAngjointVel_50ms_td.mat';
    map_foldername = 'Chips_20151211_maps';
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
        input_data.block_size = 0.075; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
        input_data.td = td;
        input_data.locs = locs;
        input_data.map_dim = map_dim;
        input_data.dec_path = 'D:\Joseph\stimModel\decoders';
        input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
        input_data.sigma = 2;
        output_data{i_file} = getStimModelSimulationData(input_data);
        output_data{i_file}.sigma = input_data.sigma;
    end
    cd(pathname);
    save('Chips_20151211_many_amps_sim_data_sigma2_block075_modelAct_20210504.mat','-v7.3');
    clear;  
    
    
 %% get folder path for maps and trial data
    pathname = 'D:\Joseph\stimModel';
    td_filename = 'Chips_20151211_RW_smoothKin_jointAngjointVel_50ms_td.mat';
    map_foldername = 'Chips_20151211_maps';
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
        input_data.block_size = 0.060; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
        input_data.td = td;
        input_data.locs = locs;
        input_data.map_dim = map_dim;
        input_data.dec_path = 'D:\Joseph\stimModel\decoders';
        input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
        input_data.sigma = 2;
        output_data{i_file} = getStimModelSimulationData(input_data);
        output_data{i_file}.sigma = input_data.sigma;
    end
    cd(pathname);
    save('Chips_20151211_many_amps_sim_data_sigma2_block060_modelAct_20210504.mat','-v7.3');
    clear;  
% %% get folder path for maps and trial data
%     pathname = 'D:\Joseph\stimModel';
%     td_filename = 'Han_20160315_RW_SmoothKin_50ms.mat';
%     map_foldername = 'Han_20160315_maps';
%     fr_files = dir([pathname filesep map_foldername filesep '*.csv']);
% 
%     load([pathname filesep 'td' filesep td_filename]);
%     td_load = td;
%     
%     
% %% for each file, load map. Do analyses
%     output_data = cell(numel(fr_files),1);
%     for i_file = 1:numel(fr_files)
%         disp(i_file);
%         td = td_load;
%         
%         % load firing rates
%         firing_rates = readtable([fr_files(i_file).folder filesep fr_files(i_file).name]);
%         firing_rates = firing_rates{:,:};
%         
%         %add to td
%         bin_size = td(1).bin_size;
%         td.VAE_firing_rates = firing_rates(:,:)/bin_size;
%         map_dim = sqrt(numel(firing_rates(1,:)) + [0,0]);
%         
%         % get location of each neuron
%         locs = zeros(map_dim(1)*map_dim(2),2);
%         [locs(:,1), locs(:,2)] = ind2sub([map_dim(1), map_dim(2)],1:map_dim(1)*map_dim(2)); 
% 
%         % match up data lengths
%         field_len = length(td.vel);
%         td_fieldnames = fieldnames(td);
%         [~,mask] = rmmissing(td.vel);
% 
%         for i_field = 1:numel(td_fieldnames)
%             if(length(td.(td_fieldnames{i_field})) == field_len)
%                 td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
%             end
%         end    
% 
%         % analyze data....
%         input_data = [];
%         input_data.block_size = 0.067; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
%         input_data.td = td;
%         input_data.locs = locs;
%         input_data.map_dim = map_dim;
%         input_data.dec_path = 'D:\Joseph\stimModel\decoders';
%         input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
%         input_data.sigma = 2;
%         output_data{i_file} = getStimModelSimulationData(input_data);
%         output_data{i_file}.sigma = input_data.sigma;
%     end
%     
%     %%
%     cd(pathname);
%     save('Han_20160315_many_amps_sim_data_sigma2_block067_modelAct_20210503.mat','-v7.3');
%     clear
%     
%     %% get folder path for maps and trial data
%     pathname = 'D:\Joseph\stimModel';
%     td_filename = 'Han_20160315_RW_SmoothKin_50ms.mat';
%     map_foldername = 'Han_20160315_maps';
%     fr_files = dir([pathname filesep map_foldername filesep '*.csv']);
% 
%     load([pathname filesep 'td' filesep td_filename]);
%     td_load = td;
%     
%     
% %% for each file, load map. Do analyses
%     output_data = cell(numel(fr_files),1);
%     for i_file = 1:numel(fr_files)
%         disp(i_file);
%         td = td_load;
%         
%         % load firing rates
%         firing_rates = readtable([fr_files(i_file).folder filesep fr_files(i_file).name]);
%         firing_rates = firing_rates{:,:};
%         
%         %add to td
%         bin_size = td(1).bin_size;
%         td.VAE_firing_rates = firing_rates(:,:)/bin_size;
%         map_dim = sqrt(numel(firing_rates(1,:)) + [0,0]);
%         
%         % get location of each neuron
%         locs = zeros(map_dim(1)*map_dim(2),2);
%         [locs(:,1), locs(:,2)] = ind2sub([map_dim(1), map_dim(2)],1:map_dim(1)*map_dim(2)); 
% 
%         % match up data lengths
%         field_len = length(td.vel);
%         td_fieldnames = fieldnames(td);
%         [~,mask] = rmmissing(td.vel);
% 
%         for i_field = 1:numel(td_fieldnames)
%             if(length(td.(td_fieldnames{i_field})) == field_len)
%                 td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
%             end
%         end    
% 
%         % analyze data....
%         input_data = [];
%         input_data.block_size = 0.075; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
%         input_data.td = td;
%         input_data.locs = locs;
%         input_data.map_dim = map_dim;
%         input_data.dec_path = 'D:\Joseph\stimModel\decoders';
%         input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
%         input_data.sigma = 2;
%         output_data{i_file} = getStimModelSimulationData(input_data);
%         output_data{i_file}.sigma = input_data.sigma;
%     end
%     
%     cd(pathname)
%     save('Han_20160315_many_amps_sim_data_sigma2_block075_modelAct_20210503.mat','-v7.3');
%     clear
%     
%     %% get folder path for maps and trial data
%     pathname = 'D:\Joseph\stimModel';
%     td_filename = 'Han_20160315_RW_SmoothKin_50ms.mat';
%     map_foldername = 'Han_20160315_maps';
%     fr_files = dir([pathname filesep map_foldername filesep '*.csv']);
% 
%     load([pathname filesep 'td' filesep td_filename]);
%     td_load = td;
%     
%     
% %% for each file, load map. Do analyses
%     output_data = cell(numel(fr_files),1);
%     for i_file = 1:numel(fr_files)
%         disp(i_file);
%         td = td_load;
%         
%         % load firing rates
%         firing_rates = readtable([fr_files(i_file).folder filesep fr_files(i_file).name]);
%         firing_rates = firing_rates{:,:};
%         
%         %add to td
%         bin_size = td(1).bin_size;
%         td.VAE_firing_rates = firing_rates(:,:)/bin_size;
%         map_dim = sqrt(numel(firing_rates(1,:)) + [0,0]);
%         
%         % get location of each neuron
%         locs = zeros(map_dim(1)*map_dim(2),2);
%         [locs(:,1), locs(:,2)] = ind2sub([map_dim(1), map_dim(2)],1:map_dim(1)*map_dim(2)); 
% 
%         % match up data lengths
%         field_len = length(td.vel);
%         td_fieldnames = fieldnames(td);
%         [~,mask] = rmmissing(td.vel);
% 
%         for i_field = 1:numel(td_fieldnames)
%             if(length(td.(td_fieldnames{i_field})) == field_len)
%                 td.(td_fieldnames{i_field}) = td.(td_fieldnames{i_field})(mask==0,:);
%             end
%         end    
% 
%         % analyze data....
%         input_data = [];
%         input_data.block_size = 0.053; % mm, 0.06 for sigma2, 0.025 for sigma3, 0.0375 for sigma2.5
%         input_data.td = td;
%         input_data.locs = locs;
%         input_data.map_dim = map_dim;
%         input_data.dec_path = 'D:\Joseph\stimModel\decoders';
%         input_data.dec_fname = [fr_files(i_file).name(1:end-4),'_dec.mat'];
%         input_data.sigma = 2;
%         output_data{i_file} = getStimModelSimulationData(input_data);
%         output_data{i_file}.sigma = input_data.sigma;
%     end
%     
%     cd(pathname)
%     save('Han_20160315_many_amps_sim_data_sigma2_block053_modelAct_20210503.mat','-v7.3');
%     clear;