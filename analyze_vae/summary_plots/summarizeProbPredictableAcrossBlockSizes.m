pathname = 'D:\Joseph\stimModel';
search_word = 'block';

degree_threshold = 0;

output_data_files = dir([pathname filesep '*' search_word '*']);

prob_predictable = [];
amp = [];
file_idx = [];
block_list = [];

%%
PSE = {};
for i = 1:numel(output_data_files)
    load([output_data_files(i).folder filesep output_data_files(i).name]);
    
    % get prob predictable for this file
    PSE{i} = zeros(numel(output_data), numel(output_data{1}.class_output_data), numel(output_data{1}.class_output_data(1).stim_amp));
    prob_predictable_file = zeros(size(PSE{i},1),size(PSE{i},3));
    for i_exp = 1:numel(output_data)
        PSE{i}(i_exp,:,:) = computePsychometricCurveShift(output_data{i_exp}.class_output_data);
    end
    
        block_idx = strfind(output_data_files(i).name,'block');
        underscore_idx = strfind(output_data_files(i).name,'_');
        underscore_idx = underscore_idx(find(underscore_idx > block_idx,1,'first'));
        
        block_val = str2num(output_data_files(i).name(block_idx+5:underscore_idx-1));
    
    for i_amp = 1:size(PSE{i},3)
        prob_predictable_file(:,i_amp) = sum(PSE{i}(:,:,i_amp) > degree_threshold,2)/(size(PSE{i},2)); % absolute threshold for now
        
        % store prob_predictable for all files. Also store file_idx
        % so that the data can be separated again
    
        prob_predictable  = [prob_predictable; prob_predictable_file(:,i_amp)];
        amp = [amp; output_data{1}.class_output_data(1).stim_amp(i_amp)*ones(size(prob_predictable_file,1),1)];
        file_idx = [file_idx; i*ones(size(prob_predictable_file,1),1)];
        block_list = [block_list; block_val*ones(size(prob_predictable_file,1),1)];
    end
    
    clear output_data
end


%%
    figure(); hold on;
    unique_blocks = unique(block_list);
    unique_amps = unique(amp);

    color_list = inferno(numel(unique_blocks)+1);
    for i_amp = 1:numel(unique_amps)
        for i = 1:numel(unique_blocks)
            mask = block_list == unique_blocks(i) & amp == unique_amps(i_amp);
            
            plot(unique_amps(i_amp) + 1*i - 2 + rand(sum(mask),1)*0.3-0.15, prob_predictable(mask), ...
                'color', color_list(i,:), 'marker','o','markersize',6,'linestyle','none'); 
            plot(unique_amps(i_amp) + i - 2, mean(prob_predictable(mask)), ...
                'color',color_list(i,:), 'marker','_','markersize',10,'linewidth',3);
        end
        
        
    end
    
    ax = gca;
    xlim([min(unique_amps)-5,max(unique_amps)+5]);
    
    plot(ax.XLim, [0.5,0.5],'k--','linewidth',1)

    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlabel('Amplitude (\muA)');
    ylabel('Proportion biased towards PD');
    
    ylim([0.4,1.0])
    
