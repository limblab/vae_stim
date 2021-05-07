pathname = 'D:\Joseph\stimModel';
search_word = 'block060_modelAct_';

degree_threshold = 0;

output_data_files = dir([pathname filesep '*' search_word '*']);


prob_predictable = [];
amp = [];
file_idx = [];
%%
for i = 1:numel(output_data_files)
    load([output_data_files(i).folder filesep output_data_files(i).name]);
    
    % get prob predictable for this file
    PSE = zeros(numel(output_data), numel(output_data{1}.class_output_data), numel(output_data{1}.class_output_data(1).stim_amp));
    prob_predictable_file = zeros(size(PSE,1),size(PSE,3));
    for i_exp = 1:numel(output_data)
        PSE(i_exp,:,:) = computePsychometricCurveShift(output_data{i_exp}.class_output_data);
    end
    
    for i_amp = 1:size(PSE,3)
        prob_predictable_file(:,i_amp) = sum(PSE(:,:,i_amp) > degree_threshold,2)/(size(PSE,2)); % absolute threshold for now
        
        % store prob_predictable for all files. Also store file_idx
        % so that the data can be separated again
    
        prob_predictable  = [prob_predictable; prob_predictable_file(:,i_amp)];
        amp = [amp; output_data{1}.class_output_data(1).stim_amp(i_amp)*ones(size(prob_predictable_file,1),1)];
        file_idx = [file_idx; i*ones(size(prob_predictable_file,1),1)];
    end
    
    clear output_data;
end


%%
    figure(); hold on;
    unique_files = unique(file_idx);
    unique_amps = unique(amp);

    for i_amp = 1:numel(unique_amps)
        mask = amp == unique_amps(i_amp);
        plot(unique_amps(i_amp), median(prob_predictable(mask)),'marker','_','markersize',28,'color','k','linewidth',3);
        for i = 1:numel(unique_files)
            mask = file_idx == unique_files(i) & amp == unique_amps(i_amp);
            
            plot(unique_amps(i_amp) + rand(sum(mask),1)*2-1, prob_predictable(mask), ...
                'color', getColorFromList(1,i-1), 'marker','o','markersize',6,'linestyle','none','linewidth',1.5);
        end
        
        
    end
    
    ax = gca;
    xlim([0,max(unique_amps)+5]);
    
    plot(ax.XLim, [0.5,0.5],'k--','linewidth',1)

    formatForLee(gcf);
    set(gca,'fontsize',14);
    xlabel('Amplitude (\muA)');
    ylabel('Proportion biased towards PD');
    
    
    
