PSE = zeros(numel(output_data), numel(output_data{1}.class_output_data), numel(output_data{1}.class_output_data(1).stim_amp));
prob_predictable = zeros(size(PSE,1),size(PSE,3));
for i_exp = 1:numel(output_data)
    PSE(i_exp,:,:) = computePsychometricCurveShift(output_data{i_exp}.class_output_data);
end

for i_amp = 1:size(PSE,3)
    prob_predictable(:,i_amp) = sum(PSE(:,:,i_amp) > 10,2)/(size(PSE,2)); % 20 degree threshold for now
end


%
num_exps = 10;

prob_pred = zeros(num_exps+1,1);%size(prob_predictable,2));

for i = 0:num_exps
%     prob_pred_temp = binopdf(i,num_exps,prob_predictable);
%     prob_pred(i+1,:) = mean(prob_pred_temp,1);
    
    prob_pred(i+1,1) = binopdf(i,num_exps,0.7);
end
f=figure();
color_list = inferno(size(prob_predictable,2)+1);
colororder(color_list);
plot(0:1:num_exps,prob_pred(:,1:end),'linewidth',2)
xlabel('Number predictable experiments');
ylabel('Probability');
formatForLee(gcf);
set(gca,'fontsize',14)
xlim([0,num_exps])