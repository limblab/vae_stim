function [f] = plotHandVelElbowVelVsStimPD(amp_input_data, amp_output_data, pd_table)

% makes a 2x2 figure for each activation. Top row = hand, bottom row = elbow. First col =
% scatter plot 2nd col = histogram of absolute differences
% amplitude is put on same panel denoted by color. Makes a figure for each
% activation combination
    color_list = inferno(numel(amp_input_data.amps_test)+1);
    bin_edges = 0:10:180;

    stim_idx = getStimLocationIdx(amp_output_data.stim_loc,amp_output_data.locs);
    stim_PD = pd_table.velPD(stim_idx);
    
    hand_vel = squeeze(mean(amp_output_data.hand_vel_stim - amp_output_data.hand_vel_no,2));
    elbow_vel = squeeze(mean(amp_output_data.elbow_vel_stim - amp_output_data.elbow_vel_no,2));
    hand_ang = atan2(hand_vel(:,2),hand_vel(:,1));
    elbow_ang = atan2(elbow_vel(:,2), elbow_vel(:,1));

    % adjust ang to prevent wrap arounds
    hand_ang_plot = angleDiff(stim_PD, hand_ang,1,1) + stim_PD; % use radians, preserve sign
    elbow_ang_plot = angleDiff(stim_PD, elbow_ang,1,1) + stim_PD;
    
    ax = [];
    max_edge = pi;
    for i_dir_act = 1:numel(amp_input_data.dir_act_fact)
        for i_trans_act = 1:numel(amp_input_data.trans_acts_test)
            f=figure('Position',[681 559 1144 420]);
            for i_row = 1:2
                switch i_row
                    case 1
                        vel = hand_vel;
                        ang = hand_ang_plot;
                    case 2
                        vel = elbow_vel;
                        ang = elbow_ang_plot;
                end
                
                
                for i_amp = 1:numel(amp_input_data.amps_test)
                    trans_act_func_mask = strcmpi(amp_output_data.trans_act_func,amp_input_data.trans_acts_test{i_trans_act})==1;
                    amp_mask = amp_output_data.amp_list == amp_input_data.amps_test(i_amp); 
                    act_fact_mask = amp_output_data.dir_act_fact == amp_input_data.dir_act_fact(i_dir_act);
                    
        %             neigh_mask = amp_output_data.neighbor_similarity > prctile(amp_output_data.neighbor_similarity,75);
                    mask = trans_act_func_mask & amp_mask & act_fact_mask; % & neigh_mask;

                    subplot(3,2,(i_row-1)*2 + 1); hold on; % plot scatter plot
                    plot(rad2deg(stim_PD(mask)), rad2deg(ang(mask)),'o','color',color_list(i_amp,:),'markersize',4);
                    
                    subplot(3,2,(i_row-1)*2 + 2); hold on; % plot histogram of diff
                    h=histogram(rad2deg(angleDiff(stim_PD(mask),ang(mask),1,0)),bin_edges,'Normalization','Probability',...
                        'edgecolor',color_list(i_amp,:),'FaceColor','none','linewidth',2);
                    
                    if(i_row==1) % only make this plot once, but do it for all amps
                        subplot(3,2,5); hold on;% scatter plot showing elbow and hand ang diff
                        plot(rad2deg(hand_ang(mask)),rad2deg(elbow_ang(mask)),'o','color',color_list(i_amp,:),'markersize',4)
                        plot([-180,180],[-180,180],'k--','linewidth',1.5);
                        xlabel('Hand dir (deg)'); ylabel('Elbow dir (deg)');
                        set(gca,'fontsize',14); formatForLee(gcf);
                        
                        subplot(3,2,6); hold on;
                        h=histogram(rad2deg(angleDiff(hand_ang(mask),elbow_ang(mask),1,0)),bin_edges,'Normalization','Probability',...
                            'edgecolor',color_list(i_amp,:),'FaceColor','none','linewidth',2);
                        xlabel('|Hand dir - elbow dir| (deg)');
                        ylabel('Proportion');
                        formatForLee(gcf);
                        set(gca,'fontsize',14);
                    end
                    
                end
                
                % clean up plots
                subplot(3,2,(i_row-1)*2 + 1); hold on;
                plot([-180,180],[-180,180],'k--','linewidth',1.5);
                xlabel('Stim PD (deg)');
                switch i_row
                    case 1
                        ylabel('Hand dir (deg)');
                    case 2
                        ylabel('Elbow dir (deg)');
                end
                xlim([-180,180]);
                ylim([-360, 360]);
                set(gca,'fontsize',14);
                formatForLee(gcf);
                
                subplot(3,2,(i_row-1)*2 + 2); hold on; % plot histogram of diff
                switch i_row
                    case 1
                        xlabel('|Stim PD - hand dir| (deg)');
                    case 2
                        xlabel('|Stim PD - elbow dir| (deg)');
                end
                xlabel('|Stim PD - joint dir| (deg)');
                ylabel('Proportion');
                formatForLee(gcf);
                set(gca,'fontsize',14);
                
            end
        end
    end
    
end