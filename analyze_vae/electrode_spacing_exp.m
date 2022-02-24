means_plot = [80,77.7,67.1,61.3,60.6];
stds_plot = [41.8,56.7,45.8,50.5,53.2];

elec_space = [12,8,4,2,1]*50; % 16 means only used a single neuron to predict


figure();

errorbar(elec_space,means_plot,stds_plot,'linestyle','none','marker','s','color','k','markerfacecolor','k','markersize',12,'linewidth',1.5)

xlim([0,13]*50)
formatForLee(gcf);
ylabel('Angular Error (deg)');
xlabel('Electrode spacing (\mum)');
set(gca,'fontsize',14)