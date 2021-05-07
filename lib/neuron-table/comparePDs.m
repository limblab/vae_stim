function comparePDs(pdTable1,pdTable2,params,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEFAULT PARAMETERS
move_corr      =  '';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some undocumented parameters
if nargin > 1, assignParams(who,params); end % overwrite parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

errorbar(pdTable1.([move_corr 'PD']),pdTable2.([move_corr 'PD']),...
        minusPi2Pi(pdTable2.([move_corr 'PD'])-pdTable2.([move_corr 'PDCI'])(:,1)),...
        minusPi2Pi(pdTable2.([move_corr 'PDCI'])(:,2)-pdTable2.([move_corr 'PD'])),...
        minusPi2Pi(pdTable1.([move_corr 'PD'])-pdTable1.([move_corr 'PDCI'])(:,1)),...
        minusPi2Pi(pdTable1.([move_corr 'PDCI'])(:,2)-pdTable1.([move_corr 'PD'])),...
        varargin{:})
hold on
plot([-pi pi],[-pi pi],'--k','linewidth',2)
plot([-pi pi],[0 0],'-k','linewidth',2)
plot([0 0],[-pi pi],'-k','linewidth',2)
set(gca,'box','off','tickdir','out','xlim',[-pi pi],'ylim',[-pi pi],'xtick',[-pi pi],'ytick',[-pi pi],'xticklabel',{'-\pi','\pi'},'yticklabel',{'-\pi','\pi'})
axis equal
