function [] = plotPDMap(ax1,velPD, map_dim, locs, idx)

    pd_map = rad2deg(reshape(velPD,map_dim));
    pd_map(pd_map<0) = pd_map(pd_map<0)+360; % convert to same limits as polarhistogram
    imagesc(pd_map);
    colormap(ax1,colorcet('C3'));
    
    b=colorbar();
    b.Label.String = 'PD (degrees)';
    b.Label.FontSize = 14;
    
    % plot stim location
    loc_idx = locs(idx,:); 
    rectangle('Position',[loc_idx(2)-0.5,loc_idx(1)-0.5,1,1],'faceColor','none','EdgeColor','m','linewidth',3);


end