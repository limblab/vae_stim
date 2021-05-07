function RDPSaveFig(H,targetDirectory)
    %takes a figure handle and saves it into the runDataProcessing folder
    %heierarchy. takes a single figure handle and the target directory of
    %the RDP script
    
    fname=get(H,'Name');
    if isempty(fname)
        fname=strcat('Figure_',num2str(double(H)));
    end
    fname(fname==' ')='_';%replace spaces in name for saving
    print('-dpdf',H,strcat(targetDirectory,['Raw_Figures' filesep 'PDF' filesep],fname,'.pdf'))
    print('-deps',H,strcat(targetDirectory,['Raw_Figures' filesep 'EPS' filesep],fname,'.eps'))
    print('-dpng',H,strcat(targetDirectory,['Raw_Figures' filesep 'PNG' filesep],fname,'.png'))
    saveas(H,strcat(targetDirectory,['Raw_Figures' filesep 'FIG' filesep],fname,'.fig'),'fig')
    
end