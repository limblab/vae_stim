function [ output_data ] = getBioModelData( input_data )
%

    output_data = {};
    % for each amplitude, get soma locations for each soma and whether it
    % was activated or not
    
    for i_amp = 1:numel(input_data.amp_list)
        % set folderpath
        folderpath = [input_data.folderpath filesep 'dataset' filesep 'Amp_' num2str(input_data.amp_list(i_amp)) 'uA'];
        soma_act = []; 
        soma_coord = [];
        for i_cell_type = 1:input_data.n_soma
            soma_loc = getTrueCoordinates(i_cell_type, folderpath);
            
            
            soma_data = load([folderpath filesep 'data_soma' num2str(i_cell_type)]);
            
            soma_act_temp = zeros(numel(soma_data.data_soma),1);
            soma_coord_temp = zeros(numel(soma_loc),3);
            
            for i_cell = 1:numel(soma_data.data_soma)
                soma_act_temp(i_cell) = ~isempty(soma_data.data_soma(i_cell).times);
                soma_coord_temp(i_cell,:) = soma_loc(i_cell).coord;
            end
            
            soma_act = [soma_act; soma_act_temp];
            soma_coord = [soma_coord; soma_coord_temp];
        end
        
        output_data{i_amp}.soma_act = soma_act;
        output_data{i_amp}.soma_coord = soma_coord;
        output_data{i_amp}.amp = input_data.amp_list(i_amp);
        
    end
    

end

