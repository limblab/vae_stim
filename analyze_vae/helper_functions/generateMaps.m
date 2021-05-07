function [output_data] = generateMaps(input_data, neigh_input_data)
% generates a bunch of PD maps where nearby neurons have similar PDs. Uses
% a genetic algorithm to do so
% input_data contains:
% n_iters : number of generations to do
% pop_size : size of population
% locs : location of neurons
% PDs : PDs of neurons
% prop_new : proportion of new maps for each generation
% bin edges : bin edges (in degrees) for neighborhood function
% nbor_prop : desired neighbor proportion in bin edges
% non_nbor_prop : desired non neighbor proportion in bin edges


% neigh_input_data contains the necessary items for getNeighborMetric call

    % initialize population
    % represent map as a vector of numbers from 1:num_neurons
    n_neurons = size(input_data.locs,1);
    pop = repmat([1:1:n_neurons],input_data.pop_size,1);
    for i_pop = 1:input_data.pop_size
        pop(i_pop,:) = pop(i_pop,randperm(n_neurons));
    end
    
    % useful variables
    weights = 1./(1:1:input_data.pop_size);
    n_rand = floor(input_data.pop_size*input_data.prop_new);
    n_top = floor(input_data.pop_size*input_data.prop_top);
    
    fit_list = zeros(input_data.n_iters,3); % populated at end of for loop
    pop_keep = zeros(1000,n_neurons);
    pop_keep_idx = 1;
    
    % for each iteration
    for i_iter = 1:input_data.n_iters
        % measure fitness of each member
        fitness = getFitness(pop,input_data, neigh_input_data);
        
        % store any well-performing maps
        store_mask = fitness < input_data.max_fit;
        num_store = sum(store_mask);
        if(size(pop_keep,1)*0.67 < pop_keep_idx + num_store) % expand pop keep if necessary
            pop_keep = [pop_keep; zeros(500, n_neurons)];
        end
        pop_keep(pop_keep_idx:pop_keep_idx+num_store-1,:) = pop(store_mask,:);
        pop_keep_idx = pop_keep_idx + num_store;
        
        % select fittest members for breeding. Low fitness is better
        % select based on fitness
        [fitness,sort_idx] = sort(fitness,'ascend');
        pop = pop(sort_idx,:);
        
        pop_best = pop(1:n_top,:);
        
        new_pop_idx = datasample(1:1:input_data.pop_size,input_data.pop_size,...
            'Weights',weights,'Replace',true);
        pop = pop(new_pop_idx,:);
        
        % create new random members to replace some in pop randomly
        idx_rand = datasample(1:1:input_data.pop_size, n_rand,'Replace',false);
        for i_pop = idx_rand
            pop(i_pop,:) = pop(i_pop,randperm(n_neurons));
        end
        
        % combine members by combining well performing parts of each map --
        % make two offspring per couple
        pop = generateOffspring(pop,input_data.locs);
        
        % mutate new members
        pop = mutateMaps(pop);

        % retain top pop
        pop(1:n_top,:) = pop_best;
        
        % repeat
        fit_list(i_iter,:) = [min(fitness), prctile(fitness,10), prctile(fitness,25)];
        
        if(mod(i_iter,10) == 0)
            disp(i_iter); 
            disp(fit_list(i_iter,:));
        end
    end

    
    output_data.pop_keep = pop_keep(1:pop_keep_idx-1,:);
    output_data.fit_list = fit_list;
    output_data.curr_pop = pop;
    output_data.curr_fit = fitness;
end


function [new_pop] = generateOffspring(pop,locs)
    
    new_pop = pop;
    pop_size = size(pop,1);
    
    for i_pop = 1:pop_size 
        % pick two maps
        swap_idx = datasample(1:1:pop_size,2,'Replace',false);
        % combine maps
        new_pop(i_pop,:) = combineMaps(pop(swap_idx(1),:), pop(swap_idx(2),:),locs);
    end

end

function [map_c] = combineMaps(map1,map2,locs)

    map_c = map1;
    % split map into N groups and get neurons in each group for each map
    n_x_groups = 3;
    n_y_groups = 3;
    
    min_x = min(locs(:,1));
    max_x = max(locs(:,1));
    min_y = min(locs(:,2));
    max_y = max(locs(:,2));
    
    x_bounds = [min_x,...
        sort(ceil(rand(1,n_x_groups-1)*(max_x-min_x-1)+min_x)),...
        max_x+1];
    
    y_bounds = [min(locs(:,1)),...
        sort(ceil(rand(1,n_y_groups-1)*(max_y-min_y-1)+min_y)),...
        max(locs(:,2))+1];
    
    % combine groups to make new map
    for i_x = 1:n_x_groups
        for i_y = 1:n_y_groups                
            if(rand() < 0.6)
                temp_locs = locs(map1,:);
                group_mask = temp_locs(:,1) >= x_bounds(i_x) & temp_locs(:,1) < x_bounds(i_x+1) & temp_locs(:,2) >= y_bounds(i_y) & temp_locs(:,2) < y_bounds(i_y+1);
                map_c(group_mask) = map1(group_mask);
            else
                temp_locs = locs(map2,:);
                group_mask = locs(:,1) >= x_bounds(i_x) & temp_locs(:,1) < x_bounds(i_x+1) & temp_locs(:,2) >= y_bounds(i_y) & temp_locs(:,2) < y_bounds(i_y+1);
                map_c(group_mask) = map2(group_mask);
            end
        end
    end
        
    
    % find left out neurons and randomly replace duplicates
    neuron_set = 1:1:numel(map1);
    neuron_diff = setdiff(neuron_set,unique(map_c));
    neuron_dup = neuron_set(sum(neuron_set == map_c') ==2);
    neuron_dup = neuron_dup(randperm(numel(neuron_dup)));
    
    for i_diff = 1:numel(neuron_diff)
        idx_dup = find(map_c == neuron_dup(i_diff));
        rep_idx = idx_dup((rand()>0.5) + 1);
        map_c(rep_idx) = neuron_diff(i_diff);
    end
    
    
    
end


function [pop] = mutateMaps(pop)
    % randomly swap a number of neurons

    pop_size = size(pop,1);
    for i_pop = 1:pop_size
        n_swap = floor((rand()*0.01+0.09)*pop_size); %
        n_swap = max(1,n_swap);
        
        for i_swap = 1:n_swap
            swap_idx = datasample(1:1:pop_size,2,'Replace',false);
            pop(i_pop,swap_idx) = pop(i_pop,flip(swap_idx));
        end
        
    end
    

end



function [scores] = getFitness(pop, input_data, neigh_data)

    scores = zeros(input_data.pop_size,1);
    
    
    for i_pop = 1:input_data.pop_size
        % get neighborhood data
        neigh_data.locs = input_data.locs(pop(i_pop,:),:);
        neigh_data.metric = input_data.PDs;
        nbor_output = getNeighborMetric(neigh_data);
        
        nbor_prop = histcounts(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==1))),input_data.bin_edges,'Normalization','probability');
        non_nbor_prop = histcounts(rad2deg(abs(nbor_output.diff(nbor_output.is_neigh==0))),input_data.bin_edges,'Normalization','probability');
        
        % score is distance from input_data proportions. low score is
        % better. Weigh neighbor distribution more heavily.
        
        scores(i_pop) = sum(10*(input_data.nbor_prop - nbor_prop).^2 + (input_data.non_nbor_prop - non_nbor_prop).^2);
        
    end




end







