clear all
clear all
close all

% cell_id=1:5 %L1 NGC-DA, clones 1-5
% cell_id=6:10 %L23 PC, clones 1-5
% cell_id=11:15 %L4 LBC, clones 1-5
% cell_id=16:20 %L5 PC, clones 1-5
% cell_id=21:25 %L6 PC, clones 1-5

cell_id=16; %L5 PC, Clone-1

load('realx.dat') %x-coordinate (in um) of soma within cortical column
load('realy.dat') %y-coordinate (in um) of soma within cortical column
load('realz.dat') %z-coordinate (in um) of soma within cortical column

load('realang.dat') %Angle (in radian) of random rotation of neuron in azimuthal direction 

load('cell_cnt.dat') %Neuron count across each cell type and each clone

intx=load(['intx_' num2str(cell_id) '.dat']); %x-coordinates (in um) of all neural section (dendrite, soma and axon) of a single neuron 
inty=load(['inty_' num2str(cell_id) '.dat']); %y-coordinates (in um) of all neural section (dendrite, soma and axon) of a single neuron
intz=load(['intz_' num2str(cell_id) '.dat']); %z-coordinates (in um) of all neural section (dendrite, soma and axon) of a single neuron

soma_coord=load(['soma_coord_' num2str(cell_id) '.dat']); %x,y,z coordinates (in um) of somatic section (of a single neuron) 


%To get true location of somatic section of each neuron within the cortical
%column
soma = [];
for k=1:cell_cnt(cell_id)
    
    net_ad_x=((realx(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(1))*cos(realang(sum(cell_cnt(1:cell_id-1))+k)))-((realz(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(3))*sin(realang(sum(cell_cnt(1:cell_id-1))+k)))-realx(sum(cell_cnt(1:cell_id-1))+k);
    net_ad_y=realy(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(2)-realy(sum(cell_cnt(1:cell_id-1))+k);
    net_ad_z=((realx(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(1))*sin(realang(sum(cell_cnt(1:cell_id-1))+k)))+((realz(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(3))*cos(realang(sum(cell_cnt(1:cell_id-1))+k)))-realz(sum(cell_cnt(1:cell_id-1))+k);
        
    x1=((realx(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(1))*cos(realang(sum(cell_cnt(1:cell_id-1))+k)))-((realz(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(3))*sin(realang(sum(cell_cnt(1:cell_id-1))+k)));
    y1=realy(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(2);
    z1=((realx(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(1))*sin(realang(sum(cell_cnt(1:cell_id-1))+k)))+((realz(sum(cell_cnt(1:cell_id-1))+k)+soma_coord(3))*cos(realang(sum(cell_cnt(1:cell_id-1))+k)));

    soma(k).coord(1)=x1-net_ad_x;
    soma(k).coord(2)=y1-net_ad_y;
    soma(k).coord(3)=z1-net_ad_z;
    soma(k).meta = 'xyz';

end




